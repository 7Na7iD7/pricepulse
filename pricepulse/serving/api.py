import logging
logging.getLogger("shap").setLevel(logging.WARNING)

import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pricepulse.api")

app = FastAPI(
    title="PricePulse API",
    description="سرویس پیش‌بینی هوشمند بازه‌ی قیمت موبایل",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = BASE_DIR / "artifacts" / "pricepulse_model.joblib"
_bundle = None
_shap_explainer = None

FEATURE_ORDER = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]

PRICE_LABELS = {0: "ارزان", 1: "متوسط", 2: "بالا", 3: "بسیار بالا"}


class PhoneSpecs(BaseModel):
    battery_power: float = Field(ge=500, le=2000, description="ظرفیت باتری (mAh)")
    blue: int = Field(ge=0, le=1)
    clock_speed: float = Field(ge=0.5, le=3.0)
    dual_sim: int = Field(ge=0, le=1)
    fc: float = Field(ge=0, le=20, description="مگاپیکسل دوربین جلو")
    four_g: int = Field(ge=0, le=1)
    int_memory: float = Field(ge=2, le=64, description="گیگابایت")
    m_dep: float = Field(ge=0.1, le=1.0, description="ضخامت گوشی (سانتی‌متر)")
    mobile_wt: float = Field(ge=80, le=200, description="گرم")
    n_cores: int = Field(ge=1, le=8)
    pc: float = Field(ge=0, le=20, description="مگاپیکسل دوربین اصلی")
    px_height: float = Field(ge=0, le=1960)
    px_width: float = Field(ge=500, le=1998)
    ram: float = Field(ge=256, le=3998, description="مگابایت")
    sc_h: float = Field(ge=5, le=19, description="سانتی‌متر")
    sc_w: float = Field(ge=0, le=18, description="سانتی‌متر")
    talk_time: float = Field(ge=2, le=20, description="ساعت")
    three_g: int = Field(ge=0, le=1)
    touch_screen: int = Field(ge=0, le=1)
    wifi: int = Field(ge=0, le=1)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "battery_power": 1500, "blue": 1, "clock_speed": 2.0, "dual_sim": 1,
                "fc": 5, "four_g": 1, "int_memory": 32, "m_dep": 0.5, "mobile_wt": 150,
                "n_cores": 4, "pc": 10, "px_height": 800, "px_width": 1200, "ram": 3500,
                "sc_h": 12, "sc_w": 7, "talk_time": 15, "three_g": 1, "touch_screen": 1,
                "wifi": 1,
            }
        }
    )


class PredictionResponse(BaseModel):
    price_range: int
    price_label: str
    confidence: float
    class_probabilities: dict
    inference_ms: float


class BatchRequest(BaseModel):
    items: List[PhoneSpecs] = Field(max_length=500)


class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    count: int
    total_inference_ms: float


class ExplanationItem(BaseModel):
    feature: str
    contribution: float


class ExplainResponse(BaseModel):
    price_range: int
    price_label: str
    top_factors: List[ExplanationItem]
    explanation_note: str


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    model_type: str | None = None
    metrics: dict | None = None
    feature_count: int


def _load_bundle():
    global _bundle
    if _bundle is None:
        if not ARTIFACT_PATH.exists():
            logger.error(f"مدل در {ARTIFACT_PATH} یافت نشد.")
            raise HTTPException(
                status_code=503,
                detail="مدل هنوز آموزش داده نشده یا فایل artifact موجود نیست.",
            )
        try:
            _bundle = joblib.load(ARTIFACT_PATH)
        except Exception as exc:
            logger.exception("خطا در بارگذاری مدل")
            raise HTTPException(status_code=500, detail=f"خطا در بارگذاری مدل: {exc}")
    return _bundle


def _load_shap_explainer(bundle):
    global _shap_explainer
    if _shap_explainer is None:
        train_sample_path = BASE_DIR / "mobile_price_data.csv"
        if not train_sample_path.exists():
            raise HTTPException(
                status_code=503,
                detail="داده‌ی مرجع برای SHAP (mobile_price_data.csv) یافت نشد.",
            )
        from explainability.shap_explainer import ShapExplainer

        df = pd.read_csv(train_sample_path)
        X = df[FEATURE_ORDER]
        scaler = bundle["scaler"]
        pca = bundle.get("pca")
        X_scaled = scaler.transform(X)
        X_final = pca.transform(X_scaled) if pca is not None else X_scaled

        explainer = ShapExplainer(bundle["model"], FEATURE_ORDER)
        explainer.fit(X_final[:100])
        _shap_explainer = explainer
        logger.info("SHAP explainer با موفقیت مقداردهی اولیه شد.")
    return _shap_explainer


def _predict_one(specs: PhoneSpecs, bundle: dict) -> PredictionResponse:
    start = time.perf_counter()
    model = bundle["model"]
    scaler = bundle["scaler"]
    pca = bundle.get("pca")

    raw = pd.DataFrame([[getattr(specs, f) for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    scaled = scaler.transform(raw)
    features = pca.transform(scaled) if pca is not None else scaled

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        price_range=prediction,
        price_label=PRICE_LABELS.get(prediction, "نامشخص"),
        confidence=float(max(probabilities)),
        class_probabilities={str(i): float(p) for i, p in enumerate(probabilities)},
        inference_ms=round(elapsed_ms, 3),
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.1f}ms)")
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ARTIFACT_PATH.exists()}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    import json
    bundle = _load_bundle()
    meta_path = ARTIFACT_PATH.parent / "pricepulse_metadata.json"
    metrics = None
    model_type = type(bundle["model"]).__name__
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
            metrics = meta.get("metrics")
    return ModelInfoResponse(
        model_loaded=True,
        model_type=model_type,
        metrics=metrics,
        feature_count=len(FEATURE_ORDER),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(specs: PhoneSpecs):
    bundle = _load_bundle()
    try:
        return _predict_one(specs, bundle)
    except Exception as exc:
        logger.exception("خطا در پیش‌بینی")
        raise HTTPException(status_code=500, detail=f"خطا در پردازش درخواست: {exc}")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    bundle = _load_bundle()
    start = time.perf_counter()
    try:
        results = [_predict_one(item, bundle) for item in batch.items]
    except Exception as exc:
        logger.exception("خطا در پیش‌بینی دسته‌ای")
        raise HTTPException(status_code=500, detail=f"خطا در پردازش دسته‌ای: {exc}")
    total_ms = (time.perf_counter() - start) * 1000
    return BatchResponse(results=results, count=len(results), total_inference_ms=round(total_ms, 3))


@app.post("/predict/explain", response_model=ExplainResponse)
def predict_explain(specs: PhoneSpecs):
    bundle = _load_bundle()
    try:
        explainer = _load_shap_explainer(bundle)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("خطا در مقداردهی SHAP")
        raise HTTPException(status_code=500, detail=f"خطا در آماده‌سازی توضیح‌پذیری: {exc}")

    model = bundle["model"]
    scaler = bundle["scaler"]
    pca = bundle.get("pca")

    raw = pd.DataFrame([[getattr(specs, f) for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    scaled = scaler.transform(raw)
    features = pca.transform(scaled) if pca is not None else scaled

    try:
        prediction = int(model.predict(features)[0])
        top_factors = explainer.explain_single(features, prediction, top_n=5)
    except Exception as exc:
        logger.exception("خطا در محاسبه‌ی SHAP")
        raise HTTPException(status_code=500, detail=f"خطا در محاسبه‌ی توضیح‌پذیری: {exc}")

    return ExplainResponse(
        price_range=prediction,
        price_label=PRICE_LABELS.get(prediction, "نامشخص"),
        top_factors=[ExplanationItem(**item) for item in top_factors],
        explanation_note="عدد مثبت یعنی این ویژگی مدل را به‌سمت این پیش‌بینی سوق داده؛ عدد منفی یعنی برخلاف آن اثر گذاشته است.",
    )
