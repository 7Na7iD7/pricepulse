import logging

import requests
from django.conf import settings
from django.shortcuts import render

from .forms import PhoneSpecsForm

logger = logging.getLogger("pricepulse.webdemo")

FEATURE_LABELS_FA = {
    "battery_power": "ظرفیت باتری", "blue": "بلوتوث", "clock_speed": "سرعت پردازنده",
    "dual_sim": "دو سیم‌کارت", "fc": "دوربین جلو", "four_g": "۴G",
    "int_memory": "حافظه‌ی داخلی", "m_dep": "ضخامت گوشی", "mobile_wt": "وزن گوشی",
    "n_cores": "تعداد هسته", "pc": "دوربین اصلی", "px_height": "ارتفاع رزولوشن",
    "px_width": "عرض رزولوشن", "ram": "حافظه‌ی RAM", "sc_h": "ارتفاع صفحه‌نمایش",
    "sc_w": "عرض صفحه‌نمایش", "talk_time": "زمان مکالمه", "three_g": "۳G",
    "touch_screen": "صفحه‌ی لمسی", "wifi": "وای‌فای",
}


def index(request):
    result = None
    error = None
    explanation = None

    if request.method == "POST":
        form = PhoneSpecsForm(request.POST)
        if form.is_valid():
            payload = form.cleaned_data
            try:
                response = requests.post(
                    f"{settings.PRICEPULSE_API_URL}/predict",
                    json=payload,
                    timeout=5,
                )
                if response.status_code == 200:
                    result = response.json()
                    result["confidence"] = result["confidence"] * 100
                    result["class_probabilities"] = {
                        k: v * 100 for k, v in result["class_probabilities"].items()
                    }

                    try:
                        explain_response = requests.post(
                            f"{settings.PRICEPULSE_API_URL}/predict/explain",
                            json=payload,
                            timeout=10,
                        )
                        if explain_response.status_code == 200:
                            explanation = explain_response.json()
                            max_abs = max(
                                (abs(item["contribution"]) for item in explanation["top_factors"]),
                                default=1,
                            ) or 1
                            for item in explanation["top_factors"]:
                                item["feature_fa"] = FEATURE_LABELS_FA.get(item["feature"], item["feature"])
                                item["is_positive"] = item["contribution"] >= 0
                                item["bar_width_pct"] = round(abs(item["contribution"]) / max_abs * 100, 1)
                    except requests.exceptions.RequestException:
                        logger.warning("SHAP explain request failed; showing prediction without explanation.")
                elif response.status_code == 503:
                    error = "مدل هنوز روی سرویس API بارگذاری نشده است. ابتدا pipeline.py را اجرا کنید."
                else:
                    error = f"سرویس API خطا برگرداند (کد {response.status_code})."
            except requests.exceptions.ConnectionError:
                error = (
                    "اتصال به سرویس API برقرار نشد. مطمئن شوید PricePulse API "
                    f"روی آدرس {settings.PRICEPULSE_API_URL} در حال اجراست "
                    "(uvicorn serving.api:app)."
                )
            except requests.exceptions.Timeout:
                error = "سرویس API به‌موقع پاسخ نداد (timeout)."
    else:
        form = PhoneSpecsForm()

    return render(request, "predictor/index.html", {
        "form": form,
        "result": result,
        "error": error,
        "explanation": explanation,
        "api_url": settings.PRICEPULSE_API_URL,
    })
