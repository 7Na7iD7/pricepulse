# PricePulse Web Demo (Django)

یک رابط کاربری وب ساده برای دموی زنده‌ی سرویس PricePulse. این پروژه **مدل را خودش لود نمی‌کند** — فقط فرم می‌گیرد و به سرویس FastAPI اصلی (پروژه‌ی `pricepulse`) درخواست می‌فرستد. این جداسازی (UI جدا از سرویس‌دهی مدل) معماری استاندارد است: مدل یک‌جا نگه‌داری و نسخه‌بندی می‌شود و هر تعداد کلاینت (وب، موبایل، ...) می‌توانند به همان API متصل شوند.

## پیش‌نیاز

سرویس PricePulse API باید از قبل در حال اجرا باشد:

```bash
cd ../pricepulse
uvicorn serving.api:app --reload
```

## اجرا

```bash
cd pricepulse_webdemo
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

سپس مرورگر را باز کن: `http://localhost:8000`

اگر FastAPI روی آدرس یا پورت دیگری اجرا می‌شود:

```bash
export PRICEPULSE_API_URL=http://localhost:8001
python manage.py runserver
```

## اجرای تست‌ها

```bash
python manage.py test predictor -v 2
```

۵ تست شامل: رندر فرم، پیش‌بینی موفق (همراه با توضیح SHAP)، پیام خطای اتصال، پیام خطای مدل بارگذاری‌نشده، و اعتبارسنجی ورودی نامعتبر.

## توضیح‌پذیری (SHAP)

بعد از هر پیش‌بینی موفق، صفحه به‌صورت خودکار درخواست دوم به `/predict/explain` می‌فرستد و یک بخش «چرا مدل این تصمیم را گرفت؟» با ۵ ویژگی مؤثر (نوار سبز برای تأثیر مثبت، قرمز برای منفی) نمایش می‌دهد. اگر این درخواست ناموفق باشد (مثلاً به‌خاطر کندی SHAP)، پیش‌بینی اصلی همچنان نمایش داده می‌شود — فقط بخش توضیح غایب خواهد بود.

## ساختار

```
pricepulse_webdemo/
├── core/              # تنظیمات Django
├── predictor/
│   ├── forms.py       # فرم مطابق PhoneSpecs در API
│   ├── views.py       # فراخوانی HTTP به FastAPI
│   ├── tests.py       # تست‌های واقعی (با mock روی لایه‌ی شبکه)
│   └── templates/
└── manage.py
```
