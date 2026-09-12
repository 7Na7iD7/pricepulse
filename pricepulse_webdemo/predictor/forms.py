from django import forms


class PhoneSpecsForm(forms.Form):
    battery_power = forms.IntegerField(
        label="ظرفیت باتری (mAh)", min_value=500, max_value=2000, initial=1500,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    blue = forms.TypedChoiceField(
        label="بلوتوث دارد؟", choices=[(1, "بله"), (0, "خیر")], coerce=int, initial=1,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    clock_speed = forms.FloatField(
        label="سرعت پردازنده (GHz)", min_value=0.5, max_value=3.0, initial=2.0,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )
    dual_sim = forms.TypedChoiceField(
        label="دو سیم‌کارت؟", choices=[(1, "بله"), (0, "خیر")], coerce=int, initial=1,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    fc = forms.IntegerField(
        label="دوربین جلو (مگاپیکسل)", min_value=0, max_value=20, initial=5,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    four_g = forms.TypedChoiceField(
        label="۴G دارد؟", choices=[(1, "بله"), (0, "خیر")], coerce=int, initial=1,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    int_memory = forms.IntegerField(
        label="حافظه‌ی داخلی (GB)", min_value=2, max_value=64, initial=32,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    m_dep = forms.FloatField(
        label="ضخامت گوشی (cm)", min_value=0.1, max_value=1.0, initial=0.5,
        widget=forms.NumberInput(attrs={"class": "form-control", "step": "0.1"}),
    )
    mobile_wt = forms.IntegerField(
        label="وزن گوشی (گرم)", min_value=80, max_value=200, initial=150,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    n_cores = forms.IntegerField(
        label="تعداد هسته‌ی پردازنده", min_value=1, max_value=8, initial=4,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    pc = forms.IntegerField(
        label="دوربین اصلی (مگاپیکسل)", min_value=0, max_value=20, initial=10,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    px_height = forms.IntegerField(
        label="ارتفاع رزولوشن (پیکسل)", min_value=0, max_value=1960, initial=800,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    px_width = forms.IntegerField(
        label="عرض رزولوشن (پیکسل)", min_value=500, max_value=1998, initial=1200,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    ram = forms.IntegerField(
        label="حافظه‌ی RAM (مگابایت)", min_value=256, max_value=3998, initial=3500,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    sc_h = forms.FloatField(
        label="ارتفاع صفحه‌نمایش (cm)", min_value=5, max_value=19, initial=12,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    sc_w = forms.FloatField(
        label="عرض صفحه‌نمایش (cm)", min_value=0, max_value=18, initial=7,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    talk_time = forms.IntegerField(
        label="زمان مکالمه (ساعت)", min_value=2, max_value=20, initial=15,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    three_g = forms.TypedChoiceField(
        label="۳G دارد؟", choices=[(1, "بله"), (0, "خیر")], coerce=int, initial=1,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    touch_screen = forms.TypedChoiceField(
        label="صفحه‌ی لمسی دارد؟", choices=[(1, "بله"), (0, "خیر")], coerce=int, initial=1,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
    wifi = forms.TypedChoiceField(
        label="وای‌فای دارد؟", choices=[(1, "بله"), (0, "خیر")], coerce=int, initial=1,
        widget=forms.Select(attrs={"class": "form-control"}),
    )
