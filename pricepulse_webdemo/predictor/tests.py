from unittest.mock import patch

from django.test import TestCase


VALID_PAYLOAD = {
    "battery_power": 1500, "blue": 1, "clock_speed": 2.0, "dual_sim": 1, "fc": 5,
    "four_g": 1, "int_memory": 32, "m_dep": 0.5, "mobile_wt": 150, "n_cores": 4,
    "pc": 10, "px_height": 800, "px_width": 1200, "ram": 3500, "sc_h": 12, "sc_w": 7,
    "talk_time": 15, "three_g": 1, "touch_screen": 1, "wifi": 1,
}


class FakeApiResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data


class PredictorViewTests(TestCase):
    def test_get_renders_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "PricePulse")
        self.assertContains(response, "پیش‌بینی قیمت")

    def test_post_valid_data_shows_prediction(self):
        fake_predict_response = FakeApiResponse(200, {
            "price_range": 2, "price_label": "بالا", "confidence": 0.95,
            "class_probabilities": {"0": 0.01, "1": 0.02, "2": 0.95, "3": 0.02},
            "inference_ms": 5.0,
        })
        fake_explain_response = FakeApiResponse(200, {
            "price_range": 2, "price_label": "بالا",
            "top_factors": [{"feature": "ram", "contribution": 0.5}],
            "explanation_note": "توضیح تستی",
        })
        with patch("predictor.views.requests.post", side_effect=[fake_predict_response, fake_explain_response]):
            response = self.client.post("/", data=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "بالا")
        self.assertContains(response, "حافظه‌ی RAM")

    def test_post_connection_error_shows_friendly_message(self):
        import requests
        with patch("predictor.views.requests.post", side_effect=requests.exceptions.ConnectionError):
            response = self.client.post("/", data=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "اتصال به سرویس API برقرار نشد")

    def test_post_model_not_loaded_shows_friendly_message(self):
        fake_response = FakeApiResponse(503, {})
        with patch("predictor.views.requests.post", return_value=fake_response):
            response = self.client.post("/", data=VALID_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "مدل هنوز روی سرویس API بارگذاری نشده است")

    def test_post_invalid_field_shows_form_errors(self):
        bad_payload = dict(VALID_PAYLOAD)
        bad_payload["ram"] = 999999
        response = self.client.post("/", data=bad_payload)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "field-error")
        self.assertContains(response, "3998")
