from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = "smart-farming-secret-key"

# Load trained models once during startup.
crop_model = joblib.load("crop_model.pkl")
soil_model = joblib.load("soil_health_model.pkl")


@dataclass
class PredictionResult:
    soil_health: str
    crop_name: str
    fertilizer_suggestions: List[str]


def fertilizer_suggestion(n: float, p: float, k: float) -> List[str]:
    suggestions: List[str] = []

    if n < 40:
        suggestions.append("Nitrogen is low: use Urea.")
    elif n > 80:
        suggestions.append("Nitrogen is high: avoid extra Urea.")

    if p < 30:
        suggestions.append("Phosphorus is low: use DAP.")
    elif p > 60:
        suggestions.append("Phosphorus is high: avoid extra DAP.")

    if k < 30:
        suggestions.append("Potassium is low: use MOP.")
    elif k > 60:
        suggestions.append("Potassium is high: avoid extra MOP.")

    if not suggestions:
        suggestions.append("NPK levels are balanced.")

    return suggestions


def run_predictions(form: dict) -> PredictionResult:
    n = float(form["nitrogen"])
    p = float(form["phosphorus"])
    k = float(form["potassium"])
    temperature = float(form["temperature"])
    humidity = float(form["humidity"])
    ph = float(form["ph"])
    rainfall = float(form["rainfall"])
    moisture = float(form["moisture"])
    organic_carbon = float(form["organic_carbon"])

    crop_data = pd.DataFrame(
        [
            {
                "N": n,
                "P": p,
                "K": k,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            }
        ]
    )

    soil_data = pd.DataFrame(
        [
            {
                "ph": ph,
                "nitrogen": n,
                "phosphorus": p,
                "potassium": k,
                "moisture": moisture,
                "organic_carbon": organic_carbon,
            }
        ]
    )

    crop_prediction = str(crop_model.predict(crop_data)[0])
    soil_prediction = str(soil_model.predict(soil_data)[0])
    fertilizer_prediction = fertilizer_suggestion(n, p, k)

    return PredictionResult(
        soil_health=soil_prediction,
        crop_name=crop_prediction,
        fertilizer_suggestions=fertilizer_prediction,
    )


def default_form_values() -> Dict[str, str]:
    return {
        "nitrogen": "",
        "phosphorus": "",
        "potassium": "",
        "temperature": "",
        "humidity": "",
        "ph": "",
        "rainfall": "",
        "moisture": "",
        "organic_carbon": "",
    }


def user_logged_in() -> bool:
    return bool(session.get("is_logged_in"))


@app.route("/", methods=["GET", "POST"])
def auth():
    if user_logged_in():
        return redirect(url_for("dashboard"))

    error_message = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if email and password:
            session["is_logged_in"] = True
            session["user_name"] = email.split("@")[0].title()
            return redirect(url_for("dashboard"))
        error_message = "Please enter both email and password."

    return render_template("auth.html", error_message=error_message)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth"))


@app.route("/dashboard")
def dashboard():
    if not user_logged_in():
        return redirect(url_for("auth"))
    return render_template("dashboard.html", user_name=session.get("user_name", "Farmer"))


@app.route("/soil-analysis", methods=["GET", "POST"])
def soil_analysis():
    if not user_logged_in():
        return redirect(url_for("auth"))

    form_values = session.get("last_input", default_form_values())
    error_message = ""
    prediction = session.get("prediction_result")

    if request.method == "POST":
        form_values = default_form_values()
        form_values.update(request.form.to_dict())
        try:
            result = run_predictions(request.form)
            prediction = asdict(result)
            session["last_input"] = form_values
            session["prediction_result"] = prediction
            return redirect(url_for("soil_analysis"))
        except Exception as exc:
            error_message = f"Could not run prediction: {exc}"

    return render_template(
        "soil_analysis.html",
        form_values=form_values,
        prediction=prediction,
        error_message=error_message,
    )


@app.route("/crop-recommendation")
def crop_recommendation():
    if not user_logged_in():
        return redirect(url_for("auth"))
    prediction = session.get("prediction_result")
    return render_template("crop_recommendation.html", prediction=prediction)


@app.route("/fertilizer-suggestion")
def fertilizer():
    if not user_logged_in():
        return redirect(url_for("auth"))
    prediction = session.get("prediction_result")
    npk = session.get("last_input", {})
    return render_template("fertilizer.html", prediction=prediction, npk=npk)


@app.route("/weather-forecast")
def weather():
    if not user_logged_in():
        return redirect(url_for("auth"))

    city = request.args.get("city", "Pune, Maharashtra")
    weather_data = {
        "city": city,
        "temperature": "28°C",
        "condition": "Partly Cloudy",
        "humidity": "60%",
        "wind": "10 km/h",
        "rainfall": "0 mm",
        "forecast": [
            {"day": "Today", "temp": "28° / 18°"},
            {"day": "Tomorrow", "temp": "27° / 17°"},
            {"day": "Fri", "temp": "26° / 16°"},
            {"day": "Sat", "temp": "27° / 17°"},
        ],
    }
    return render_template("weather.html", weather=weather_data)


if __name__ == "__main__":
    # Listen on all interfaces so it works reliably in browser/network testing.
    app.run(host="0.0.0.0", port=5000, debug=True)
