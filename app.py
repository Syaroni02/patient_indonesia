from flask import Flask, request, render_template
import joblib
import numpy as np
import os

# Base path (PythonAnywhere uses the working directory)
base_path = os.path.abspath(".")

app = Flask(
    __name__,
    template_folder=os.path.join(base_path, "templates"),
    static_folder=os.path.join(base_path, "static")
)

# Load trained model
model_path = os.path.join(base_path, "SVM_patient_model.pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    print("Error loading model:", e)

# Reference limits for validation
limits = {
    "HAEMATOCRIT": [
        {"age_min": 0, "age_max": 1, "range": (45, 62)},
        {"age_min": 1, "age_max": 2, "range": (30, 44)},
        {"age_min": 2, "age_max": 6, "range": (32, 41)},
        {"age_min": 7, "age_max": 12, "range": (36, 42)},
        {"age_min": 13, "age_max": 200, "range": (30, 44)},
    ],
    "HAEMOGLOBINS": {"M": (12.5, 16.7), "F": (12, 15.6)},
    "ERYTHROCYTE": {"M": (4.1, 6.0), "F": (4.0, 5.3)},
    "LEUCOCYTE": {"M": (4.56, 10.3), "F": (4.56, 10.3)},
    "THROMBOCYTE": {"M": (159, 391), "F": (159, 391)},
    "MCH": {"M": (28, 32), "F": (28, 32)},
    "MCHC": {"M": (33, 37), "F": (33, 37)},
    "MCV": {"M": (75, 96), "F": (75, 96)},
}


# ---------------------------------------------
# Validation Function
# ---------------------------------------------
def validate_input(values, age, gender):
    warnings = []
    gender = gender.upper()

    # HAEMATOCRIT → age-based
    haematocrit_val = values["HAEMATOCRIT_status"]

    for limit in limits["HAEMATOCRIT"]:
        if limit["age_min"] <= age < limit["age_max"]:
            low, high = limit["range"]
            if not (low <= haematocrit_val <= high):
                warnings.append(f"HAEMATOCRIT out of range ({low}-{high})")
            break

    # Gender-based parameters
    for key in [
        "HAEMOGLOBINS_status", "ERYTHROCYTE_status", "LEUCOCYTE_status",
        "THROMBOCYTE_status", "MCH_status", "MCHC_status", "MCV_status"
    ]:
        base = key.replace("_status", "").upper()
        val = values[key]

        if base in limits:
            low, high = limits[base][gender]
            if not (low <= val <= high):
                warnings.append(f"{base} out of range ({low}-{high})")

    return warnings


# ---------------------------------------------
# Routes
# ---------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", warnings=[], prediction=None)


@app.route("/check_warning", methods=["POST"])
def check_warning():
    try:
        age = float(request.form.get("age", 30))
        gender = request.form.get("gender", "M")

        values = {key: float(request.form[key]) for key in request.form if "_status" in key}

        warnings = validate_input(values, age, gender)

    except Exception as e:
        warnings = [f"Error: {str(e)}"]

    return render_template("index.html", warnings=warnings, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    warnings = []
    prediction = None

    try:
        if model is None:
            return render_template("index.html", warnings=["Model not loaded"], prediction=None)

        values = {
            "HAEMATOCRIT_status": float(request.form["HAEMATOCRIT_status"]),
            "HAEMOGLOBINS_status": float(request.form["HAEMOGLOBINS_status"]),
            "ERYTHROCYTE_status": float(request.form["ERYTHROCYTE_status"]),
            "LEUCOCYTE_status": float(request.form["LEUCOCYTE_status"]),
            "THROMBOCYTE_status": float(request.form["THROMBOCYTE_status"]),
            "MCH_status": float(request.form["MCH_status"]),
            "MCHC_status": float(request.form["MCHC_status"]),
            "MCV_status": float(request.form["MCV_status"]),
        }

        X = np.array(list(values.values())).reshape(1, -1)
        pred = model.predict(X)[0]
        prediction = "Inpatient" if pred == 1 else "Outpatient"

    except Exception as e:
        prediction = f"Error: {str(e)}"

    return render_template("index.html", warnings=warnings, prediction=prediction)


# DO NOT USE app.run() on PythonAnywhere
if __name__ == "__main__":
    app.run(debug=True)
