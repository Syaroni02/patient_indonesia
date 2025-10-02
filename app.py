# from flask import Flask, request, render_template
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load trained model
# model = joblib.load("SVM_patient_model.pkl")

# @app.route("/", methods=["GET", "POST"])
# def home():
#     prediction = None
#     if request.method == "POST":
#         try:
#             # Get values from form in the correct order
#             values = [
#                 float(request.form["HAEMATOCRIT_status"]),
#                 float(request.form["HAEMOGLOBINS_status"]),
#                 float(request.form["ERYTHROCYTE_status"]),
#                 float(request.form["LEUCOCYTE_status"]),
#                 float(request.form["THROMBOCYTE_status"]),
#                 float(request.form["MCH_status"]),
#                 float(request.form["MCHC_status"]),
#                 float(request.form["MCV_status"])
#             ]
#             X = np.array(values).reshape(1, -1)
#             pred = model.predict(X)[0]
#             prediction = "Inpatient" if pred == 1 else "Outpatient"
#         except Exception as e:
#             prediction = f"Error: {str(e)}"
#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True, use_reloader=False)


from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("SVM_patient_model.pkl")

# Reference limits for validation
limits = {
    "HAEMATOCRIT": [
        {"age_min": 0, "age_max": 1, "range": (45, 62)},
        {"age_min": 1, "age_max": 2, "range": (30, 44)},
        {"age_min": 2, "age_max": 6, "range": (32, 41)},
        {"age_min": 7, "age_max": 12, "range": (36, 42)},
        {"age_min": 13, "age_max": 200, "range": (30, 44)}
    ],
    "HAEMOGLOBINS": {"M": (12.5, 16.7), "F": (12, 15.6)},
    "ERYTHROCYTE": {"M": (4.1, 6.0), "F": (4.0, 5.3)},
    "LEUCOCYTE": {"M": (4.56, 10.3), "F": (4.56, 10.3)},
    "THROMBOCYTE": {"M": (159, 391), "F": (159, 391)},
    "MCH": {"M": (28, 32), "F": (28, 32)},
    "MCHC": {"M": (33, 37), "F": (33, 37)},
    "MCV": {"M": (75, 96), "F": (75, 96)},
}

# Function to validate values
def validate_input(values, age, gender):
    warnings = []

    # HAEMATOCRIT uses age ranges
    haematocrit_val = values["HAEMATOCRIT_status"]
    for limit in limits["HAEMATOCRIT"]:
        if limit["age_min"] <= age < limit["age_max"]:
            low, high = limit["range"]
            if not (low <= haematocrit_val <= high):
                warnings.append(f"HAEMATOCRIT out of normal range ({low}-{high})")
            break

    # Other parameters use gender
    for key in ["HAEMOGLOBINS_status", "ERYTHROCYTE_status", "LEUCOCYTE_status",
                "THROMBOCYTE_status", "MCH_status", "MCHC_status", "MCV_status"]:
        param_name = key.replace("_status", "").upper()
        val = values[key]
        if param_name in limits:
            low, high = limits[param_name][gender.upper()]
            if not (low <= val <= high):
                warnings.append(f"{param_name} out of normal range ({low}-{high})")

    return warnings

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    warnings = []
    if request.method == "POST":
        try:
            # Example: get age and gender from form
            age = float(request.form.get("age", 30))  # default 30
            gender = request.form.get("gender", "M")  # default Male

            # Collect lab values
            values = {
                "HAEMATOCRIT_status": float(request.form["HAEMATOCRIT_status"]),
                "HAEMOGLOBINS_status": float(request.form["HAEMOGLOBINS_status"]),
                "ERYTHROCYTE_status": float(request.form["ERYTHROCYTE_status"]),
                "LEUCOCYTE_status": float(request.form["LEUCOCYTE_status"]),
                "THROMBOCYTE_status": float(request.form["THROMBOCYTE_status"]),
                "MCH_status": float(request.form["MCH_status"]),
                "MCHC_status": float(request.form["MCHC_status"]),
                "MCV_status": float(request.form["MCV_status"])
            }

            # Validate inputs
            warnings = validate_input(values, age, gender)

            # Make prediction
            X = np.array(list(values.values())).reshape(1, -1)
            pred = model.predict(X)[0]
            prediction = "Inpatient" if pred == 1 else "Outpatient"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, warnings=warnings)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

