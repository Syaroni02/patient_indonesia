from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("SVM_patient_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Get values from form in the correct order
            values = [
                float(request.form["HAEMATOCRIT_status"]),
                float(request.form["HAEMOGLOBINS_status"]),
                float(request.form["ERYTHROCYTE_status"]),
                float(request.form["LEUCOCYTE_status"]),
                float(request.form["THROMBOCYTE_status"]),
                float(request.form["MCH_status"]),
                float(request.form["MCHC_status"]),
                float(request.form["MCV_status"])
            ]
            X = np.array(values).reshape(1, -1)
            pred = model.predict(X)[0]
            prediction = "Inpatient" if pred == 1 else "Outpatient"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
