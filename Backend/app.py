from flask import Flask, render_template, request, redirect, session, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "secret123"

DATA_PATH = "final_dataset.csv"

model = joblib.load("deterioration_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# ================= LOGIN =================
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    role = request.form["role"]

    if role == "admin":
        session["user"] = "Admin"
        return redirect("/admin")

    elif role == "doctor":
        session["user"] = "Doctor"
        return redirect("/doctor")

    return "Invalid login"

# ================= ADMIN =================
@app.route("/admin")
def admin():
    df = pd.read_csv(DATA_PATH)

    if "patient_id" not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "patient_id"}, inplace=True)
        df.to_csv(DATA_PATH, index=False)

    return render_template("admin.html",
        patient_ids=df["patient_id"].tolist(),
        name=session.get("user")
    )

# ================= PATIENT DETAILS =================
@app.route("/patient/<int:id>")
def patient_detail(id):
    df = pd.read_csv(DATA_PATH)
    row = df[df["patient_id"] == id].iloc[0]

    input_data = np.array([row[feature_names]])
    input_scaled = scaler.transform(input_data)

    prob = float(model.predict_proba(input_scaled)[0][1])

    return render_template("patient_detail.html",
        patient=row.to_dict(),
        risk=round(prob*100,2)
    )

# ================= DOCTOR =================
@app.route("/doctor")
def doctor():
    return render_template("doctor.html", name=session.get("user"))

# ================= API =================
@app.route("/api/unstable_patients")
def unstable_patients():

    df = pd.read_csv(DATA_PATH)

    # 🔥 RANDOM DATA EACH TIME
    df = df.sample(200)

    results = []

    for _, row in df.iterrows():
        try:
            input_data = np.array([row[feature_names]])
            input_scaled = scaler.transform(input_data)

            prob = float(model.predict_proba(input_scaled)[0][1])

            # 🔥 BETTER THRESHOLDS
            if prob > 0.4:
                status = "CRITICAL"
            elif prob > 0.25:
                status = "UNSTABLE"
            elif prob > 0.1:
                status = "WATCHING"
            else:
                status = "STABLE"

            results.append({
                "id": int(row["patient_id"]),
                "status": status,
                "heart_rate": float(row.get("heart_rate_x", 0)),
                "spo2": float(row.get("spo2_pct_x", 0)),
                "risk_score": float(round(prob * 100, 2))
            })

        except:
            continue

    # 🔥 SORT BY HIGH RISK
    results = sorted(results, key=lambda x: x["risk_score"], reverse=True)

    return jsonify(results[:10])   # send top 10

# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)