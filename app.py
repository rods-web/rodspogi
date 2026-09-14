from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import joblib
import numpy as np
import json
import os
import csv
from datetime import datetime
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "change_this_to_something_secret_2025"

USERS_FILE = "users.json"
HISTORY_FILE = "history.json"
MODELS_DIR = "models"


# ============================================================
# STORAGE HELPERS
# ============================================================
def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_users():   return load_json(USERS_FILE, {})
def save_users(u):  save_json(USERS_FILE, u)
def load_history(): return load_json(HISTORY_FILE, {})
def save_history(h):save_json(HISTORY_FILE, h)


# ============================================================
# LOGIN REQUIRED
# ============================================================
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Please log in first.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


# ============================================================
# LOAD ML MODEL
# ============================================================
model = joblib.load(f"{MODELS_DIR}/enhanced_rf.pkl")
scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
feature_mask = joblib.load(f"{MODELS_DIR}/feature_mask.pkl")
feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")

comparison_data = []
comparison_path = f"{MODELS_DIR}/comparison_results.csv"
if os.path.exists(comparison_path):
    with open(comparison_path) as f:
        for row in csv.DictReader(f):
            comparison_data.append({
                "model": row["Model"],
                "accuracy": float(row["Accuracy"]),
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1"]),
                "auc": float(row["AUC"]),
            })

# ============================================================
# CHATBOT (Rule-based)
# ============================================================
CHATBOT_RESPONSES = [
    {
        "keywords": ["what", "website", "site", "purpose", "about"],
        "response": (
            "DiaPredict is a web-based diabetes risk prediction system. "
            "It uses an Enhanced Random Forest machine learning model to analyze "
            "your clinical data (like glucose, BMI, blood pressure) and estimate "
            "your risk of diabetes. It also lets you save your history and "
            "compare model performance. Note: It is not a medical diagnosis."
        )
    },
    {
        "keywords": ["diabetes", "what is", "define"],
        "response": (
            "Diabetes mellitus is a chronic metabolic disorder where blood sugar "
            "(glucose) is too high. It happens when the body either doesn't produce "
            "enough insulin or can't use insulin properly. Over time, high blood sugar "
            "damages the heart, kidneys, nerves, and eyes. Type 1, Type 2, and "
            "gestational diabetes are the main types."
        )
    },
    {
        "keywords": ["symptom", "signs", "feel"],
        "response": (
            "Common diabetes symptoms include: frequent urination, excessive thirst, "
            "unexplained weight loss, fatigue, blurred vision, slow-healing wounds, "
            "and tingling in hands/feet. Many people have no symptoms at all, which "
            "is why screening matters."
        )
    },
    {
        "keywords": ["risk", "factor", "cause"],
        "response": (
            "Key diabetes risk factors: age (45+), family history, being overweight or "
            "obese, high blood pressure, high cholesterol, physical inactivity, poor diet, "
            "and ethnicity. The PIMA dataset used here includes Glucose, BMI, Blood Pressure, "
            "Insulin, Age, Pregnancies, Skin Thickness, and Diabetes Pedigree Function."
        )
    },
    {
        "keywords": ["prevent", "avoid", "reduce", "stop"],
        "response": (
            "To reduce diabetes risk: maintain a healthy weight, exercise 30 minutes "
            "daily, eat more fiber and vegetables, limit sugar and refined carbs, "
            "avoid smoking, manage stress, and get regular checkups. Early screening "
            "is key — that's what this system helps with."
        )
    },
    {
        "keywords": ["treatment", "cure", "medicine", "insulin"],
        "response": (
            "Diabetes can be managed but not cured. Treatment includes: healthy diet, "
            "physical activity, blood sugar monitoring, oral medications (like metformin), "
            "and insulin therapy when needed. Always consult a licensed doctor."
        )
    },
    {
        "keywords": ["type", "gestational"],
        "response": (
            "Type 1 diabetes: autoimmune, body attacks insulin-producing cells — usually "
            "in children/young adults. Type 2 diabetes: body becomes insulin resistant — "
            "most common, tied to lifestyle. Gestational: occurs during pregnancy. "
            "This system primarily targets Type 2 risk."
        )
    },
    {
        "keywords": ["accuracy", "model", "random forest", "reliable"],
        "response": (
            "This system uses an Enhanced Random Forest (HFS-RF) model optimized with "
            "feature selection, SMOTE for class imbalance, and hyperparameter tuning. "
            "You can view the exact accuracy, precision, recall, F1, and AUC on the "
            "dashboard charts."
        )
    },
    {
        "keywords": ["how", "use", "predict", "start"],
        "response": (
            "Steps: 1) Log in, 2) Click 'New Prediction' in the sidebar, "
            "3) Enter the patient's clinical values (glucose, BMI, etc.), "
            "4) Click 'Generate Prediction'. You'll get a High Risk / Low Risk "
            "result with confidence. Saved automatically in History."
        )
    },
    {
        "keywords": ["history", "past", "records", "previous"],
        "response": (
            "Your prediction history is stored per account. Click 'History' "
            "in the sidebar to see all past predictions with timestamps, results, "
            "and the exact input values used. You can clear history anytime."
        )
    },
    {
        "keywords": ["privacy", "data", "secure", "safe"],
        "response": (
            "Your account data is stored locally on this server. Passwords are "
            "hashed (never stored in plain text). Predictions are only visible to you. "
            "Never share real medical data on public demos."
        )
    },
    {
        "keywords": ["doctor", "medical", "advice", "diagnos"],
        "response": (
            "DiaPredict does NOT provide medical diagnosis. It's a screening tool "
            "based on statistical patterns. Always consult a licensed physician for "
            "medical decisions or if you experience symptoms."
        )
    },
    {
        "keywords": ["hello", "hi", "hey", "good morning", "good evening"],
        "response": "Hi there! I can help with questions about diabetes, this website, or how to use the prediction system. What would you like to know?"
    },
    {
        "keywords": ["thank", "thanks", "salamat"],
        "response": "You're welcome! Let me know if you have any other questions about diabetes or this system."
    },
    {
        "keywords": ["help", "what can you", "options"],
        "response": (
            "I can answer questions about:\n"
            "• What this website does\n"
            "• Basics of diabetes (symptoms, types, risk factors)\n"
            "• How to use the prediction system\n"
            "• Prevention and treatment basics\n"
            "• Privacy and data handling\n\n"
            "Try asking: 'What is diabetes?' or 'How do I predict?'"
        )
    },
]

DEFAULT_RESPONSE = (
    "I'm not sure I understand that. Try asking about:\n"
    "• 'What does this website do?'\n"
    "• 'What is diabetes?'\n"
    "• 'What are the symptoms?'\n"
    "• 'How do I make a prediction?'\n"
    "• 'Is my data safe?'"
)


def get_bot_response(message):
    message_lower = message.lower()
    best_match = None
    best_score = 0
    for entry in CHATBOT_RESPONSES:
        score = sum(1 for kw in entry["keywords"] if kw in message_lower)
        if score > best_score:
            best_score = score
            best_match = entry["response"]
    return best_match or DEFAULT_RESPONSE


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Please type a message."})
    return jsonify({"reply": get_bot_response(message)})


# ============================================================
# PUBLIC ROUTES
# ============================================================
@app.route("/")
def landing():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not username or not email or not password:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))

        users = load_users()
        if username in users:
            flash("Username already exists.", "error")
            return redirect(url_for("signup"))

        users[username] = {
            "email": email,
            "password": generate_password_hash(password),
            "created": datetime.now().isoformat()
        }
        save_users(users)
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        users = load_users()
        if username not in users or not check_password_hash(users[username]["password"], password):
            flash("Invalid username or password.", "error")
            return redirect(url_for("login"))

        session["user"] = username
        session["email"] = users[username]["email"]
        flash(f"Welcome back, {username}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You've been logged out.", "success")
    return redirect(url_for("landing"))


# ============================================================
# PROTECTED ROUTES
# ============================================================
@app.route("/dashboard")
@login_required
def dashboard():
    history = load_history().get(session["user"], [])
    total_predictions = len(history)
    high_risk_count = sum(1 for h in history if h["prediction"] == "High Risk")
    low_risk_count = total_predictions - high_risk_count
    last_prediction = history[-1] if history else None

    return render_template(
        "dashboard.html",
        user=session["user"],
        email=session.get("email"),
        comparison=comparison_data,
        total_predictions=total_predictions,
        high_risk_count=high_risk_count,
        low_risk_count=low_risk_count,
        last_prediction=last_prediction
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            # Collect input values
            input_data = {f: float(request.form[f]) for f in feature_names}
            values = [input_data[f] for f in feature_names]
            arr = np.array(values).reshape(1, -1)
            arr_scaled = scaler.transform(arr)
            arr_selected = arr_scaled[:, feature_mask]

            pred = model.predict(arr_selected)[0]
            proba = model.predict_proba(arr_selected)[0][1]

            risk_level = "High Risk" if pred == 1 else "Low Risk"
            confidence = round(proba * 100, 2)

            # Save to history
            history = load_history()
            username = session["user"]
            history.setdefault(username, []).append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": risk_level,
                "confidence": confidence,
                "inputs": input_data
            })
            save_history(history)

            return render_template(
                "result.html",
                user=session["user"],
                prediction=risk_level,
                confidence=confidence
            )
        except Exception as e:
            flash(f"Prediction error: {e}", "error")
            return redirect(url_for("predict"))

    return render_template("predict.html", user=session["user"], features=feature_names)


@app.route("/history")
@login_required
def history():
    user_history = load_history().get(session["user"], [])

    total = len(user_history)
    high_risk = sum(1 for h in user_history if h["prediction"] == "High Risk")
    low_risk = total - high_risk
    avg_conf = round(sum(h["confidence"] for h in user_history) / total, 2) if total else 0

    # Reverse so newest first
    reversed_history = list(reversed(user_history))

    return render_template(
        "history.html",
        user=session["user"],
        history=reversed_history,
        total=total,
        high_risk=high_risk,
        low_risk=low_risk,
        avg_conf=avg_conf,
        feature_names=feature_names
    )


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    history = load_history()
    history[session["user"]] = []
    save_history(history)
    flash("Prediction history cleared.", "success")
    return redirect(url_for("history"))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    users = load_users()
    username = session["user"]

    if request.method == "POST":
        action = request.form.get("action")

        # ---- Update email ----
        if action == "update_email":
            new_email = request.form.get("email", "").strip()
            if not new_email:
                flash("Email cannot be empty.", "error")
            else:
                users[username]["email"] = new_email
                save_users(users)
                session["email"] = new_email
                flash("Email updated successfully.", "success")

        # ---- Change password ----
        elif action == "change_password":
            current = request.form.get("current_password", "")
            new_pw = request.form.get("new_password", "")
            confirm = request.form.get("confirm_password", "")

            if not check_password_hash(users[username]["password"], current):
                flash("Current password is incorrect.", "error")
            elif len(new_pw) < 6:
                flash("New password must be at least 6 characters.", "error")
            elif new_pw != confirm:
                flash("New passwords do not match.", "error")
            else:
                users[username]["password"] = generate_password_hash(new_pw)
                save_users(users)
                flash("Password changed successfully.", "success")

        # ---- Delete account ----
        elif action == "delete_account":
            confirm = request.form.get("confirm_delete", "")
            if confirm != "DELETE":
                flash("Type DELETE to confirm.", "error")
            else:
                del users[username]
                save_users(users)

                # Clean history
                history = load_history()
                history.pop(username, None)
                save_history(history)

                session.clear()
                flash("Your account has been deleted.", "success")
                return redirect(url_for("landing"))

        return redirect(url_for("settings"))

    return render_template(
        "settings.html",
        user=username,
        email=users[username].get("email", ""),
        created=users[username].get("created", "")
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)