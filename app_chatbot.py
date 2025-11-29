# from flask import Flask, request, jsonify, render_template
# import google.generativeai as genai
# import os
#
# app = Flask(__name__)
#
# # ------------------ Configure Gemini API ------------------
# genai.configure(api_key="AIzaSyC3GQT3T4EAei8X-7kIXNDmhc3hLG7kBeI")
# model = genai.GenerativeModel("gemini-2.5-flash")
#
# # ------------------ GLOBAL MEMORY -------------------------
# GLOBAL_PREDICTION_DATA = {
#     "status": None,
#     "risk_score": None,
#     "top_factors": None
# }
#
# # ------------------ STORE PREDICTION -----------------------
# @app.route("/store_prediction", methods=["POST"])
# def store_prediction():
#     try:
#         data = request.json
#         print("📥 Received from Streamlit:", data)
#
#         GLOBAL_PREDICTION_DATA["status"] = data.get("status", "Unknown")
#         GLOBAL_PREDICTION_DATA["risk_score"] = float(data.get("risk_score", 0))
#         GLOBAL_PREDICTION_DATA["top_factors"] = data.get("top_factors", [])
#
#         print("✅ Stored prediction:", GLOBAL_PREDICTION_DATA)
#
#         return jsonify({"message": "Prediction stored successfully"}), 200
#
#     except Exception as e:
#         print("❌ Store prediction error:", e)
#         return jsonify({"error": str(e)}), 500
#
#
#
#
# # ------------------ FRONTEND PAGE --------------------------
# @app.route("/chatbot_api", methods=["POST"])
# def chatbot_api():
#     user_msg = request.json.get("message", "").strip()
#
#     # If no prediction received yet
#     if GLOBAL_PREDICTION_DATA["status"] is None:
#         return jsonify({
#             "reply": (
#                 "Hi! Before I can guide you, please run a prediction on the "
#                 "Dropout Prediction page. Once you do that, I will analyze your "
#                 "risk score and explain the top factors influencing your result."
#             )
#         })
#
#     status = GLOBAL_PREDICTION_DATA["status"]
#     risk_score = GLOBAL_PREDICTION_DATA["risk_score"]
#     factors = GLOBAL_PREDICTION_DATA["top_factors"]
#
#     # Build a clean readable list for prompt
#     if isinstance(factors, list) and len(factors) > 0:
#         factor_text = "\n".join([f"- {f['feature']} (importance: {f['importance']:.3f})"
#                                  for f in factors])
#     else:
#         factor_text = "No important factors were detected."
#
#     prompt = f"""
# You are a friendly, supportive AI student counselor.
#
# Here is the student's ML dropout prediction:
#
# - Status: {status}
# - Risk Score: {risk_score:.4f}
# - Top Influencing Factors:
# {factor_text}
#
# Guidelines:
# - Speak naturally like a human counselor.
# - If the student asks about their result, explain it simply.
# - If risk is high, give supportive advice.
# - If risk is low, congratulate and motivate.
# - Avoid robotic tone.
# - Use conversation-based language.
# - Do NOT repeat the same lines.
#
# Student says:
# "{user_msg}"
#
# Respond helpfully.
# """
#
#     response = model.generate_content(prompt)
#     reply = response.text
#
#     return jsonify({"reply": reply})
#
#
# # ------------------ MAIN -----------------------------------
# if __name__ == "__main__":
#     print("Flask is running on port 5000")
#     app.run(debug=True, port=5000)


# app_chatbot.py (corrected)
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os

app = Flask(__name__)

# ------------------ Configure Gemini API ------------------
genai.configure(api_key="AIzaSyC3GQT3T4EAei8X-7kIXNDmhc3hLG7kBeI")
model = genai.GenerativeModel("gemini-2.5-flash")

# ------------------ GLOBAL MEMORY -------------------------
GLOBAL_PREDICTION_DATA = {
    "status": None,
    "risk_score": None,
    "top_factors": None
}

# ------------------ STORE PREDICTION -----------------------
@app.route("/store_prediction", methods=["POST"])
def store_prediction():
    try:
        data = request.json

        GLOBAL_PREDICTION_DATA["status"] = data.get("status")
        GLOBAL_PREDICTION_DATA["risk_score"] = float(data.get("risk_score"))
        GLOBAL_PREDICTION_DATA["top_factors"] = data.get("top_factors")

        print("🎯 Updated Chatbot Memory:", GLOBAL_PREDICTION_DATA)

        return jsonify({"message": "Prediction stored OK"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------ CHATBOT PAGE --------------------------
@app.route("/chatbot", methods=["GET"])
def chatbot_page():
    return render_template("app_chatbot.html")

# ------------------ CHATBOT API ---------------------------
@app.route("/chatbot_api", methods=["POST"])
def chatbot_api():
    user_msg = request.json.get("message", "").strip()

    # BLOCK CHATBOT UNTIL PREDICTION IS MADE
    if GLOBAL_PREDICTION_DATA["status"] is None:
        return jsonify({
            "reply": (
                "Please first generate a prediction using the Dropout "
                "Prediction tool. Once the system analyzes your data, "
                "I will explain your risk score and guiding factors 😊."
            )
        })

    status = GLOBAL_PREDICTION_DATA["status"]
    risk_score = GLOBAL_PREDICTION_DATA["risk_score"]
    factors = GLOBAL_PREDICTION_DATA["top_factors"]

    # format factors
    if isinstance(factors, list) and len(factors) > 0:
        factor_text = "\n".join(
            f"- {f['feature']} (importance: {f['importance']:.3f})"
            for f in factors
        )
    else:
        factor_text = "No significant factors were identified."

    # Prompt
    prompt = f"""
You are a warm, friendly academic counselor.

STUDENT ML RESULT:
- Status: {status}
- Risk Score: {risk_score:.4f}
- Top Influencing Factors:
{factor_text}

RESPOND RULES:
- Speak like a human mentor.
- Explain results clearly.
- If risk is high → give emotional support + action steps.
- If risk is low → motivate and guide.
- Use simple, natural conversational tone.
- Avoid robotic or repetitive responses.

Student says:
"{user_msg}"
"""

    response = model.generate_content(prompt)
    reply = response.text

    return jsonify({"reply": reply})

# ------------------ MAIN ----------------------------------
if __name__ == "__main__":
    print("Flask running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
