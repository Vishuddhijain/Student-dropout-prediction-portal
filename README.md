# 🎓 Student Dropout Prediction Portal
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python\&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![Explainable AI](https://img.shields.io/badge/Explainable%20AI-SHAP-green)
![Status](https://img.shields.io/badge/Project-Academic%20Minor-success)

An end-to-end **AI-powered educational support system** that predicts student dropout risk at an early stage and provides **transparent explanations** along with **personalized AI-driven counselling assistance**.

---

## 📌 Overview

Student dropout is a major challenge faced by higher education institutions, often leading to academic, financial, and social consequences.  
The **Student Dropout Prediction Portal** leverages **Machine Learning**, **Explainable AI (XAI)**, and **Generative AI** to assist institutions in identifying at-risk students early and supporting timely intervention.

The system not only predicts dropout probability but also explains *why* a student is at risk, enabling educators and counsellors to make informed, ethical, and data-driven decisions.
![HOME](home.png)

---

## 🚀 Key Capabilities

- Machine learning–based dropout prediction
- Ensemble learning using Random Forest, Decision Tree, and Logistic Regression
- Risk classification into **Low, Medium, High, and Extreme**
- Explainable AI using SHAP and feature importance
- Interactive web interface built with Streamlit
- Flask-based backend for prediction storage and communication
- AI-powered counselling assistant using Google Gemini
- Modular, scalable, and deployment-ready architecture

---

## 🧠 Technologies & Tools

- **Python**
- **Scikit-learn**
- **SHAP (Explainable AI)**
- **Streamlit**
- **Flask**
- **Google Gemini API**
- **Pandas, NumPy, Matplotlib**
- **Git & GitHub**

---

## 🏗️ System Architecture


![System Architecture](architecture.png)


---

## 📊 Prediction Dashboard

The dashboard allows users to input student demographic, academic, and socio-economic data and receive real-time predictions with risk visualization.

![Prediction Dashboard](prediction.png)

---

## 🔍 Explainable AI (XAI)

To ensure transparency and trust, the system explains each prediction using SHAP values and feature importance, highlighting the most influential factors contributing to dropout risk.

![SHAP Explanation](shap.png)

---

## 🤖 AI Counselling Assistant

An AI-powered counselling module generates personalized academic guidance based on the predicted risk level and contributing factors.

<!-- Smaller chatbot image --> <img src="chatbot.png" alt="AI Chatbot" width="350"/>
---

## 📈 Risk Interpretation

| Risk Level | Meaning |
|-----------|--------|
| Low | Student is academically stable |
| Medium | Monitoring and guidance advised |
| High | Counselling intervention recommended |
| Extreme | Immediate academic intervention required |

---

## 🧪 Testing & Validation

- Model evaluation using accuracy, precision, recall, and F1-score
- Cross-model consistency testing
- API and UI integration testing
- Edge-case handling for incomplete or extreme inputs

---

## ⚖️ Ethical AI Considerations

- Transparent and interpretable predictions
- No automated enforcement of decisions
- Designed to assist educators, not replace human judgement
- Responsible use of student data

---
Great README already 👍
Below is a **clean, professional “How to Run Locally” section** you can **directly paste** into your README. It matches your project structure and is viva-safe.

---

## ▶️ How to Run Locally

Follow the steps below to run the **Student Dropout Prediction Portal** on your local machine.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Vishuddhijain/Student-dropout-prediction-portal.git
cd Student-dropout-prediction-portal
```

---

### 2️⃣ Create & Activate Virtual Environment (Recommended)

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Start Backend (Flask API + Chatbot)

Open a new terminal (keep virtual environment active):

```bash
python app_chatbot.py
```

✔ Flask server will run at:

```
http://127.0.0.1:5000
```

---

### 5️⃣ Start Frontend (Streamlit App)

In another terminal:

```bash
streamlit run app.py
```

✔ Streamlit app will open automatically at:

```
http://localhost:8501
```

---

### 6️⃣ Use the Application

* Enter student demographic and academic details
* Click **Predict Dropout**
* View:

  * Dropout probability
  * Risk level (Low → Extreme)
  * Feature importance & SHAP explanations
* Interact with the **AI Counselling Assistant**

---

### ⚠️ Notes

* Ensure all `.pkl` model files are present in the project root
* Internet connection is required for Google Gemini AI
* This project is intended for **academic and research purposes**

---

## 📚 References

- UCI Student Performance Dataset
- Scikit-learn Documentation
- SHAP Documentation
- Streamlit Documentation
- Flask Documentation

---

## 👩‍💻 Author

**Vishuddhi Jain**  
📧 Email: [vishuddhi0303.jain@gmail.com](mailto:vishuddhi0303.jain@gmail.com)  
🎓 B.Tech Engineering Student

---
