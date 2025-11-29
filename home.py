import streamlit as st
from style import load_theme

def show_home():
    load_theme()  # 👈 Load light blue theme
    st.title("🎓 Student Dropout Predictor")
    st.markdown("""
            <style>
                /* Background gradient */
                [data-testid="stAppViewContainer"] {
                    background: linear-gradient(135deg, #b3e5fc, #e3f2fd);
                    color: #003566;
                    font-family: "Segoe UI", sans-serif;
                }

                /* Titles and text */
                h1, h2, h3, h4, h5 {
                    color: #002855;
                }

                /* Buttons styling */
                .stButton>button {
                    background-color: white !important;
                    color: #003566 !important;
                    border-radius: 10px;
                    padding: 0.6em 1.2em;
                    border: 1px solid #90caf9;
                    font-weight: 600;
                    transition: all 0.25s ease-in-out;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
                }

                .stButton>button:hover {
                    background-color: #e3f2fd !important;
                    border: 1px solid #64b5f6;
                    transform: scale(1.03);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }

                /* Center all main content */
                .block-container {
                    padding-top: 3rem;
                    padding-bottom: 3rem;
                }
            </style>
        """, unsafe_allow_html=True)

    # --- Page Content ---
    st.subheader("AI-Powered Insights for Early Intervention")

    st.markdown("""
        Welcome to the **Student Dropout Prediction System**.  
        This platform uses **machine learning** to analyze student demographic, family, and academic data  
        to predict whether a student is at risk of dropping out.

        ---

        ### 🔍 Features:
        - **Prediction Page:** Enter student details to predict dropout likelihood  
        - **Visualization Page:** View insights like feature importance and prediction distribution  
        - **Built With:** Streamlit, Scikit-learn, Python  

        ---

        ### 📚 About the Project:
        Using **data-driven insights**, this system helps educators identify at-risk students early and  
        implement preventive strategies to improve retention.

        Use the navigation bar or click below to continue:
        """)

    # --- Go to Prediction Button ---
    if st.button("🔍 Go to Prediction"):
        st.session_state.page = "prediction"
        st.rerun()

