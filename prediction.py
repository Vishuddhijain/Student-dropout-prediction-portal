# prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import matplotlib.pyplot as plt
from style import load_theme


# -------------------------
# Utility helpers
# -------------------------
def safe_load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

def extract_id(value):
    """Extract leading integer id from strings like '1 - Something' or '1 – Something'."""
    try:
        return int(value.split('–')[0].strip())
    except Exception:
        try:
            return int(value.split('-')[0].strip())
        except Exception:
            try:
                return int(value.split()[0].strip())
            except Exception:
                return 0

def prepare_for_model(input_df, encoder, scaler, model_columns, categorical_cols, numeric_cols):
    """
    Given the raw input (one-row dataframe), produce a dataframe aligned to model_columns.
    Returns aligned_df (1-row) or raises an exception.
    """
    X = input_df.copy().reset_index(drop=True)

    # ensure categorical cols exist
    for c in categorical_cols:
        if c not in X.columns:
            X[c] = "Unknown"

    # Encode categoricals using provided encoder
    if encoder is None:
        # fallback to pandas get_dummies
        enc_df = pd.get_dummies(X[categorical_cols], drop_first=False)
    else:
        enc_arr = encoder.transform(X[categorical_cols])
        if hasattr(enc_arr, "toarray"):
            enc_arr = enc_arr.toarray()
        enc_df = pd.DataFrame(enc_arr, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)

    # numeric part
    num_df = X[numeric_cols].astype(float).reset_index(drop=True)

    # other cols (booleans etc.)
    other_cols = [c for c in X.columns if c not in categorical_cols + numeric_cols]
    other_df = X[other_cols].reset_index(drop=True)

    final = pd.concat([num_df, enc_df.reset_index(drop=True), other_df], axis=1, sort=False)

    # scale numeric if scaler provided
    if scaler is not None and len(numeric_cols) > 0:
        try:
            final[numeric_cols] = scaler.transform(final[numeric_cols])
        except Exception:
            # if scaler fails, skip scaling but don't break
            pass

    # ensure all training columns exist
    if model_columns is not None:
        for c in model_columns:
            if c not in final.columns:
                final[c] = 0
        # align order
        final = final.reindex(columns=model_columns, fill_value=0)

    return final

# -------------------------
# Main app
# -------------------------
def show_prediction():
    load_theme()

    st.markdown("<h1>🎓 Student Dropout Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Enter student data and get dropout risk and counselling.</h3>", unsafe_allow_html=True)

    # we'll default to using these feature lists (same as your notebook)
    categorical_cols = [
        'Application_mode','Course','Marital_status','Nacionality',
        'Mothers_qualification','Fathers_qualification','Mothers_occupation','Fathers_occupation'
    ]
    numeric_cols = [
        'Previous_qualification_grade','Admission_grade',
        'Curricular_units_1st_sem_credited','Curricular_units_1st_sem_enrolled',
        'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_approved',
        'Age_at_enrollment'
    ]

    # ---------------------------
    # Load models (all three) & resources
    # ---------------------------
    rf_model = safe_load("rf_model.pkl")
    rf_encoder = safe_load("rf_encoder.pkl")
    rf_scaler = safe_load("rf_scaler.pkl")
    rf_cols = safe_load("rf_columns.pkl")

    dt_model = safe_load("dt_model.pkl")
    dt_encoder = safe_load("dt_encoder.pkl")
    dt_scaler = safe_load("dt_scaler.pkl")
    dt_cols = safe_load("dt_columns.pkl")

    lr_model = safe_load("lr_model.pkl")
    lr_encoder = safe_load("lr_encoder.pkl")
    lr_scaler = safe_load("lr_scaler.pkl")
    lr_cols = safe_load("lr_columns.pkl")
    ensemble_possible = all([
        rf_model, dt_model, lr_model,
        rf_encoder, dt_encoder, lr_encoder,
        rf_cols, dt_cols, lr_cols
    ])

    # If ensemble possible we will show a note; otherwise use RF only
    # if ensemble_possible:
    #     st.info("✅ Ensemble mode available: Predictions from RF, DT, and LR will be combined (majority vote).")
    # else:
    #     st.info(
    #         "ℹ️ Ensemble not available — using Random Forest only (ensure all model files are present to enable ensemble).")

    # ---------------------------
    # User input form ALWAYS DEFINED
    # ---------------------------
    def user_input_features():
        st.header("Student Demographics")
        col1, col2 = st.columns(2)
        with col1:
            marital_status = st.selectbox(
                'Marital Status',
                [
                    '1 – Single', '2 – Married', '3 – Widower', '4 – Divorced',
                    '5 – Facto Union', '6 – Legally Separated'
                ]
            )
            nationality = st.selectbox(
                'Nationality',
                [
                    '1 - Portuguese', '2 - German', '6 - Spanish', '11 - Italian', '13 - Dutch',
                    '14 - English', '17 - Lithuanian', '21 - Angolan', '22 - Cape Verdean',
                    '24 - Guinean', '25 - Mozambican', '26 - Santomean', '32 - Turkish',
                    '41 - Brazilian', '62 - Romanian', '100 - Moldova (Republic of)',
                    '101 - Mexican', '103 - Ukrainian', '105 - Russian', '108 - Cuban', '109 - Colombian'
                ]
            )
            gender = st.selectbox('Gender', ['1 – Male', '0 – Female'])

        with col2:
            age_at_enrollment = st.slider('Age at Enrollment', 17, 70, 18)
            displaced = st.selectbox('Displaced', ['1 – Yes', '0 – No'])
            international = st.selectbox('International', ['1 – Yes', '0 – No'])

        # Family Background
        st.header("Family Background")
        col3, col4 = st.columns(2)

        with col3:
            mothers_qualification = st.selectbox(
                "Mother's Qualification",
                [
                    '1 - Secondary Education - 12th Year of Schooling or Eq.',
                    '2 - Higher Education - Bachelor\'s Degree',
                    '3 - Higher Education - Degree', '4 - Higher Education - Master\'s',
                    '5 - Higher Education - Doctorate', '6 - Frequency of Higher Education',
                    '9 - 12th Year of Schooling - Not Completed',
                    '10 - 11th Year of Schooling - Not Completed', '11 - 7th Year (Old)',
                    '12 - Other - 11th Year of Schooling', '14 - 10th Year of Schooling',
                    '18 - General commerce course',
                    '19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
                    '22 - Technical-professional course', '26 - 7th year of schooling',
                    '27 - 2nd cycle of the general high school course',
                    '29 - 9th Year of Schooling - Not Completed',
                    '30 - 8th year of schooling', '34 - Unknown', '35 - Can\'t read or write',
                    '36 - Can read without having a 4th year of schooling',
                    '37 - Basic education 1st cycle (4th/5th year) or equiv.',
                    '38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
                    '39 - Technological specialization course',
                    '40 - Higher education - degree (1st cycle)',
                    '41 - Specialized higher studies course',
                    '42 - Professional higher technical course',
                    '43 - Higher Education - Master (2nd cycle)',
                    '44 - Higher Education - Doctorate (3rd cycle)'
                ]
            )

            fathers_qualification = st.selectbox(
                "Father's Qualification",
                [
                    '1 - Secondary Education - 12th Year of Schooling or Eq.',
                    '2 - Higher Education - Bachelor\'s Degree',
                    '3 - Higher Education - Degree', '4 - Higher Education - Master\'s',
                    '5 - Higher Education - Doctorate', '6 - Frequency of Higher Education',
                    '9 - 12th Year of Schooling - Not Completed',
                    '10 - 11th Year of Schooling - Not Completed', '11 - 7th Year (Old)',
                    '12 - Other - 11th Year of Schooling',
                    '13 - 2nd year complementary high school course',
                    '14 - 10th Year of Schooling', '18 - General commerce course',
                    '19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
                    '20 - Complementary High School Course',
                    '22 - Technical-professional course',
                    '25 - Complementary High School Course - not concluded',
                    '26 - 7th year of schooling',
                    '27 - 2nd cycle of the general high school course',
                    '29 - 9th Year of Schooling - Not Completed',
                    '30 - 8th year of schooling',
                    '31 - General Course of Administration and Commerce',
                    '33 - Supplementary Accounting and Administration',
                    '34 - Unknown', '35 - Can\'t read or write',
                    '36 - Can read without having a 4th year of schooling',
                    '37 - Basic education 1st cycle (4th/5th year) or equiv.',
                    '38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
                    '39 - Technological specialization course',
                    '40 - Higher education - degree (1st cycle)',
                    '41 - Specialized higher studies course',
                    '42 - Professional higher technical course',
                    '43 - Higher Education - Master (2nd cycle)',
                    '44 - Higher Education - Doctorate (3rd cycle)'
                ]
            )

        with col4:
            mothers_occupation = st.selectbox(
                "Mother's Occupation",
                [
                    '0 - Student', '1 - Representatives of the Legislative Power and Executive Bodies...',
                    '2 - Specialists in Intellectual and Scientific Activities',
                    '3 - Intermediate Level Technicians and Professions', '4 - Administrative staff',
                    '5 - Personal Services, Security and Safety Workers and Sellers',
                    '6 - Farmers and Skilled Workers...',
                    '7 - Skilled Workers in Industry, Construction and Craftsmen',
                    '8 - Installation and Machine Operators',
                    '9 - Unskilled Workers', '10 - Armed Forces Professions',
                    '90 - Other Situation', '99 - (blank)',
                    '122 - Health professionals', '123 - Teachers',
                    '125 - Specialists in ICT', '131 - Science Technicians',
                    '132 - Health Technicians',
                    '134 - Legal/Social/Sports Technicians',
                    '141 - Office workers', '143 - Data/Accounting Staff',
                    '144 - Other administrative support staff',
                    '151 - Personal service workers', '152 - Sellers', '153 - Personal care workers',
                    '171 - Skilled construction workers', '173 - Skilled printing workers',
                    '175 - Food/woodworking/other industry workers',
                    '191 - Cleaning workers', '192 - Unskilled in agriculture',
                    '193 - Unskilled in extractive industry', '194 - Meal preparation assistants'
                ]
            )

            fathers_occupation = st.selectbox(
                "Father's Occupation",
                [
                    '0 - Student',
                    '1 - Representatives of the Legislative Power and Executive Bodies...',
                    '2 - Specialists in Intellectual and Scientific Activities',
                    '3 - Intermediate Level Technicians and Professions',
                    '4 - Administrative staff', '5 - Personal Services Workers',
                    '6 - Farmers and Skilled Agriculture Workers',
                    '7 - Skilled Workers in Industry', '8 - Machine Operators',
                    '9 - Unskilled Workers', '10 - Armed Forces Professions',
                    '90 - Other Situation', '99 - (blank)',
                    '101 - Armed Forces Officers', '102 - Armed Forces Sergeants',
                    '103 - Other Armed Forces personnel',
                    '112 - Directors of administrative services',
                    '114 - Hotel/catering directors',
                    '121 - Physical science specialists',
                    '122 - Health professionals', '123 - Teachers',
                    '124 - Finance/Accounting specialists',
                    '131 - Engineering technicians', '132 - Health technicians',
                    '134 - Legal/Social technicians',
                    '135 - ICT technicians', '141 - Office workers',
                    '143 - Financial services operators',
                    '144 - Administrative support',
                    '151 - Personal service workers', '152 - Sellers',
                    '153 - Personal care workers',
                    '154 - Security personnel',
                    '161 - Skilled agricultural workers', '163 - Subsistence farmers',
                    '171 - Construction workers',
                    '172 - Metalworkers', '174 - Electricians',
                    '175 - Industry workers', '181 - Machine operators',
                    '182 - Assembly workers', '183 - Drivers',
                    '192 - Agriculture unskilled workers',
                    '193 - Unskilled in manufacturing',
                    '194 - Meal assistants',
                    '195 - Street vendors'
                ]
            )

        # Academic background
        st.header("Academic Background")
        col5, col6 = st.columns(2)

        with col5:
            previous_qualification = st.selectbox(
                'Previous Qualification',
                [
                    '1 - Secondary education', '2 - Higher education - bachelor\'s degree',
                    '3 - Higher education - degree',
                    '4 - Higher education - master\'s', '5 - Higher education - doctorate',
                    '6 - Frequency of higher education',
                    '9 - 12th year not completed', '10 - 11th year not completed',
                    '12 - Other 11th year', '14 - 10th year',
                    '15 - 10th year not completed',
                    '19 - Basic education 3rd cycle',
                    '38 - Basic education 2nd cycle',
                    '39 - Technological specialization course',
                    '40 - Higher education degree', '42 - Professional technical course',
                    '43 - Higher education master'
                ]
            )
            previous_qualification_grade = st.slider('Previous Qualification Grade', 0.0, 200.0, 150.0)
            admission_grade = st.slider('Admission Grade', 0.0, 200.0, 150.0)

        with col6:
            application_mode = st.selectbox(
                'Application Mode',
                [
                    '1 - 1st phase - general contingent',
                    '2 - Ordinance No. 612/93',
                    '5 - 1st phase - special contingent (Azores Island)',
                    '7 - Holders of other higher courses', '10 - Ordinance No. 854-B/99',
                    '15 - International student (bachelor)',
                    '16 - 1st phase special Madeira', '17 - 2nd phase general',
                    '18 - 3rd phase general', '26 - Ordinance 533-A/99...b2',
                    '27 - Ordinance 533-A/99...b3',
                    '39 - Over 23 years old', '42 - Transfer',
                    '43 - Change of course', '44 - Tech diploma holders',
                    '51 - Change of institution/course', '53 - Short cycle diploma',
                    '57 - International change'
                ]
            )
            application_order = st.slider('Application Order', 0, 9, 0)
            course = st.selectbox(
                'Course',
                [
                    '33 - Biofuel Production Technologies', '171 - Animation and Multimedia Design',
                    '8014 - Social Service (evening)', '9003 - Agronomy',
                    '9070 - Communication Design', '9085 - Veterinary Nursing',
                    '9119 - Informatics Engineering', '9130 - Equinculture', '9147 - Management',
                    '9238 - Social Service', '9254 - Tourism', '9500 - Nursing',
                    '9556 - Oral Hygiene',
                    '9670 - Advertising and Marketing Management',
                    '9773 - Journalism and Communication',
                    '9853 - Basic Education', '9991 - Management (evening)'
                ]
            )

        # Current performance
        st.header("Current Academic Performance")
        col7, col8 = st.columns(2)

        with col7:
            daytime_evening_attendance = st.selectbox('Daytime/Evening Attendance', ['1 – Daytime', '0 - Evening'])
            curricular_units_1st_sem_credited = st.slider('Curricular Units 1st Sem (Credited)', 0, 60, 30)
            curricular_units_1st_sem_enrolled = st.slider('Curricular Units 1st Sem (Enrolled)', 0, 60, 30)

        with col8:
            curricular_units_1st_sem_evaluations = st.slider('Curricular Units 1st Sem (Evaluations)', 0, 60, 30)
            curricular_units_1st_sem_approved = st.slider('Curricular Units 1st Sem (Approved)', 0, 60, 30)

        # Additional info
        st.header("Additional Information")
        col9, col10 = st.columns(2)

        with col9:
            educational_special_needs = st.selectbox('Educational Special Needs', ['1 – Yes', '0 – No'])
            debtor = st.selectbox('Debtor', ['1 – Yes', '0 – No'])

        with col10:
            tuition_fees_up_to_date = st.selectbox('Tuition Fees Up to Date', ['1 – Yes', '0 – No'])
            scholarship_holder = st.selectbox('Scholarship Holder', ['1 – Yes', '0 – No'])

        data = {
            'Marital_status': extract_id(marital_status),
            'Application_mode': extract_id(application_mode),
            'Application_order': application_order,
            'Course': extract_id(course),
            'Daytime_evening_attendance': extract_id(daytime_evening_attendance),
            'Previous_qualification': extract_id(previous_qualification),
            'Previous_qualification_grade': previous_qualification_grade,
            'Nacionality': extract_id(nationality),
            'Mothers_qualification': extract_id(mothers_qualification),
            'Fathers_qualification': extract_id(fathers_qualification),
            'Mothers_occupation': extract_id(mothers_occupation),
            'Fathers_occupation': extract_id(fathers_occupation),
            'Admission_grade': admission_grade,
            'Displaced': extract_id(displaced),
            'Educational_special_needs': extract_id(educational_special_needs),
            'Debtor': extract_id(debtor),
            'Tuition_fees_up_to_date': extract_id(tuition_fees_up_to_date),
            'Gender': extract_id(gender),
            'Scholarship_holder': extract_id(scholarship_holder),
            'Age_at_enrollment': age_at_enrollment,
            'International': extract_id(international),
            'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
            'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
            'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
            'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved
        }

        return pd.DataFrame(data, index=[0])

    # -------------------------
    # Run UI & Prediction
    # -------------------------
    input_df = user_input_features()

    if st.button("🔍 Predict Dropout"):
        try:
            # -------------------------
            # Prepare per-model predictions
            # -------------------------
            model_votes = []
            dropout_probs = []

            if ensemble_possible:
                model_entries = [
                    ("Random Forest", rf_model, rf_encoder, rf_scaler, rf_cols),
                    ("Decision Tree", dt_model, dt_encoder, dt_scaler, dt_cols),
                    ("Logistic Regression", lr_model, lr_encoder, lr_scaler, lr_cols)
                ]
            else:
                model_entries = [
                    ("Random Forest", rf_model, rf_encoder, rf_scaler, rf_cols)
                ]

            for name, m, enc, scl, cols in model_entries:
                if m is None:
                    st.warning(f"{name} model missing — skipping.")
                    continue

                try:
                    X_model = prepare_for_model(
                        input_df, enc, scl, cols,
                        categorical_cols, numeric_cols
                    )
                    pred = int(m.predict(X_model)[0])
                    model_votes.append(pred)

                    # dropout probability (class 0)
                    prob = 0.0
                    if hasattr(m, "predict_proba"):
                        probs = m.predict_proba(X_model)[0]
                        idx0 = list(m.classes_).index(0) if 0 in list(m.classes_) else 0
                        prob = float(probs[idx0])

                    dropout_probs.append(prob)

                except Exception as e:
                    st.warning(f"{name} prediction failed: {e}")

            if len(model_votes) == 0:
                st.error("No model predictions available.")
                raise RuntimeError("No model predictions.")

            # -------------------------
            # Ensemble aggregation
            # -------------------------
            final_pred = int(pd.Series(model_votes).mode().iat[0])
            avg_dropout_prob = float(np.mean(dropout_probs))

            status_map = {
                0: "🚨 Dropout Risk",
                1: "✅ Not Dropout",
                2: "🎓 Graduate"
            }
            predicted_label_text = status_map.get(final_pred, "Unknown")

            st.success(f"🎯 Final Prediction: **{predicted_label_text}**")
            st.write(f"📊 **Estimated Dropout Probability:** **{avg_dropout_prob * 100:.2f}%**")

            # -------------------------
            # RF Feature Importance
            # -------------------------
            top_factors = []
            try:
                if hasattr(rf_model, "estimators_"):
                    estimator = (
                        rf_model.estimators_[list(rf_model.classes_).index(0)]
                        if 0 in list(rf_model.classes_) else rf_model
                    )

                    if hasattr(estimator, "feature_importances_"):
                        fi = estimator.feature_importances_
                        pairs = list(zip(rf_cols, fi))
                        pairs.sort(key=lambda x: x[1], reverse=True)
                        top_factors = [
                            {"feature": f, "importance": float(im)}
                            for f, im in pairs[:10]
                        ]
            except Exception as e:
                st.warning(f"Top factor extraction failed: {e}")

            if top_factors:
                st.markdown("### 🔍 Top 10 Influencing Factors (RF)")
                for tf in top_factors:
                    st.write(f"- {tf['feature']} — {tf['importance']:.4f}")

            # -------------------------
            # Risk Thresholds & UI
            # -------------------------
            try:
                risk_thresholds = pickle.load(open("risk_thresholds.pkl", "rb"))
            except Exception:
                risk_thresholds = {"low": 0.20, "medium": 0.45, "high": 0.70,"Extreme": 0.95}

            def get_risk_level(p):
                if p < 0.20:
                    return "Low"
                elif p < 0.40:
                    return "Medium"
                elif p < 0.70:
                    return "High"
                else:
                    return "Extreme"

            risk_level = get_risk_level(avg_dropout_prob)

            st.markdown("## 🧭 Risk Dashboard")
            risk_colors = {
                "Low": "#4CAF50",
                "Medium": "#FFC107",
                "High": "#FF5722",
                "Extreme": "#B71C1C"
            }
            st.markdown(f"""
                <div style="
                    background-color: {risk_colors[risk_level]};
                    padding: 18px;
                    border-radius: 12px;
                    text-align: center;
                    color: white;
                    font-size: 22px;
                    font-weight: 700;">
                    Risk Level: {risk_level}
                </div>
            """, unsafe_allow_html=True)

            st.progress(min(max(avg_dropout_prob, 0.0), 1.0))

            # Alerts
            if risk_level == "Extreme":
                st.error("🚨 Extreme dropout risk. Immediate intervention required.")
            elif risk_level == "High":
                st.error("⚠ High dropout risk detected. Counselling recommended.")
            elif risk_level == "Medium":
                st.warning("🟡 Medium risk. Monitor and plan interventions.")
            else:
                st.success("🟢 Low risk. Student appears stable.")

            # -------------------------
            # Mini Pie Chart (Improved layout)
            # -------------------------
            fig, ax = plt.subplots(figsize=(1.4, 1.4), dpi=250)
            sizes = [avg_dropout_prob, 1 - avg_dropout_prob]
            colors = ["#1E88E5", "#FB8C00"]
            explode = (0.08, 0)

            wedges, _ = ax.pie(
                sizes, explode=explode, startangle=90,
                colors=colors, wedgeprops={'linewidth': 0.6, 'edgecolor': 'white'}
            )

            ax.legend(
                wedges,
                [f"Dropout {sizes[0] * 100:.1f}%", f"Not {sizes[1] * 100:.1f}%"],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.05),
                fontsize=6,
                ncol=2,
                frameon=False
            )
            ax.axis('equal')
            st.pyplot(fig)
            plt.close(fig)

            # -------------------------
            # Send to Chatbot Backend
            # -------------------------
            try:
                _ = requests.post(
                    "http://127.0.0.1:5000/store_prediction",
                    json={
                        "status": predicted_label_text,
                        "risk_score": float(avg_dropout_prob),
                        "top_factors": top_factors
                    },
                    timeout=3
                )
            except Exception:
                st.warning("⚠ Could not reach backend server to store prediction.")

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

    # Must be OUTSIDE all functions
    if __name__ == "__main__":
        show_prediction()





# import streamlit as st
# import pandas as pd
# import pickle
# import numpy as np
# import requests
# import shap
# import matplotlib.pyplot as plt
# from style import load_theme
# import streamlit.components.v1 as components
#
# def show_prediction():
#     load_theme()
#
#     st.markdown("<h1>🎓 Student Dropout Prediction</h1>", unsafe_allow_html=True)
#     st.markdown("<h3>Select a machine learning model & predict dropout likelihood.</h3>",
#                 unsafe_allow_html=True)
#
#     # ------------------------ MODEL SELECTION ------------------------
#     model_choice = st.selectbox(
#         "Select Model",
#         ["Random Forest", "Decision Tree", "Logistic Regression"]
#     )
#
#     # ------------------------ LOAD MODELS ACCORDINGLY ------------------------
#     if model_choice == "Random Forest":
#         model_file = "rf_model.pkl"
#         encoder_file = "rf_encoder.pkl"
#         scaler_file = "rf_scaler.pkl"
#         columns_file = "rf_columns.pkl"
#
#     elif model_choice == "Decision Tree":
#         model_file = "dt_model.pkl"
#         encoder_file = "dt_encoder.pkl"
#         scaler_file = "dt_scaler.pkl"
#         columns_file = "dt_columns.pkl"
#
#     elif model_choice == "Logistic Regression":
#         model_file = "lr_model.pkl"
#         encoder_file = "lr_encoder.pkl"
#         scaler_file = "lr_scaler.pkl"
#         columns_file = "lr_columns.pkl"
#
#     # -------- SAFE LOADING --------
#     with open(model_file, "rb") as f:
#         model = pickle.load(f)
#
#     with open(encoder_file, "rb") as f:
#         encoder = pickle.load(f)
#
#     with open(scaler_file, "rb") as f:
#         scaler = pickle.load(f)
#
#     with open(columns_file, "rb") as f:
#         training_columns = pickle.load(f)
#
#     # ---------------- HELPER ----------------
#     def extract_id(value):
#         try:
#             return int(value.split('–')[0].strip())
#         except:
#             try:
#                 return int(value.split('-')[0].strip())
#             except:
#                 return 0
#
#     # ----------------------------------------------------
#     # USER INPUT FUNCTION (UNCHANGED — ONLY INDENT FIXED)
#     # ----------------------------------------------------
#     def user_input_features():
#         st.header("Student Demographics")
#         col1, col2 = st.columns(2)
#
#         with col1:
#             marital_status = st.selectbox(
#                 'Marital Status',
#                 [
#                     '1 – Single', '2 – Married', '3 – Widower', '4 – Divorced',
#                     '5 – Facto Union', '6 – Legally Separated'
#                 ]
#             )
#             nationality = st.selectbox(
#                 'Nacionality',
#                 [
#                     '1 - Portuguese', '2 - German', '6 - Spanish', '11 - Italian', '13 - Dutch',
#                     '14 - English', '17 - Lithuanian', '21 - Angolan', '22 - Cape Verdean',
#                     '24 - Guinean', '25 - Mozambican', '26 - Santomean', '32 - Turkish',
#                     '41 - Brazilian', '62 - Romanian', '100 - Moldova (Republic of)',
#                     '101 - Mexican', '103 - Ukrainian', '105 - Russian', '108 - Cuban', '109 - Colombian'
#                 ]
#             )
#             gender = st.selectbox('Gender', ['1 – Male', '0 – Female'])
#
#         with col2:
#             age_at_enrollment = st.slider('Age at Enrollment', 17, 70, 18)
#             displaced = st.selectbox('Displaced', ['1 – Yes', '0 – No'])
#             international = st.selectbox('International', ['1 – Yes', '0 – No'])
#
#         # Family Background
#         st.header("Family Background")
#         col3, col4 = st.columns(2)
#
#         with col3:
#             mothers_qualification = st.selectbox(
#                 "Mother's Qualification",
#                 [
#                     '1 - Secondary Education - 12th Year of Schooling or Eq.',
#                     '2 - Higher Education - Bachelor\'s Degree',
#                     '3 - Higher Education - Degree', '4 - Higher Education - Master\'s',
#                     '5 - Higher Education - Doctorate', '6 - Frequency of Higher Education',
#                     '9 - 12th Year of Schooling - Not Completed',
#                     '10 - 11th Year of Schooling - Not Completed', '11 - 7th Year (Old)',
#                     '12 - Other - 11th Year of Schooling', '14 - 10th Year of Schooling',
#                     '18 - General commerce course',
#                     '19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
#                     '22 - Technical-professional course', '26 - 7th year of schooling',
#                     '27 - 2nd cycle of the general high school course',
#                     '29 - 9th Year of Schooling - Not Completed',
#                     '30 - 8th year of schooling', '34 - Unknown', '35 - Can\'t read or write',
#                     '36 - Can read without having a 4th year of schooling',
#                     '37 - Basic education 1st cycle (4th/5th year) or equiv.',
#                     '38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
#                     '39 - Technological specialization course',
#                     '40 - Higher education - degree (1st cycle)',
#                     '41 - Specialized higher studies course',
#                     '42 - Professional higher technical course',
#                     '43 - Higher Education - Master (2nd cycle)',
#                     '44 - Higher Education - Doctorate (3rd cycle)'
#                 ]
#             )
#
#             fathers_qualification = st.selectbox(
#                 "Father's Qualification",
#                 [
#                     '1 - Secondary Education - 12th Year of Schooling or Eq.',
#                     '2 - Higher Education - Bachelor\'s Degree',
#                     '3 - Higher Education - Degree', '4 - Higher Education - Master\'s',
#                     '5 - Higher Education - Doctorate', '6 - Frequency of Higher Education',
#                     '9 - 12th Year of Schooling - Not Completed',
#                     '10 - 11th Year of Schooling - Not Completed', '11 - 7th Year (Old)',
#                     '12 - Other - 11th Year of Schooling',
#                     '13 - 2nd year complementary high school course',
#                     '14 - 10th Year of Schooling', '18 - General commerce course',
#                     '19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.',
#                     '20 - Complementary High School Course',
#                     '22 - Technical-professional course',
#                     '25 - Complementary High School Course - not concluded',
#                     '26 - 7th year of schooling',
#                     '27 - 2nd cycle of the general high school course',
#                     '29 - 9th Year of Schooling - Not Completed',
#                     '30 - 8th year of schooling',
#                     '31 - General Course of Administration and Commerce',
#                     '33 - Supplementary Accounting and Administration',
#                     '34 - Unknown', '35 - Can\'t read or write',
#                     '36 - Can read without having a 4th year of schooling',
#                     '37 - Basic education 1st cycle (4th/5th year) or equiv.',
#                     '38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.',
#                     '39 - Technological specialization course',
#                     '40 - Higher education - degree (1st cycle)',
#                     '41 - Specialized higher studies course',
#                     '42 - Professional higher technical course',
#                     '43 - Higher Education - Master (2nd cycle)',
#                     '44 - Higher Education - Doctorate (3rd cycle)'
#                 ]
#             )
#
#         with col4:
#             mothers_occupation = st.selectbox(
#                 "Mother's Occupation",
#                 [
#                     '0 - Student', '1 - Representatives of the Legislative Power and Executive Bodies...',
#                     '2 - Specialists in Intellectual and Scientific Activities',
#                     '3 - Intermediate Level Technicians and Professions', '4 - Administrative staff',
#                     '5 - Personal Services, Security and Safety Workers and Sellers',
#                     '6 - Farmers and Skilled Workers...',
#                     '7 - Skilled Workers in Industry, Construction and Craftsmen',
#                     '8 - Installation and Machine Operators',
#                     '9 - Unskilled Workers', '10 - Armed Forces Professions',
#                     '90 - Other Situation', '99 - (blank)',
#                     '122 - Health professionals', '123 - Teachers',
#                     '125 - Specialists in ICT', '131 - Science Technicians',
#                     '132 - Health Technicians',
#                     '134 - Legal/Social/Sports Technicians',
#                     '141 - Office workers', '143 - Data/Accounting Staff',
#                     '144 - Other administrative support staff',
#                     '151 - Personal service workers', '152 - Sellers', '153 - Personal care workers',
#                     '171 - Skilled construction workers', '173 - Skilled printing workers',
#                     '175 - Food/woodworking/other industry workers',
#                     '191 - Cleaning workers', '192 - Unskilled in agriculture',
#                     '193 - Unskilled in extractive industry', '194 - Meal preparation assistants'
#                 ]
#             )
#
#             fathers_occupation = st.selectbox(
#                 "Father's Occupation",
#                 [
#                     '0 - Student',
#                     '1 - Representatives of the Legislative Power and Executive Bodies...',
#                     '2 - Specialists in Intellectual and Scientific Activities',
#                     '3 - Intermediate Level Technicians and Professions',
#                     '4 - Administrative staff', '5 - Personal Services Workers',
#                     '6 - Farmers and Skilled Agriculture Workers',
#                     '7 - Skilled Workers in Industry', '8 - Machine Operators',
#                     '9 - Unskilled Workers', '10 - Armed Forces Professions',
#                     '90 - Other Situation', '99 - (blank)',
#                     '101 - Armed Forces Officers', '102 - Armed Forces Sergeants',
#                     '103 - Other Armed Forces personnel',
#                     '112 - Directors of administrative services',
#                     '114 - Hotel/catering directors',
#                     '121 - Physical science specialists',
#                     '122 - Health professionals', '123 - Teachers',
#                     '124 - Finance/Accounting specialists',
#                     '131 - Engineering technicians', '132 - Health technicians',
#                     '134 - Legal/Social technicians',
#                     '135 - ICT technicians', '141 - Office workers',
#                     '143 - Financial services operators',
#                     '144 - Administrative support',
#                     '151 - Personal service workers', '152 - Sellers',
#                     '153 - Personal care workers',
#                     '154 - Security personnel',
#                     '161 - Skilled agricultural workers', '163 - Subsistence farmers',
#                     '171 - Construction workers',
#                     '172 - Metalworkers', '174 - Electricians',
#                     '175 - Industry workers', '181 - Machine operators',
#                     '182 - Assembly workers', '183 - Drivers',
#                     '192 - Agriculture unskilled workers',
#                     '193 - Unskilled in manufacturing',
#                     '194 - Meal assistants',
#                     '195 - Street vendors'
#                 ]
#             )
#
#         # Academic background
#         st.header("Academic Background")
#         col5, col6 = st.columns(2)
#
#         with col5:
#             previous_qualification = st.selectbox(
#                 'Previous Qualification',
#                 [
#                     '1 - Secondary education', '2 - Higher education - bachelor\'s degree',
#                     '3 - Higher education - degree',
#                     '4 - Higher education - master\'s', '5 - Higher education - doctorate',
#                     '6 - Frequency of higher education',
#                     '9 - 12th year not completed', '10 - 11th year not completed',
#                     '12 - Other 11th year', '14 - 10th year',
#                     '15 - 10th year not completed',
#                     '19 - Basic education 3rd cycle',
#                     '38 - Basic education 2nd cycle',
#                     '39 - Technological specialization course',
#                     '40 - Higher education degree', '42 - Professional technical course',
#                     '43 - Higher education master'
#                 ]
#             )
#             previous_qualification_grade = st.slider('Previous Qualification Grade', 0.0, 200.0, 150.0)
#             admission_grade = st.slider('Admission Grade', 0.0, 200.0, 150.0)
#
#         with col6:
#             application_mode = st.selectbox(
#                 'Application Mode',
#                 [
#                     '1 - 1st phase - general contingent',
#                     '2 - Ordinance No. 612/93',
#                     '5 - 1st phase - special contingent (Azores Island)',
#                     '7 - Holders of other higher courses', '10 - Ordinance No. 854-B/99',
#                     '15 - International student (bachelor)',
#                     '16 - 1st phase special Madeira', '17 - 2nd phase general',
#                     '18 - 3rd phase general', '26 - Ordinance 533-A/99...b2',
#                     '27 - Ordinance 533-A/99...b3',
#                     '39 - Over 23 years old', '42 - Transfer',
#                     '43 - Change of course', '44 - Tech diploma holders',
#                     '51 - Change of institution/course', '53 - Short cycle diploma',
#                     '57 - International change'
#                 ]
#             )
#             application_order = st.slider('Application Order', 0, 9, 0)
#             course = st.selectbox(
#                 'Course',
#                 [
#                     '33 - Biofuel Production Technologies', '171 - Animation and Multimedia Design',
#                     '8014 - Social Service (evening)', '9003 - Agronomy',
#                     '9070 - Communication Design', '9085 - Veterinary Nursing',
#                     '9119 - Informatics Engineering', '9130 - Equinculture', '9147 - Management',
#                     '9238 - Social Service', '9254 - Tourism', '9500 - Nursing',
#                     '9556 - Oral Hygiene',
#                     '9670 - Advertising and Marketing Management',
#                     '9773 - Journalism and Communication',
#                     '9853 - Basic Education', '9991 - Management (evening)'
#                 ]
#             )
#
#         # Current performance
#         st.header("Current Academic Performance")
#         col7, col8 = st.columns(2)
#
#         with col7:
#             daytime_evening_attendance = st.selectbox('Daytime/Evening Attendance', ['1 – Daytime', '0 - Evening'])
#             curricular_units_1st_sem_credited = st.slider('Curricular Units 1st Sem (Credited)', 0, 60, 30)
#             curricular_units_1st_sem_enrolled = st.slider('Curricular Units 1st Sem (Enrolled)', 0, 60, 30)
#
#         with col8:
#             curricular_units_1st_sem_evaluations = st.slider('Curricular Units 1st Sem (Evaluations)', 0, 60, 30)
#             curricular_units_1st_sem_approved = st.slider('Curricular Units 1st Sem (Approved)', 0, 60, 30)
#
#         # Additional info
#         st.header("Additional Information")
#         col9, col10 = st.columns(2)
#
#         with col9:
#             educational_special_needs = st.selectbox('Educational Special Needs', ['1 – Yes', '0 – No'])
#             debtor = st.selectbox('Debtor', ['1 – Yes', '0 – No'])
#
#         with col10:
#             tuition_fees_up_to_date = st.selectbox('Tuition Fees Up to Date', ['1 – Yes', '0 – No'])
#             scholarship_holder = st.selectbox('Scholarship Holder', ['1 – Yes', '0 – No'])
#
#         data = {
#             'Marital_status': extract_id(marital_status),
#             'Application_mode': extract_id(application_mode),
#             'Application_order': application_order,
#             'Course': extract_id(course),
#             'Daytime_evening_attendance': extract_id(daytime_evening_attendance),
#             'Previous_qualification': extract_id(previous_qualification),
#             'Previous_qualification_grade': previous_qualification_grade,
#             'Nacionality': extract_id(nationality),
#             'Mothers_qualification': extract_id(mothers_qualification),
#             'Fathers_qualification': extract_id(fathers_qualification),
#             'Mothers_occupation': extract_id(mothers_occupation),
#             'Fathers_occupation': extract_id(fathers_occupation),
#             'Admission_grade': admission_grade,
#             'Displaced': extract_id(displaced),
#             'Educational_special_needs': extract_id(educational_special_needs),
#             'Debtor': extract_id(debtor),
#             'Tuition_fees_up_to_date': extract_id(tuition_fees_up_to_date),
#             'Gender': extract_id(gender),
#             'Scholarship_holder': extract_id(scholarship_holder),
#             'Age_at_enrollment': age_at_enrollment,
#             'International': extract_id(international),
#             'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
#             'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
#             'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
#             'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved
#         }
#
#         return pd.DataFrame(data, index=[0])
#
#     # ------------------------ PREDICTION LOGIC ------------------------
#     input_df = user_input_features()
#
#     if st.button("🔍 Predict Dropout"):
#
#         try:
#             # -------------------------
#             # Categorical and Numerical
#             # -------------------------
#             categorical_cols = [
#                 'Application_mode', 'Course', 'Marital_status', 'Nacionality',
#                 'Mothers_qualification', 'Fathers_qualification',
#                 'Mothers_occupation', 'Fathers_occupation'
#             ]
#
#             numerical_cols = [
#                 'Previous_qualification_grade', 'Admission_grade',
#                 'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
#                 'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
#                 'Age_at_enrollment'
#             ]
#
#             # -------------------------
#             # ENCODING
#             # -------------------------
#             enc_input = input_df[categorical_cols]
#             encoded_arr = encoder.transform(enc_input)
#             encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
#
#             encoded_df = pd.DataFrame(
#                 encoded_arr if isinstance(encoded_arr, np.ndarray) else encoded_arr.toarray(),
#                 columns=encoded_cols,
#                 index=input_df.index
#             )
#
#             # Numeric + other columns
#             numeric_df = input_df[numerical_cols].astype(float).reset_index(drop=True)
#             other_cols = [c for c in input_df.columns if c not in categorical_cols + numerical_cols]
#             other_df = input_df[other_cols].reset_index(drop=True)
#
#             # FINAL DF
#             final_df = pd.concat([numeric_df, encoded_df, other_df], axis=1)
#
#             # Scale numeric
#             final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])
#
#             # Add missing training columns
#             for col in training_columns:
#                 if col not in final_df.columns:
#                     final_df[col] = 0
#
#             # Arrange column order
#             final_df = final_df[training_columns]
#
#             # -------------------------
#             # MODEL PREDICTION
#             # -------------------------
#             predicted_value = model.predict(final_df)[0]
#
#             # ----- Always compute dropout probability (class 0)
#             dropout_prob = float(model.predict_proba(final_df)[0][0])
#
#             status_map = {
#                 0: "🚨 Dropout Risk",
#                 1: "✅ Not Dropout",
#                 2: "🎓 Graduate"
#             }
#
#             predicted_label_text = status_map.get(predicted_value, "Unknown Outcome")
#
#             st.success(f"🎯 Model Used: **{model_choice}**")
#             st.success(f"Prediction: **{predicted_label_text}**")
#
#             st.write(f"📊 Dropout Probability: **{dropout_prob * 100:.2f}%**")
#
#             # -------------------------
#             # LOAD RISK THRESHOLDS
#             # -------------------------
#             try:
#                 risk_thresholds = pickle.load(open("risk_thresholds.pkl", "rb"))
#             except:
#                 risk_thresholds = {"low": 0.20, "medium": 0.45, "high": 0.70}
#
#             # -------------------------
#             # DETERMINE RISK LEVEL
#             # -------------------------
#             def get_risk_level(p):
#                 if p <= risk_thresholds["low"]:
#                     return "Low"
#                 elif p <= risk_thresholds["medium"]:
#                     return "Medium"
#                 elif p <= risk_thresholds["high"]:
#                     return "High"
#                 else:
#                     return "Extreme"
#
#             risk_level = get_risk_level(dropout_prob)
#
#             # -------------------------
#             # RISK DASHBOARD UI
#             # -------------------------
#             st.markdown("## 🧭 Risk Dashboard")
#
#             risk_colors = {
#                 "Low": "#4CAF50",
#                 "Medium": "#FFC107",
#                 "High": "#FF5722",
#                 "Extreme": "#B71C1C"
#             }
#
#             st.markdown(f"""
#                 <div style="
#                     background-color: {risk_colors[risk_level]};
#                     padding: 22px;
#                     border-radius: 12px;
#                     text-align: center;
#                     color: white;
#                     font-size: 26px;
#                     font-weight: bold;">
#                     Risk Level: {risk_level}
#                 </div>
#             """, unsafe_allow_html=True)
#
#             # Progress bar (fixed)
#             st.progress(min(max(dropout_prob, 0.01), 0.99))
#
#             # Messages
#             if risk_level == "Extreme":
#                 st.error("🚨 Extreme dropout risk. Immediate intervention required.")
#             elif risk_level == "High":
#                 st.error("⚠ High dropout risk detected. Counselling recommended.")
#             elif risk_level == "Medium":
#                 st.warning("🟡 Medium risk. Monitor regularly.")
#             else:
#                 st.success("🟢 Low risk. Student appears stable.")
#
#             # -------------------------
#             # Backend Log
#             # -------------------------
#             try:
#                 _ = requests.post(
#                     "http://127.0.0.1:5000/store_prediction",
#                     json={
#                         "status": predicted_label_text,
#                         "risk_score": dropout_prob,
#                         "top_factors": None
#                     },
#                     timeout=3
#                 )
#             except:
#                 pass
#
#         except Exception as e:
#             st.error(f"❌ Prediction error: {e}")

    # input_df = user_input_features()
    #
    # if st.button("🔍 Predict Dropout"):
    #
    #     try:
    #         categorical_cols = [
    #             'Application_mode', 'Course', 'Marital_status', 'Nacionality',
    #             'Mothers_qualification', 'Fathers_qualification',
    #             'Mothers_occupation', 'Fathers_occupation'
    #         ]
    #
    #         numerical_cols = [
    #             'Previous_qualification_grade', 'Admission_grade',
    #             'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    #             'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    #             'Age_at_enrollment'
    #         ]
    #
    #         enc_input = input_df[categorical_cols]
    #         encoded_arr = encoder.transform(enc_input)
    #         encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    #
    #         encoded_df = pd.DataFrame(
    #             encoded_arr.toarray() if hasattr(encoded_arr, "toarray") else encoded_arr,
    #             columns=encoded_cols, index=input_df.index
    #         )
    #
    #         numeric_df = input_df[numerical_cols].astype(float).reset_index(drop=True)
    #
    #         other_cols = [c for c in input_df.columns if c not in categorical_cols + numerical_cols]
    #         other_df = input_df[other_cols].reset_index(drop=True)
    #
    #         final_df = pd.concat([numeric_df, encoded_df, other_df], axis=1)
    #
    #         if numerical_cols:
    #             final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])
    #
    #         for col in training_columns:
    #             if col not in final_df.columns:
    #                 final_df[col] = 0
    #
    #         final_df = final_df[training_columns]
    #
    #         predicted_value = model.predict(final_df)[0]
    #
    #         if hasattr(model, "predict_proba"):
    #             pred_prob = float(model.predict_proba(final_df)[0][predicted_value])
    #         else:
    #             pred_prob = None
    #
    #         status_map = {0: "🚨 Dropout Risk", 1: "✅ Not Dropout", 2: "🎓 Graduate"}
    #         predicted_label_text = status_map.get(predicted_value, "Unknown")
    #
    #         st.success(f"🎯 Model Used: **{model_choice}**")
    #         st.success(f"Prediction: **{predicted_label_text}**")
    #
    #         if pred_prob is not None:
    #             st.write(f"📊 Confidence Score: **{pred_prob * 100:.2f}%**")
    #         # ============================
    #         # CORRECT DROPOUT PROBABILITY
    #         # ============================
    #         # Always take probability of class 0 (Dropout)
    #         dropout_prob = float(model.predict_proba(final_df)[0][0])
    #
    #         st.write(f"📊 Dropout Probability: **{dropout_prob * 100:.2f}%**")
    #
    #         # ============================
    #         # LOAD RISK THRESHOLDS
    #         # ============================
    #         try:
    #             risk_thresholds = pickle.load(open("risk_thresholds.pkl", "rb"))
    #         except:
    #             risk_thresholds = {"low": 0.20, "medium": 0.45, "high": 0.70}
    #
    #         # ============================
    #         # DETERMINE RISK LEVEL
    #         # ============================
    #         def get_risk_level(p):
    #             if p <= risk_thresholds["low"]:
    #                 return "Low"
    #             elif p <= risk_thresholds["medium"]:
    #                 return "Medium"
    #             elif p <= risk_thresholds["high"]:
    #                 return "High"
    #             else:
    #                 return "Extreme"
    #
    #         risk_level = get_risk_level(dropout_prob)
    #
    #         # ============================
    #         # RISK DASHBOARD UI
    #         # ============================
    #         risk_colors = {
    #             "Low": "#4CAF50",
    #             "Medium": "#FFC107",
    #             "High": "#F44336",
    #             "Extreme": "#8B0000"
    #         }
    #
    #         st.markdown("## 🧭 Risk Dashboard")
    #
    #         st.markdown(f"""
    #             <div style="
    #                 background-color: {risk_colors[risk_level]};
    #                 padding: 20px;
    #                 border-radius: 10px;
    #                 text-align: center;
    #                 color: white;
    #                 font-size: 26px;
    #                 font-weight: bold;">
    #                 Risk Level: {risk_level}
    #             </div>
    #         """, unsafe_allow_html=True)
    #
    #         st.progress(dropout_prob)
    #
    #         # Message based on risk
    #         if risk_level == "Extreme":
    #             st.error("🚨 Extreme dropout risk. Immediate intervention required.")
    #         elif risk_level == "High":
    #             st.error("⚠ High dropout risk detected.")
    #         elif risk_level == "Medium":
    #             st.warning("🟡 Medium risk. Monitor regularly.")
    #         else:
    #             st.success("🟢 Low risk. Student is stable.")
    #
    #         try:
    #             res = requests.post(
    #                 "http://127.0.0.1:5000/store_prediction",
    #                 json={
    #                     "status": predicted_label_text,
    #                     "risk_score": float(pred_prob) if pred_prob is not None else None,
    #                     "top_factors": None
    #                 },
    #                 timeout=3
    #             )
    #             if res.status_code == 200:
    #                 st.info("🤖 Counselling Assistant is ready .")
    #             else:
    #                 st.warning("⚠ Backend did not accept the prediction.")
    #
    #         except Exception as e:
    #             st.error(f"❌ Backend connection failed: {e}")
    #
    #     except Exception as e:
    #         st.error(f"Prediction error: {e}")
    #
    # # ------------------ FLOATING CHATBOT BUTTON ------------------
    # chatbot_css = """
    # <style>
    # #floatingChatBtn {
    #     position: fixed;
    #     bottom: 20px;
    #     right: 25px;
    #     background: linear-gradient(90deg, #2A7B9B, #57C785);
    #     color: white;
    #     width: 62px;
    #     height: 62px;
    #     border-radius: 50%;
    #     display: flex;
    #     justify-content: center;
    #     align-items: center;
    #     font-size: 26px;
    #     cursor: pointer;
    #     box-shadow: 0 10px 20px rgba(0,0,0,0.25);
    #     z-index: 9999;
    #     transition: 0.3s ease;
    # }
    # #floatingChatBtn:hover {
    #     transform: scale(1.15);
    # }
    # </style>
    #
    # <div id="floatingChatBtn" onclick="toggleChatbot()">🤖</div>
    #
    # <script>
    # let chatbotVisible = false;
    #
    # function toggleChatbot() {
    #     const iframe = document.getElementById("chatbotPopup");
    #
    #     if (!iframe) {
    #         const newIframe = document.createElement("iframe");
    #         newIframe.id = "chatbotPopup";
    #         newIframe.src = "http://127.0.0.1:5000/chatbot";
    #         newIframe.style.cssText = `
    #             position: fixed;
    #             bottom: 90px;
    #             right: 30px;
    #             width: 380px;
    #             height: 550px;
    #             border-radius: 18px;
    #             border: none;
    #             box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    #             z-index: 9999;
    #         `;
    #         document.body.appendChild(newIframe);
    #         chatbotVisible = true;
    #     } else {
    #         iframe.remove();
    #         chatbotVisible = false;
    #     }
    # }
    # </script>
    # """
    #
    # st.markdown(chatbot_css, unsafe_allow_html=True)

    # input_df = user_input_features()
    #
    # if st.button("🔍 Predict Dropout"):
    #
    #     try:
    #         categorical_cols = [
    #             'Application_mode', 'Course', 'Marital_status', 'Nacionality',
    #             'Mothers_qualification', 'Fathers_qualification',
    #             'Mothers_occupation', 'Fathers_occupation'
    #         ]
    #
    #         numerical_cols = [
    #             'Previous_qualification_grade', 'Admission_grade',
    #             'Curricular_units_1st_sem_credited', 'Curricular_units_1st_sem_enrolled',
    #             'Curricular_units_1st_sem_evaluations', 'Curricular_units_1st_sem_approved',
    #             'Age_at_enrollment'
    #         ]
    #
    #         enc_input = input_df[categorical_cols]
    #         encoded_arr = encoder.transform(enc_input)
    #         encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    #
    #         encoded_df = pd.DataFrame(
    #             encoded_arr.toarray() if hasattr(encoded_arr, "toarray") else encoded_arr,
    #             columns=encoded_cols, index=input_df.index
    #         )
    #
    #         numeric_df = input_df[numerical_cols].astype(float).reset_index(drop=True)
    #
    #         other_cols = [c for c in input_df.columns if c not in categorical_cols + numerical_cols]
    #         other_df = input_df[other_cols].reset_index(drop=True)
    #
    #         final_df = pd.concat([numeric_df, encoded_df, other_df], axis=1)
    #
    #         if numerical_cols:
    #             final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])
    #
    #         for col in training_columns:
    #             if col not in final_df.columns:
    #                 final_df[col] = 0
    #
    #         final_df = final_df[training_columns]
    #
    #         predicted_value = model.predict(final_df)[0]
    #
    #         if hasattr(model, "predict_proba"):
    #             pred_prob = float(model.predict_proba(final_df)[0][predicted_value])
    #         else:
    #             pred_prob = None
    #
    #     except Exception as e:
    #         st.error(f"Prediction error: {e}")
    #         st.stop()
    #
    #     # ------------------- AFTER ML LOGIC -------------------
    #     import requests
    #
    #     status_map = {
    #         0: "🚨 Dropout Risk",
    #         1: "✅ Not Dropout",
    #         2: "🎓 Graduate"
    #     }
    #
    #     predicted_label_text = status_map.get(predicted_value, "Unknown")
    #
    #     st.success(f"🎯 Model Used: **{model_choice}**")
    #     st.success(f"Prediction: **{predicted_label_text}**")
    #
    #     if pred_prob is not None:
    #         st.write(f"📊 Confidence Score: **{pred_prob * 100:.2f}%**")
    #
    #     # ---------------- SEND DATA TO BACKEND ----------------
    #     backend_ok = False
    #     try:
    #         res = requests.post(
    #             "http://127.0.0.1:5000/store_prediction",
    #             json={
    #                 "status": predicted_label_text,
    #                 "risk_score": float(pred_prob) if pred_prob is not None else None,
    #                 "top_factors": None
    #             },
    #             timeout=4
    #         )
    #
    #         if res.status_code == 200:
    #             backend_ok = True
    #         else:
    #             st.warning(f"⚠ Backend did not accept the prediction (Code {res.status_code})")
    #
    #     except Exception as e:
    #         st.error(f"❌ Backend connection failed: {e}")
    #
    #     # -------------------------------------------------------
    #     # 🔵 SHOW PULSE BUTTON ONLY WHEN BACKEND WORKED
    #     # -------------------------------------------------------
    #     import streamlit.components.v1 as components
    #
    #     if backend_ok:
    #
    #         chatbot_html = """
    #             <style>
    #             #floatingChatBtn {
    #                 position: fixed;
    #                 bottom: 20px;
    #                 right: 25px;
    #                 background: linear-gradient(90deg, #2A7B9B, #57C785);
    #                 color: white;
    #                 width: 62px;
    #                 height: 62px;
    #                 border-radius: 50%;
    #                 display: flex;
    #                 justify-content: center;
    #                 align-items: center;
    #                 font-size: 26px;
    #                 cursor: pointer;
    #                 box-shadow: 0 10px 20px rgba(0,0,0,0.25);
    #                 z-index: 9999;
    #                 transition: 0.3s ease;
    #             }
    #             #floatingChatBtn:hover {
    #                 transform: scale(1.15);
    #             }
    #             </style>
    #
    #             <div id="floatingChatBtn" onclick="toggleChatbot()">🤖</div>
    #
    #             <script>
    #             function toggleChatbot() {
    #                 let iframe = document.getElementById("chatbotPopup");
    #
    #                 if (!iframe) {
    #                     iframe = document.createElement("iframe");
    #                     iframe.id = "chatbotPopup";
    #                     iframe.src = "http://127.0.0.1:5000/chatbot";
    #                     iframe.style.position = "fixed";
    #                     iframe.style.bottom = "90px";
    #                     iframe.style.right = "30px";
    #                     iframe.style.width = "380px";
    #                     iframe.style.height = "550px";
    #                     iframe.style.borderRadius = "18px";
    #                     iframe.style.border = "none";
    #                     iframe.style.boxShadow = "0 8px 25px rgba(0,0,0,0.3)";
    #                     iframe.style.zIndex = "9999";
    #
    #                     document.body.appendChild(iframe);
    #                 } else {
    #                     iframe.remove();
    #                 }
    #             }
    #             </script>
    #         """
    #
    #         components.html(chatbot_html, height=0, width=0)
    #     else:
    #         st.info("⚠ Prediction store failed — Chatbot button disabled.")


