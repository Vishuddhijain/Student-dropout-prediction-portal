
# visualization.py (final fixed version)
# visualization.py
# visualization.py
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pickle
# import shap
# import warnings
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from style import load_theme
#
# warnings.filterwarnings("ignore")
# sns.set_style("whitegrid")
#
# # ---------------------------
# # Utility helpers
# # ---------------------------
# def _load_pickle(path):
#     try:
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     except Exception:
#         return None
#
# def map_status_to_code(series):
#     return series.map({"Dropout": 0, "Graduate": 1, "Enrolled": 2})
#
# def _align_features_for_model(X_df: pd.DataFrame, model, training_columns_saved):
#     X = X_df.copy().reset_index(drop=True)
#     model_expected = None
#     n_expected = getattr(model, "n_features_in_", None)
#
#     if hasattr(model, "feature_names_in_"):
#         try:
#             model_expected = list(model.feature_names_in_)
#         except Exception:
#             model_expected = None
#
#     if model_expected is None and training_columns_saved is not None:
#         if n_expected is None or len(training_columns_saved) == n_expected:
#             model_expected = training_columns_saved.copy()
#
#     if model_expected is None and training_columns_saved is not None:
#         model_expected = training_columns_saved.copy()
#
#     if model_expected is None:
#         return X, {"aligned": False, "reason": "no expected feature names available"}
#
#     aligned = X.reindex(columns=model_expected, fill_value=0)
#     info = {
#         "aligned": True,
#         "expected_count": len(model_expected),
#         "got_count": X.shape[1],
#         "missing_cols": [c for c in model_expected if c not in X.columns],
#         "extra_cols": [c for c in X.columns if c not in model_expected]
#     }
#     return aligned, info
#
# # ---------------------------
# # Build encoded dataframe (per-model encoder)
# # ---------------------------
# def build_encoded_dataframe(X_raw_in: pd.DataFrame, encoder, training_columns, categorical_columns):
#     """
#     encoder: fitted OneHotEncoder (expecting categorical_columns) or None
#     training_columns: list or None
#     categorical_columns: list of column names to encode
#     """
#     X = X_raw_in.copy()
#     # If no encoder, fallback to pandas.get_dummies
#     if encoder is None:
#         X_enc = pd.get_dummies(X, drop_first=False)
#         if training_columns is not None:
#             X_enc = X_enc.reindex(columns=training_columns, fill_value=0)
#         return X_enc
#
#     # ensure categorical columns exist
#     for c in categorical_columns:
#         if c not in X.columns:
#             X[c] = "Unknown"
#
#     # transform
#     try:
#         enc_arr = encoder.transform(X[categorical_columns])
#     except Exception as e:
#         raise RuntimeError(f"OneHotEncoder.transform failed: {e}")
#
#     if not isinstance(enc_arr, np.ndarray):
#         try:
#             enc_arr = enc_arr.toarray()
#         except Exception:
#             enc_arr = np.asarray(enc_arr)
#
#     enc_df = pd.DataFrame(
#         enc_arr,
#         columns=encoder.get_feature_names_out(categorical_columns),
#         index=X.index
#     )
#
#     numeric_cols = [c for c in X.columns if c not in categorical_columns]
#     num_df = X[numeric_cols].reset_index(drop=True)
#     enc_df = enc_df.reset_index(drop=True)
#     X_final_local = pd.concat([num_df, enc_df], axis=1)
#
#     if training_columns is not None:
#         X_final_local = X_final_local.reindex(columns=training_columns, fill_value=0)
#
#     return X_final_local
#
# # ---------------------------
# # Load model bundle (CASE A)
# # ---------------------------
# def load_model_bundle(model_key):
#     """
#     model_key: 'rf' | 'dt' | 'lr'
#     returns model, encoder, scaler, columns
#     """
#     if model_key == "rf":
#         model_path = "rf_model.pkl"
#         enc_path = "rf_encoder.pkl"
#         scl_path = "rf_scaler.pkl"
#         col_path = "rf_columns.pkl"
#     elif model_key == "dt":
#         model_path = "dt_model.pkl"
#         enc_path = "dt_encoder.pkl"
#         scl_path = "dt_scaler.pkl"
#         col_path = "dt_columns.pkl"
#     elif model_key == "lr":
#         model_path = "lr_model.pkl"
#         enc_path = "lr_encoder.pkl"
#         scl_path = "lr_scaler.pkl"
#         col_path = "lr_columns.pkl"
#     else:
#         return None, None, None, None
#
#     model = _load_pickle(model_path)
#     encoder = _load_pickle(enc_path)
#     scaler = _load_pickle(scl_path)
#     columns = _load_pickle(col_path)
#     return model, encoder, scaler, columns
#
# # ---------------------------
# # Main app
# # ---------------------------
# def show_visualization():
#     load_theme()
#     st.title("🎓 Student Dropout Analysis Dashboard")
#     st.write("Visualize dataset insights and evaluate trained models (RF, DT, LR).")
#
#     # ---------- Load dataset ----------
#     data_path = "data.csv"
#     if not os.path.exists(data_path):
#         st.error("❌ data.csv not found in the working directory.")
#         return
#
#     df = pd.read_csv(data_path, delimiter=";")
#     st.success(f"✅ Loaded dataset — rows: {df.shape[0]}, cols: {df.shape[1]}")
#
#     # ---------- Distribution plots ----------
#     st.header("📊 Basic Distributions")
#     fig = plt.figure(figsize=(8, 6))
#     sns.countplot(x="Marital_status", data=df, palette="Set2")
#     plt.title("Distribution of Marital Status")
#     st.pyplot(fig)
#     plt.close(fig)
#
#     fig = plt.figure(figsize=(12, 8))
#     sns.countplot(y="Course", data=df, order=df["Course"].value_counts().index, palette="Set2")
#     plt.title("Distribution of Courses")
#     st.pyplot(fig)
#     plt.close(fig)
#
#     fig = plt.figure(figsize=(8, 6))
#     sns.countplot(x="Gender", data=df, palette="Set2")
#     plt.title("Distribution of Gender")
#     st.pyplot(fig)
#     plt.close(fig)
#
#     fig = plt.figure(figsize=(8, 6))
#     sns.countplot(x="Status", data=df, palette="Set2")
#     plt.title("Distribution of Status (Dropout vs Graduate vs Enrolled)")
#     st.pyplot(fig)
#     plt.close(fig)
#
#     # ---------- Correlation heatmap ----------
#     st.header("🔥 Correlation Heatmap (numeric features)")
#     numeric_df = df.select_dtypes(include=["float64", "int64"])
#     if numeric_df.shape[1] > 0:
#         fig = plt.figure(figsize=(14, 12))
#         sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
#         plt.title("Correlation Matrix")
#         st.pyplot(fig)
#         plt.close(fig)
#     else:
#         st.info("No numeric columns found for correlation heatmap.")
#
#     # ---------- Target encoding to maintain parity with backend ----------
#     if "Status" not in df.columns:
#         st.error("❌ 'Status' column missing in data.csv (expected 'Dropout', 'Graduate', 'Enrolled').")
#         return
#
#     df["Status_encoded"] = map_status_to_code(df["Status"])
#     if df["Status_encoded"].isna().any():
#         st.error("❌ Some Status values could not be mapped to 0/1/2. Please ensure values are Dropout/Graduate/Enrolled.")
#         return
#
#     # ---------- Define categorical and numeric columns (same as backend) ----------
#     categorical_columns = [
#         'Application_mode','Course','Marital_status','Nacionality',
#         'Mothers_qualification','Fathers_qualification','Mothers_occupation','Fathers_occupation'
#     ]
#     numeric_columns = [
#         'Previous_qualification_grade','Admission_grade',
#         'Curricular_units_1st_sem_credited','Curricular_units_1st_sem_enrolled',
#         'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_approved',
#         'Age_at_enrollment'
#     ]
#
#     # ---------- Try to load shared processed test/train saved by backend ----------
#     shared_X_test = _load_pickle("shared_X_test.pkl")
#     shared_y_test = _load_pickle("shared_y_test.pkl")
#     shared_X_train = _load_pickle("shared_X_train.pkl")
#     shared_y_train = _load_pickle("shared_y_train.pkl")
#
#     if shared_X_test is None or shared_y_test is None:
#         st.warning("Shared test set (shared_X_test.pkl/shared_y_test.pkl) not found. Falling back to local split (may not match backend exactly).")
#         # fallback: create processed features similar to backend using get_dummies (best-effort)
#         X_raw = df.drop(columns=["Status", "Status_encoded"]).copy()
#         X_all = pd.get_dummies(X_raw, drop_first=False)
#         y_all = df["Status_encoded"].astype(int).values
#         idx = np.arange(len(X_all))
#         try:
#             idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_all)
#         except Exception:
#             idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
#         shared_X_test = X_all.iloc[idx_test].reset_index(drop=True)
#         shared_y_test = df["Status_encoded"].iloc[idx_test].astype(int).values
#         # no shared_X_train / y_train available in this fallback
#     else:
#         st.success("✅ Loaded shared_X_test.pkl and shared_y_test.pkl — frontend will match backend evaluation exactly.")
#         # ensure indices reset
#         shared_X_test = shared_X_test.reset_index(drop=True)
#         try:
#             shared_y_test = np.array(shared_y_test).astype(int)
#         except Exception:
#             shared_y_test = np.array(shared_y_test)
#
#     # If shared_X_train exists, make small background sample for Kernel SHAP fallback
#     if shared_X_train is not None:
#         shared_X_train = shared_X_train.reset_index(drop=True)
#
#     # ---------- Helper to present confusion matrix nicely ----------
#     def show_confusion(cm, title, cmap="Blues"):
#         fig, ax = plt.subplots(figsize=(5, 4))
#         sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax)
#         ax.set_xlabel("Predicted")
#         ax.set_ylabel("Actual")
#         ax.set_title(title)
#         st.pyplot(fig)
#         plt.close(fig)
#
#     model_results = {}
#
#     # -------- RANDOM FOREST --------
#     st.subheader("🌲 Random Forest Results")
#     try:
#         rf_model, rf_encoder, rf_scaler, rf_columns = load_model_bundle("rf")
#         if rf_model is None:
#             st.warning("rf_model.pkl not found — skipping Random Forest evaluation.")
#         else:
#             # If shared_X_test is processed (as backend saved), align directly.
#             # Otherwise build per-model encoded features from raw inputs.
#             try:
#                 # try aligning directly
#                 X_rf_aligned, info_rf = _align_features_for_model(shared_X_test, rf_model, rf_columns)
#                 if info_rf.get("missing_cols"):
#                     st.warning(f"RF: filling {len(info_rf['missing_cols'])} missing columns with zeros.")
#             except Exception:
#                 # fallback: build from raw using rf_encoder
#                 X_raw = df.drop(columns=["Status", "Status_encoded"]).copy()
#                 X_rf_all = build_encoded_dataframe(X_raw, encoder=rf_encoder, training_columns=rf_columns, categorical_columns=categorical_columns)
#                 X_rf_aligned = X_rf_all.iloc[shared_X_test.index].reset_index(drop=True)
#
#             # Apply RF scaler to numeric columns if exists (backend saved one)
#             try:
#                 if rf_scaler is not None:
#                     # scaler was fit on numeric_columns in backend; safely transform if columns exist
#                     present_nums = [c for c in numeric_columns if c in X_rf_aligned.columns]
#                     if present_nums:
#                         X_rf_aligned[present_nums] = rf_scaler.transform(X_rf_aligned[present_nums])
#             except Exception:
#                 pass
#
#             # Predict
#             preds_rf = rf_model.predict(X_rf_aligned)
#             y_test = np.array(shared_y_test).astype(int)
#             acc_rf = accuracy_score(y_test, preds_rf)
#             model_results["Random Forest"] = acc_rf
#
#             st.metric("Random Forest Accuracy", f"{acc_rf*100:.2f}%")
#             st.text("Classification Report:")
#             st.text(classification_report(y_test, preds_rf, digits=4))
#
#             cm_rf = confusion_matrix(y_test, preds_rf)
#             show_confusion(cm_rf, "Random Forest - Confusion Matrix", cmap="Blues")
#
#             # Feature importance (aggregate if OVR)
#             feat_imp = None
#             try:
#                 if hasattr(rf_model, "estimators_") and isinstance(rf_model.estimators_, list):
#                     imps = []
#                     for sub in rf_model.estimators_:
#                         if hasattr(sub, "feature_importances_"):
#                             imps.append(sub.feature_importances_)
#                     if len(imps) > 0:
#                         feat_imp = np.mean(imps, axis=0)
#                 elif hasattr(rf_model, "feature_importances_"):
#                     feat_imp = rf_model.feature_importances_
#             except Exception:
#                 feat_imp = None
#
#             if feat_imp is not None:
#                 try:
#                     feat_series = pd.Series(feat_imp, index=X_rf_aligned.columns)
#                     top10 = feat_series.nlargest(10)
#                     fig, ax = plt.subplots(figsize=(7, 5))
#                     top10.sort_values().plot(kind="barh", ax=ax)
#                     ax.set_title("🌲 Random Forest - Top 10 Important Features")
#                     st.pyplot(fig)
#                     plt.close(fig)
#                 except Exception as e:
#                     st.warning(f"Could not plot RF top features: {e}")
#             else:
#                 st.info("RF feature importances not available.")
#
#             # SHAP for RF (TreeExplainer) — sample for speed
#             try:
#                 st.markdown("**SHAP Explainability (Random Forest)**")
#                 rf_sample = X_rf_aligned.sample(min(len(X_rf_aligned), 200), random_state=42)
#                 # If OVR, TreeExplainer expects a tree estimator -> pick estimator[0]
#                 rf_tree = rf_model.estimators_[0] if hasattr(rf_model, "estimators_") else rf_model
#                 explainer = shap.TreeExplainer(rf_tree)
#                 shap_values = explainer.shap_values(rf_sample)
#                 plt.figure()
#                 if isinstance(shap_values, list):
#                     # multiclass -> shap_values is list; show a representative class (1)
#                     shap.summary_plot(shap_values[1], rf_sample, show=False)
#                 else:
#                     shap.summary_plot(shap_values, rf_sample, show=False)
#                 st.pyplot(plt.gcf())
#                 plt.close()
#             except Exception as e:
#                 st.warning(f"RF SHAP failed: {e}")
#
#     except Exception as e:
#         st.error(f"Random Forest evaluation error: {e}")
#
#     # -------- DECISION TREE --------
#     st.subheader("🌳 Decision Tree Results")
#     try:
#         dt_model, dt_encoder, dt_scaler, dt_columns = load_model_bundle("dt")
#         if dt_model is None:
#             st.warning("dt_model.pkl not found — skipping Decision Tree evaluation.")
#         else:
#             # Align
#             try:
#                 X_dt_aligned, info_dt = _align_features_for_model(shared_X_test, dt_model, dt_columns)
#                 if info_dt.get("missing_cols"):
#                     st.warning(f"DT: filling {len(info_dt['missing_cols'])} missing columns with zeros.")
#             except Exception:
#                 X_raw = df.drop(columns=["Status", "Status_encoded"]).copy()
#                 X_dt_all = build_encoded_dataframe(X_raw, encoder=dt_encoder, training_columns=dt_columns, categorical_columns=categorical_columns)
#                 X_dt_aligned = X_dt_all.iloc[shared_X_test.index].reset_index(drop=True)
#
#             # Apply dt scaler if present
#             try:
#                 if dt_scaler is not None:
#                     present_nums = [c for c in numeric_columns if c in X_dt_aligned.columns]
#                     if present_nums:
#                         X_dt_aligned[present_nums] = dt_scaler.transform(X_dt_aligned[present_nums])
#             except Exception:
#                 pass
#
#             preds_dt = dt_model.predict(X_dt_aligned)
#             y_test = np.array(shared_y_test).astype(int)
#             acc_dt = accuracy_score(y_test, preds_dt)
#             model_results["Decision Tree"] = acc_dt
#
#             st.metric("Decision Tree Accuracy", f"{acc_dt*100:.2f}%")
#             st.text("Classification Report:")
#             st.text(classification_report(y_test, preds_dt, digits=4))
#
#             cm_dt = confusion_matrix(y_test, preds_dt)
#             show_confusion(cm_dt, "Decision Tree - Confusion Matrix", cmap="Oranges")
#
#             # feature importances
#             if hasattr(dt_model, "estimators_") and isinstance(dt_model.estimators_, list):
#                 # aggregate like RF
#                 imps = []
#                 for sub in dt_model.estimators_:
#                     if hasattr(sub, "feature_importances_"):
#                         imps.append(sub.feature_importances_)
#                 if len(imps) > 0:
#                     feat_imp_dt = np.mean(imps, axis=0)
#                 else:
#                     feat_imp_dt = None
#             elif hasattr(dt_model, "feature_importances_"):
#                 feat_imp_dt = dt_model.feature_importances_
#             else:
#                 feat_imp_dt = None
#
#             if feat_imp_dt is not None:
#                 try:
#                     feat_series_dt = pd.Series(feat_imp_dt, index=X_dt_aligned.columns)
#                     top10_dt = feat_series_dt.nlargest(10)
#                     fig, ax = plt.subplots(figsize=(7, 5))
#                     top10_dt.sort_values().plot(kind="barh", ax=ax, color="orange")
#                     ax.set_title("🌳 Decision Tree - Top 10 Important Features")
#                     st.pyplot(fig)
#                     plt.close(fig)
#                 except Exception as e:
#                     st.warning(f"Could not plot DT importances: {e}")
#
#             # SHAP for DT: try TreeExplainer on a wrapped estimator; if not supported, fallback to KernelExplainer
#             try:
#                 st.markdown("**SHAP Explainability (Decision Tree)**")
#                 dt_sample = X_dt_aligned.sample(min(len(X_dt_aligned), 200), random_state=42)
#                 # If wrapped as OneVsRest, try to get the first estimator
#                 dt_tree = dt_model.estimators_[0] if hasattr(dt_model, "estimators_") else dt_model
#                 try:
#                     explainer_dt = shap.TreeExplainer(dt_tree)
#                     shap_vals_dt = explainer_dt.shap_values(dt_sample)
#                     plt.figure()
#                     if isinstance(shap_vals_dt, list):
#                         shap.summary_plot(shap_vals_dt[1], dt_sample, show=False)
#                     else:
#                         shap.summary_plot(shap_vals_dt, dt_sample, show=False)
#                     st.pyplot(plt.gcf())
#                     plt.close()
#                 except Exception:
#                     # fallback: KernelExplainer (slower) using small background from shared_X_train if available
#                     if shared_X_train is not None:
#                         background = shared_X_train.sample(min(50, len(shared_X_train)), random_state=42)
#                     else:
#                         background = X_dt_aligned.sample(min(50, len(X_dt_aligned)), random_state=42)
#                     expl = shap.KernelExplainer(dt_model.predict_proba, background)
#                     shap_vals = expl.shap_values(dt_sample)
#                     plt.figure()
#                     # shap_vals is list for multiclass
#                     if isinstance(shap_vals, list):
#                         shap.summary_plot(shap_vals[1], dt_sample, show=False)
#                     else:
#                         shap.summary_plot(shap_vals, dt_sample, show=False)
#                     st.pyplot(plt.gcf())
#                     plt.close()
#             except Exception as e:
#                 st.warning(f"DT SHAP failed: {e}")
#
#     except Exception as e:
#         st.error(f"Decision Tree evaluation error: {e}")
#
#     # -------- LOGISTIC REGRESSION --------
#     st.subheader("📉 Logistic Regression Results")
#     try:
#         lr_model, lr_encoder, lr_scaler, lr_columns = load_model_bundle("lr")
#         if lr_model is None:
#             st.warning("lr_model.pkl not found — skipping Logistic Regression evaluation.")
#         else:
#             try:
#                 X_lr_aligned, info_lr = _align_features_for_model(shared_X_test, lr_model, lr_columns)
#                 if info_lr.get("missing_cols"):
#                     st.warning(f"LR: filling {len(info_lr['missing_cols'])} missing columns with zeros.")
#             except Exception:
#                 X_raw = df.drop(columns=["Status", "Status_encoded"]).copy()
#                 X_lr_all = build_encoded_dataframe(X_raw, encoder=lr_encoder, training_columns=lr_columns, categorical_columns=categorical_columns)
#                 X_lr_aligned = X_lr_all.iloc[shared_X_test.index].reset_index(drop=True)
#
#             # apply scaler if available (scale numeric_columns)
#             try:
#                 if lr_scaler is not None:
#                     present_nums = [c for c in numeric_columns if c in X_lr_aligned.columns]
#                     if present_nums:
#                         X_lr_aligned[present_nums] = lr_scaler.transform(X_lr_aligned[present_nums])
#             except Exception:
#                 st.warning("LR scaler could not be applied; proceeding without scaling.")
#
#             # predict
#             try:
#                 preds_lr = lr_model.predict(X_lr_aligned)
#             except Exception:
#                 preds_lr = lr_model.predict(X_lr_aligned.values)
#
#             y_test = np.array(shared_y_test).astype(int)
#             acc_lr = accuracy_score(y_test, preds_lr)
#             model_results["Logistic Regression"] = acc_lr
#
#             st.metric("Logistic Regression Accuracy", f"{acc_lr*100:.2f}%")
#             st.text("Classification Report:")
#             st.text(classification_report(y_test, preds_lr, digits=4))
#
#             cm_lr = confusion_matrix(y_test, preds_lr)
#             show_confusion(cm_lr, "Logistic Regression - Confusion Matrix", cmap="Purples")
#
#             # top 10 features for LR
#             try:
#                 if hasattr(lr_model, "estimators_") and isinstance(lr_model.estimators_, list):
#                     coefs = []
#                     for est in lr_model.estimators_:
#                         if hasattr(est, "coef_"):
#                             coefs.append(np.abs(est.coef_).mean(axis=0))
#                     if len(coefs) > 0:
#                         avg_coef = np.mean(coefs, axis=0)
#                         coef_series = pd.Series(avg_coef, index=X_lr_aligned.columns)
#                     else:
#                         coef_series = None
#                 elif hasattr(lr_model, "coef_"):
#                     coef_series = pd.Series(np.abs(lr_model.coef_).mean(axis=0), index=X_lr_aligned.columns)
#                 else:
#                     coef_series = None
#
#                 if coef_series is not None:
#                     top10_lr = coef_series.nlargest(10)
#                     fig, ax = plt.subplots(figsize=(7, 5))
#                     top10_lr.sort_values().plot(kind="barh", ax=ax, color="purple")
#                     ax.set_title("📉 Logistic Regression - Top 10 Features (by |coef|)")
#                     st.pyplot(fig)
#                     plt.close(fig)
#             except Exception as e:
#                 st.warning(f"Could not compute LR top features: {e}")
#
#     except Exception as e:
#         st.error(f"Logistic Regression evaluation error: {e}")
#
#     # ---------- Comparison ----------
#     st.markdown("---")
#     st.header("📈 Model Accuracy Comparison")
#     acc_df = pd.DataFrame({
#         "Model": list(model_results.keys()),
#         "Accuracy": [v for v in model_results.values()]
#     }).set_index("Model")
#
#     if not acc_df.empty:
#         st.write(acc_df.style.format("{:.2%}"))
#         st.bar_chart(acc_df)
#
# if __name__ == "__main__":
#     show_visualization()


# visualization.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import shap
import warnings
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from style import load_theme

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ---------------------------
# Helpers
# ---------------------------
def _load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def map_status_to_code(series):
    return series.map({"Dropout": 0, "Graduate": 1, "Enrolled": 2})

def _align_features_for_model(X_df: pd.DataFrame, training_columns_saved):
    """
    Align X_df to training_columns_saved (best-effort). Returns aligned_df, info.
    We purposely ignore model.feature_names_in_ here — backend saved columns are authoritative.
    """
    X = X_df.copy().reset_index(drop=True)
    if training_columns_saved is None:
        return X, {"aligned": False, "reason": "no training_columns provided"}
    aligned = X.reindex(columns=training_columns_saved, fill_value=0)
    info = {
        "aligned": True,
        "expected_count": len(training_columns_saved),
        "got_count": X.shape[1],
        "missing_cols": [c for c in training_columns_saved if c not in X.columns],
        "extra_cols": [c for c in X.columns if c not in training_columns_saved]
    }
    return aligned, info

def show_confusion(cm, title, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def load_model_bundle(model_key):
    """
    Load per-model bundle files (case A).
    returns model, encoder, scaler, columns
    """
    if model_key == "rf":
        model_path = "rf_model.pkl"
        enc_path = "rf_encoder.pkl"
        scl_path = "rf_scaler.pkl"
        col_path = "rf_columns.pkl"
    elif model_key == "dt":
        model_path = "dt_model.pkl"
        enc_path = "dt_encoder.pkl"
        scl_path = "dt_scaler.pkl"
        col_path = "dt_columns.pkl"
    elif model_key == "lr":
        model_path = "lr_model.pkl"
        enc_path = "lr_encoder.pkl"
        scl_path = "lr_scaler.pkl"
        col_path = "lr_columns.pkl"
    else:
        return None, None, None, None

    model = _load_pickle(model_path)
    encoder = _load_pickle(enc_path)
    scaler = _load_pickle(scl_path)
    columns = _load_pickle(col_path)
    return model, encoder, scaler, columns

# ---------------------------
# Main
# ---------------------------
def show_visualization():
    load_theme()
    st.title("🎓 Student Dropout Analysis Dashboard")
    # ---------- Load dataset for visualizations (raw) ----------
    data_path = "data.csv"
    if not os.path.exists(data_path):
        st.error("❌ data.csv not found in working directory.")
        return

    df = pd.read_csv(data_path, delimiter=";")
    st.success(f"✅ Loaded dataset — rows: {df.shape[0]}, cols: {df.shape[1]}")

    # ---------- Basic distribution plots ----------
    st.header("📊 Basic Distributions")
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x="Marital_status", data=df, palette="Set2")
    plt.title("Distribution of Marital Status")
    st.pyplot(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 8))
    sns.countplot(y="Course", data=df, order=df["Course"].value_counts().index, palette="Set2")
    plt.title("Distribution of Courses")
    st.pyplot(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x="Gender", data=df, palette="Set2")
    plt.title("Distribution of Gender")
    st.pyplot(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x="Status", data=df, palette="Set2")
    plt.title("Distribution of Status (Dropout vs Graduate vs Enrolled)")
    st.pyplot(fig)
    plt.close(fig)

    # ---------- Correlation heatmap ----------
    st.header("🔥 Correlation Heatmap (numeric features)")
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    if numeric_df.shape[1] > 0:
        fig = plt.figure(figsize=(14, 12))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric columns found for correlation heatmap.")

    # ---------- Ensure target exists ----------
    if "Status" not in df.columns:
        st.error("❌ 'Status' column missing in data.csv (expected 'Dropout','Graduate','Enrolled').")
        return

    # mapped target (only for display if needed)
    try:
        df["Status_encoded"] = map_status_to_code(df["Status"])
    except Exception:
        pass

    # ---------- Try to load shared processed test set from backend ----------
    shared_X_test = _load_pickle("shared_X_test.pkl")
    shared_y_test = _load_pickle("shared_y_test.pkl")
    shared_X_train = _load_pickle("shared_X_train.pkl")  # optional, for SHAP background

    if shared_X_test is None or shared_y_test is None:
        st.warning("Shared test set not found. This frontend will try an approximate fallback, but results may differ.")
        # fallback: build processed X using get_dummies (best-effort) and create split
        X_raw = df.drop(columns=["Status", "Status_encoded"], errors="ignore").copy()
        X_all = pd.get_dummies(X_raw, drop_first=False)
        y_all = df["Status_encoded"].astype(int).values if "Status_encoded" in df else map_status_to_code(df["Status"]).astype(int).values
        idx = np.arange(len(X_all))
        try:
            idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_all)
        except Exception:
            idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)
        shared_X_test = X_all.iloc[idx_test].reset_index(drop=True)
        shared_y_test = df["Status_encoded"].iloc[idx_test].astype(int).values
        st.info("Fallback test-split created (will probably NOT match backend exactly).")
    else:
        # st.success("✅ Loaded shared_X_test.pkl and shared_y_test.pkl — frontend will match backend evaluation exactly.")
        shared_X_test = shared_X_test.reset_index(drop=True)
        try:
            shared_y_test = np.array(shared_y_test).astype(int)
        except Exception:
            shared_y_test = np.array(shared_y_test)

    # ---------- Numeric columns list (used only if we must re-scale when building from raw) ----------
    numeric_columns = [
        'Previous_qualification_grade','Admission_grade',
        'Curricular_units_1st_sem_credited','Curricular_units_1st_sem_enrolled',
        'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_approved',
        'Age_at_enrollment'
    ]

    model_results = {}

    # ----------------
    # Evaluate RF
    # ----------------
    st.subheader("🌲 Random Forest Results")
    try:
        rf_model, rf_encoder, rf_scaler, rf_columns = load_model_bundle("rf")
        if rf_model is None:
            st.warning("rf_model.pkl not found — skipping Random Forest.")
        else:
            # If backend shared X_test exists and is already processed, use it directly and only align columns
            X_rf_aligned, info_rf = _align_features_for_model(shared_X_test, rf_columns)
            if info_rf.get("missing_cols"):
                st.warning(f"RF: filling {len(info_rf['missing_cols'])} missing columns with zeros.")

            # DO NOT rescale: shared_X_test is already processed by backend (important!)
            preds_rf = rf_model.predict(X_rf_aligned)
            y_test = shared_y_test
            acc_rf = accuracy_score(y_test, preds_rf)
            model_results["Random Forest"] = acc_rf

            st.metric("Random Forest Accuracy", f"{acc_rf*100:.2f}%")
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds_rf, digits=4))

            cm_rf = confusion_matrix(y_test, preds_rf)
            show_confusion(cm_rf, "Random Forest - Confusion Matrix", cmap="Blues")

            # Feature importances: aggregate if OVR
            feat_imp = None
            try:
                if hasattr(rf_model, "estimators_") and isinstance(rf_model.estimators_, list):
                    imps = []
                    for sub in rf_model.estimators_:
                        if hasattr(sub, "feature_importances_"):
                            imps.append(sub.feature_importances_)
                    if len(imps) > 0:
                        feat_imp = np.mean(imps, axis=0)
                elif hasattr(rf_model, "feature_importances_"):
                    feat_imp = rf_model.feature_importances_
            except Exception:
                feat_imp = None

            if feat_imp is not None:
                try:
                    feat_series = pd.Series(feat_imp, index=X_rf_aligned.columns)
                    top10 = feat_series.nlargest(10)
                    fig, ax = plt.subplots(figsize=(7, 5))
                    top10.sort_values().plot(kind="barh", ax=ax)
                    ax.set_title("🌲 Random Forest - Top 10 Important Features")
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Could not plot RF top features: {e}")
            else:
                st.info("RF feature importances not available.")

            # SHAP - TreeExplainer on first estimator (works for RF)
            try:
                st.markdown("**SHAP Explainability (Random Forest)**")
                rf_tree = rf_model.estimators_[0] if hasattr(rf_model, "estimators_") else rf_model
                rf_sample = X_rf_aligned.sample(min(len(X_rf_aligned), 200), random_state=42)
                explainer = shap.TreeExplainer(rf_tree)
                shap_values = explainer.shap_values(rf_sample)
                plt.figure()
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[1], rf_sample, show=False)
                else:
                    shap.summary_plot(shap_values, rf_sample, show=False)
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.warning(f"RF SHAP failed: {e}")

    except Exception as e:
        st.error(f"Random Forest evaluation error: {e}")

    # ----------------
    # Evaluate DT
    # ----------------
    st.subheader("🌳 Decision Tree Results")
    try:
        dt_model, dt_encoder, dt_scaler, dt_columns = load_model_bundle("dt")
        if dt_model is None:
            st.warning("dt_model.pkl not found — skipping Decision Tree.")
        else:
            X_dt_aligned, info_dt = _align_features_for_model(shared_X_test, dt_columns)
            if info_dt.get("missing_cols"):
                st.warning(f"DT: filling {len(info_dt['missing_cols'])} missing columns with zeros.")

            preds_dt = dt_model.predict(X_dt_aligned)
            y_test = shared_y_test
            acc_dt = accuracy_score(y_test, preds_dt)
            model_results["Decision Tree"] = acc_dt

            st.metric("Decision Tree Accuracy", f"{acc_dt*100:.2f}%")
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds_dt, digits=4))

            cm_dt = confusion_matrix(y_test, preds_dt)
            show_confusion(cm_dt, "Decision Tree - Confusion Matrix", cmap="Oranges")

            # feature importances: aggregate for OVR if present
            feat_imp_dt = None
            try:
                if hasattr(dt_model, "estimators_") and isinstance(dt_model.estimators_, list):
                    imps = []
                    for sub in dt_model.estimators_:
                        if hasattr(sub, "feature_importances_"):
                            imps.append(sub.feature_importances_)
                    if len(imps) > 0:
                        feat_imp_dt = np.mean(imps, axis=0)
                elif hasattr(dt_model, "feature_importances_"):
                    feat_imp_dt = dt_model.feature_importances_
            except Exception:
                feat_imp_dt = None

            if feat_imp_dt is not None:
                try:
                    feat_series_dt = pd.Series(feat_imp_dt, index=X_dt_aligned.columns)
                    top10_dt = feat_series_dt.nlargest(10)
                    fig, ax = plt.subplots(figsize=(7, 5))
                    top10_dt.sort_values().plot(kind="barh", ax=ax, color="orange")
                    ax.set_title("🌳 Decision Tree - Top 10 Important Features")
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"Could not plot DT importances: {e}")

            # SHAP for DT: try TreeExplainer on first estimator; fallback to KernelExplainer using shared_X_train
            try:
                st.markdown("**SHAP Explainability (Decision Tree)**")
                dt_tree = dt_model.estimators_[0] if hasattr(dt_model, "estimators_") else dt_model
                dt_sample = X_dt_aligned.sample(min(len(X_dt_aligned), 200), random_state=42)
                try:
                    explainer_dt = shap.TreeExplainer(dt_tree)
                    shap_vals_dt = explainer_dt.shap_values(dt_sample)
                    plt.figure()
                    if isinstance(shap_vals_dt, list):
                        shap.summary_plot(shap_vals_dt[1], dt_sample, show=False)
                    else:
                        shap.summary_plot(shap_vals_dt, dt_sample, show=False)
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception:
                    # fallback KernelExplainer (slower) — use shared_X_train as background if available
                    background = shared_X_train.sample(min(50, len(shared_X_train)), random_state=42) if shared_X_train is not None else X_dt_aligned.sample(min(50, len(X_dt_aligned)), random_state=42)
                    expl = shap.KernelExplainer(dt_model.predict_proba, background)
                    shap_vals = expl.shap_values(dt_sample)
                    plt.figure()
                    if isinstance(shap_vals, list):
                        shap.summary_plot(shap_vals[1], dt_sample, show=False)
                    else:
                        shap.summary_plot(shap_vals, dt_sample, show=False)
                    st.pyplot(plt.gcf())
                    plt.close()
            except Exception as e:
                st.warning(f"DT SHAP failed: {e}")

    except Exception as e:
        st.error(f"Decision Tree evaluation error: {e}")

    # ----------------
    # Evaluate LR
    # ----------------
    st.subheader("📉 Logistic Regression Results")
    try:
        lr_model, lr_encoder, lr_scaler, lr_columns = load_model_bundle("lr")
        if lr_model is None:
            st.warning("lr_model.pkl not found — skipping Logistic Regression.")
        else:
            X_lr_aligned, info_lr = _align_features_for_model(shared_X_test, lr_columns)
            if info_lr.get("missing_cols"):
                st.warning(f"LR: filling {len(info_lr['missing_cols'])} missing columns with zeros.")

            # DO NOT rescale if shared_X_test is provided (backend already scaled)
            try:
                preds_lr = lr_model.predict(X_lr_aligned)
            except Exception:
                preds_lr = lr_model.predict(X_lr_aligned.values)

            y_test = shared_y_test
            acc_lr = accuracy_score(y_test, preds_lr)
            model_results["Logistic Regression"] = acc_lr

            st.metric("Logistic Regression Accuracy", f"{acc_lr*100:.2f}%")
            st.text("Classification Report:")
            st.text(classification_report(y_test, preds_lr, digits=4))

            cm_lr = confusion_matrix(y_test, preds_lr)
            show_confusion(cm_lr, "Logistic Regression - Confusion Matrix", cmap="Purples")

            # Top 10 features for LR via averaged absolute coefficients (if OVR)
            try:
                coef_series = None
                if hasattr(lr_model, "estimators_") and isinstance(lr_model.estimators_, list):
                    coefs = []
                    for est in lr_model.estimators_:
                        if hasattr(est, "coef_"):
                            coefs.append(np.abs(est.coef_).mean(axis=0))
                    if len(coefs) > 0:
                        avg_coef = np.mean(coefs, axis=0)
                        coef_series = pd.Series(avg_coef, index=X_lr_aligned.columns)
                elif hasattr(lr_model, "coef_"):
                    coef_series = pd.Series(np.abs(lr_model.coef_).mean(axis=0), index=X_lr_aligned.columns)

                if coef_series is not None:
                    top10_lr = coef_series.nlargest(10)
                    fig, ax = plt.subplots(figsize=(7, 5))
                    top10_lr.sort_values().plot(kind="barh", ax=ax, color="purple")
                    ax.set_title("📉 Logistic Regression - Top 10 Features (by |coef|)")
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.warning(f"Could not compute LR top features: {e}")

    except Exception as e:
        st.error(f"Logistic Regression evaluation error: {e}")

    # ---------- comparison ----------
    st.markdown("---")
    st.header("📈 Model Accuracy Comparison")
    acc_df = pd.DataFrame({
        "Model": list(model_results.keys()),
        "Accuracy": [v for v in model_results.values()]
    }).set_index("Model")

    if not acc_df.empty:
        st.write(acc_df.style.format("{:.2%}"))
        st.bar_chart(acc_df)

if __name__ == "__main__":
    show_visualization()
