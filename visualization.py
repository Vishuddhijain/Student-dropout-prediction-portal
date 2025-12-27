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
