import streamlit as st
import pandas as pd
import os
import zipfile
from utils import eda, visualize
from models import train_tabular, train_image
from tutor import explain

st.set_page_config(page_title="AI Learning Buddy", layout="wide")
st.title("🤖 AI Learning Buddy – Learn ML by Doing")

menu = ["Home", "Upload Data", "EDA", "Train Model", "Tutor"]
choice = st.sidebar.radio("Navigation", menu)

# ---------------- HOME ----------------
if choice == "Home":
    st.markdown("### Welcome to AI Learning Buddy 🎉")
    st.write("""
    This platform helps you **upload datasets (CSV or Images)**, explore them,
    train machine learning models (tabular & images), 
    and learn with AI explanations powered by Gemini.
    """)

# ---------------- UPLOAD DATA ----------------
elif choice == "Upload Data":
    st.subheader("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Images (ZIP)", type=["csv", "zip"])

    if uploaded_file:
        # If CSV file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            st.success("✅ CSV uploaded successfully!")
            st.write(df.head())

        # If ZIP file (images)
        elif uploaded_file.name.endswith(".zip"):
            zip_path = os.path.join("uploads", uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("uploads/images")

            st.success("✅ Images extracted to `uploads/images/`")
            st.info("👉 Now go to **Train Model → Image Classification** to train your model.")

# ---------------- EDA ----------------
elif choice == "EDA":
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload a CSV dataset first.")
    else:
        df = st.session_state['data']
        eda.show_summary(df)
        visualize.plot_distributions(df)
        visualize.plot_heatmap(df)

# ---------------- TRAIN MODEL ----------------
elif choice == "Train Model":
    st.subheader("⚡ Train a Model")
    model_type = st.selectbox(
        "Choose Model Type", 
        ["Logistic Regression", "Random Forest", "XGBoost", "Naive Bayes", "SVM", "Image Classification"]
    )

    # Tabular Models
    if model_type in ["Logistic Regression", "Random Forest", "XGBoost", "Naive Bayes", "SVM"]:
        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload a CSV dataset first.")
        else:
            df = st.session_state['data']
            target_col = st.selectbox("Select Target Column", df.columns)

            if st.button("🚀 Train Tabular Model"):
                results = train_tabular.run(df, target_col, model_type)

                st.write("### ✅ Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results['accuracy']:.2f}")
                col2.metric("Precision", f"{results['precision']:.2f}")
                col3.metric("Recall", f"{results['recall']:.2f}")
                col4.metric("F1-Score", f"{results['f1']:.2f}")

                cm_summary = visualize.plot_confusion_matrix(results["y_test"], results["y_pred"])
                roc_summary = visualize.plot_roc(results["y_test"], results["y_proba"])

                st.session_state['last_results'] = results
                st.session_state['last_model'] = model_type
                st.session_state['last_graph_summary'] = cm_summary + " " + roc_summary

    # Image Model
    elif model_type == "Image Classification":
        image_dir = "uploads/images"
        epochs = st.slider("Epochs", 1, 20, 5)

        if st.button("🚀 Train Image Model"):
            if os.path.exists(image_dir):
                results, history, gradcam_paths = train_image.run(image_dir, epochs=epochs)

                st.write("### ✅ Image Model Results")
                st.metric("Train Accuracy", f"{results['train_accuracy']:.2f}")
                st.metric("Val Accuracy", f"{results['val_accuracy']:.2f}")

                cm_summary = visualize.plot_confusion_matrix(
                    results["classification_report_labels"]["true_labels"], 
                    results["classification_report_labels"]["pred_labels"]
                )
                curve_summary = visualize.plot_training_curves(history)

                st.write("### 🔍 Grad-CAM Explanations")
                for path in gradcam_paths:
                    st.image(path, caption="Grad-CAM", use_column_width=True)

                st.session_state['last_results'] = results
                st.session_state['last_model'] = "Image Classification"
                st.session_state['last_graph_summary'] = cm_summary + " " + curve_summary
            else:
                st.error("❌ Image folder not found! Please upload a ZIP first.")

# ---------------- TUTOR ----------------
elif choice == "Tutor":
    st.subheader("📘 AI Tutor (Gemini-powered)")
    question = st.text_area("Ask me anything about ML concepts, results, or graphs:")
    if st.button("Ask Gemini / Get Guidance"):
        results = st.session_state.get('last_results', None)
        model_type = st.session_state.get('last_model', None)
        graph_summary = st.session_state.get('last_graph_summary', None)
        answer_text = explain.answer(question, results, model_type, graph_summary)
        st.write(answer_text)
