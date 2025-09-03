import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from sklearn.preprocessing import label_binarize

def plot_distributions(df):
    st.write("### 📈 Feature Distributions")
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

def plot_heatmap(df):
    st.write("### 🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Not enough numeric columns for correlation heatmap.")

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / total
    return f"The confusion matrix shows {correct}/{total} correct predictions (~{acc:.2f} accuracy)."


def plot_roc(y_test, y_proba):
    y_proba = np.array(y_proba)

    # If multiclass, do One-vs-Rest ROC
    if len(np.unique(y_test)) > 2:
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]

        fig, ax = plt.subplots()
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")

        ax.plot([0,1], [0,1], "--", color="gray")
        ax.legend()
        st.pyplot(fig)

        return f"The ROC curves for {n_classes} classes are shown above (One-vs-Rest strategy)."
    
    else:  # Binary case
        if y_proba.ndim == 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
        else:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0,1],[0,1],'--')
        ax.legend()
        st.pyplot(fig)

        return f"The ROC curve has an AUC of {roc_auc:.2f}."


def plot_training_curves(history):
    fig, ax = plt.subplots()
    ax.plot(history["accuracy"], label="Train Acc")
    ax.plot(history["val_accuracy"], label="Val Acc")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    best_val = max(history["val_accuracy"])
    return f"The training curve shows validation accuracy peaked at {best_val:.2f}."
