from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def run(df, target_col, model_type):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "Random Forest":
        model = RandomForestClassifier()
    elif model_type == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif model_type == "Naive Bayes":
        model = GaussianNB()
    elif model_type == "SVM":
        model = SVC(probability=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist()
    }
    return results
