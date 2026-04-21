from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import os

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier()
    svm = SVC()
    xgb = XGBClassifier(eval_metric='mlogloss')

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

    print("RF:", rf_acc)
    print("SVM:", svm_acc)
    print("XGB:", xgb_acc)

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, "models/xgboost_model.pkl")

    # Save accuracy plot
    os.makedirs("results", exist_ok=True)
    models = ["RF", "SVM", "XGB"]
    scores = [rf_acc, svm_acc, xgb_acc]

    plt.figure()
    plt.bar(models, scores)
    plt.title("Model Accuracy Comparison")
    plt.savefig("results/accuracy.png")

    return xgb