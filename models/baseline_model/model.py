"""
Phase: Model Training Definition
Expects: X_train, X_train_scaled, y_train, and random_state seed
Outputs: Trained RandomForestClassifier and LogisticRegression instances
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_random_forest(X_train, y_train, random_state):
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


def train_logistic_regression(X_train_scaled, y_train, random_state):
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    lr.fit(X_train_scaled, y_train)
    return lr
