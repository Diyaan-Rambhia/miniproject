"""
Phase: Dataset Preparation & Feature Scaling
Expects: Feature matrix X, target labels y, test_size ratio, and random_state seed
Outputs: X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, and fitted StandardScaler
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_train_test(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
