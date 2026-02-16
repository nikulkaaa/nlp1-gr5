from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
import numpy as np

def train_model(model_name: str, X_train: np.ndarray, y_train):
    if model_name == 'logistic_regression':
        model = LogisticRegression()
    elif model_name == 'linear_svc':
        model = LinearSVC()
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: LogisticRegression | LinearSVC, X_test: np.ndarray, Y_test: np.ndarray) -> dict[str, ]:
    Y_pred = model.predict(X_test)
    metrics = {"accuracy": accuracy_score(Y_test, Y_pred)}