import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드 및 전처리
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow 실험 시작
mlflow.set_experiment("iris_classification")

with mlflow.start_run():
    # 모델 파라미터 설정
    n_estimators = 100
    max_depth = 7

    # 모델 학습
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    # 예측 및 평가
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # MLflow에 파라미터와 메트릭 기록
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # 모델 저장
    mlflow.sklearn.log_model(rf, "random_forest_model")

    print(f"Accuracy: {accuracy}")