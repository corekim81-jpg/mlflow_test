import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# 가상의 데이터 생성
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 0부터 10까지의 랜덤 값
y = 2.5 * X.squeeze() + np.random.randn(100) * 2  # y = 2.5x + noise

# 데이터프레임으로 변환
data = pd.DataFrame(data=np.hstack((X, y.reshape(-1, 1))), columns=["feature", "target"])

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data[["feature"]], data["target"], test_size=0.2, random_state=42)

# MLflow로 실험 시작
# mlflow.set_experiment("my_experiment")  # 실험 이름 설정

# 실험 이름 설정
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("LinearRegression_test")

with mlflow.start_run():
    
    mlflow.log_param("learning_rate", 0.01)
    # mlflow.log_metric("accuracy", 0.92)
    # mlflow.log_artifact("model.pkl")
        
    # 모델 훈련
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측 및 메트릭 기록
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # 매개변수 및 메트릭 기록
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)

    # 모델 저장
    mlflow.sklearn.log_model(model, "model")

    print(f"Logged model with MSE: {mse}")

# MLflow UI를 통해 결과 확인
print("MLflow UI를 실행하려면 'mlflow ui' 명령어를 사용하세요.")
