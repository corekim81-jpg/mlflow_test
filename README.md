# mlflow_test

MLflow 실행 및 기능 테스트 프로젝트.

## 파일
| 파일 | 설명 |
|---|---|
| `mlflow_exam.py` | 기본 예제: 선형회귀 학습 + param/metric/model 로깅 |
| `mlflow_exam2.py` | 통합 예제: Tracking, Dataset, 모델 로깅, Nested runs, Autolog, Evaluate, Custom PyFunc, Model Registry(alias), Run 검색, 모델 로드/서빙, Tracing, Prompt Registry |
| `setup_venv.bat` | Windows용 `.venv` 생성 + 패키지 설치 |
| `requirements.txt` | 의존 패키지 |

## 실행 (Windows PowerShell)
```powershell
.\setup_venv.bat
.\.venv\Scripts\Activate.ps1

# 터미널 1: Tracking 서버 (Model Registry/Prompt 기능을 위해 DB 백엔드 권장)
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

# 터미널 2
python mlflow_exam2.py
```
결과는 http://127.0.0.1:5000 에서 확인합니다.

서버 없이 실행: `$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"; python mlflow_exam2.py`
