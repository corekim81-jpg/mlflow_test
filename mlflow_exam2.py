"""
MLflow 통합 기능 예제 (mlflow_exam2.py)
=====================================

mlflow_exam.py(기본 로깅)를 확장해 MLflow의 주요 기능을 한 파일에서 실습합니다.

  1. Tracking 기본       : params / metrics(step) / tags / 각종 artifact(text, dict, figure, image, table, file)
  2. Dataset 추적        : mlflow.data + log_input
  3. 모델 로깅           : signature, input_example, LoggedModel
  4. 하이퍼파라미터 탐색 : 부모-자식(nested) run
  5. Autolog             : mlflow.sklearn.autolog()
  6. 모델 평가           : mlflow.models.evaluate()
  7. Custom PyFunc 모델  : 전처리+후처리를 포함한 사용자 정의 모델
  8. Model Registry      : 등록, 버전, alias(@champion), 태그/설명, alias로 로드
  9. Run 검색            : mlflow.search_runs() / MlflowClient
 10. 모델 로드 & 추론    : runs:/ , models:/ URI
 11. Tracing             : @mlflow.trace, start_span
 12. Prompt Registry     : mlflow.genai.register_prompt / load_prompt (MLflow 3.x)

실행 전 준비
-----------
  # 터미널 1: Tracking 서버 (Model Registry 를 쓰려면 DB 백엔드 권장)
  mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db

  # 터미널 2
  python mlflow_exam2.py

  서버 없이 로컬에서만 돌리려면:
  (PowerShell) $env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"; python mlflow_exam2.py

각 섹션은 독립적으로 try/except 로 감싸져 있어, 한 기능이 실패해도 나머지는 계속 실행됩니다.
"""

import json
import os
import tempfile
import time

import matplotlib

matplotlib.use("Agg")  # GUI 없이 그림 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 0. 공통 설정
# ---------------------------------------------------------------------------
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "MLflow_All_Features"
REGISTERED_MODEL_NAME = "diabetes_regressor"

mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_experiment_tags({"project": "mlflow_test", "owner": "K", "purpose": "feature-tour"})
client = MlflowClient()

# 데이터: sklearn 내장 diabetes 데이터셋 (다운로드 불필요)
raw = load_diabetes(as_frame=True)
df = raw.frame  # feature 10개 + target
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RESULTS = {}  # 섹션별 성공/실패 요약


def section(title):
    """섹션 실행 데코레이터: 제목 출력 + 예외 격리."""

    def deco(fn):
        def wrapper(*args, **kwargs):
            print(f"\n{'=' * 70}\n▶ {title}\n{'=' * 70}")
            try:
                out = fn(*args, **kwargs)
                RESULTS[title] = "OK"
                return out
            except Exception as e:  # noqa: BLE001
                RESULTS[title] = f"FAIL: {type(e).__name__}: {e}"
                print(f"  ✗ 실패: {e}")
                if mlflow.active_run():
                    mlflow.end_run(status="FAILED")
                return None

        return wrapper

    return deco


def tree_model_kwargs():
    """MLflow 최신 버전은 sklearn 모델을 skops 로 저장하며, 트리 모델은 신뢰 타입 지정이 필요.
    구버전(해당 파라미터 없음)과도 호환되도록 시그니처를 확인해서 넘긴다."""
    import inspect

    if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
        return {"skops_trusted_types": ["sklearn.tree._tree.Tree"]}
    return {}


def regression_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# 1. Tracking 기본 + 2. Dataset 추적 + 3. 모델 로깅
# ---------------------------------------------------------------------------
@section("1~3. Tracking 기본 / Dataset / 모델 로깅")
def basic_tracking():
    with mlflow.start_run(run_name="basic_linear_regression") as run:
        # --- 태그 ---
        mlflow.set_tags({"stage": "baseline", "algorithm": "LinearRegression"})
        mlflow.set_tag("mlflow.note.content", "기본 선형회귀 베이스라인 run 입니다.")  # UI 의 Description

        # --- 파라미터 (단일/여러 개) ---
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_params({"test_size": 0.2, "random_state": 42, "n_features": X.shape[1]})

        # --- Dataset 추적: 학습/평가 데이터를 run 의 input 으로 기록 ---
        train_ds = mlflow.data.from_pandas(
            pd.concat([X_train, y_train], axis=1), name="diabetes_train", targets="target"
        )
        test_ds = mlflow.data.from_pandas(
            pd.concat([X_test, y_test], axis=1), name="diabetes_test", targets="target"
        )
        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(test_ds, context="evaluation")

        # --- 학습 ---
        model = LinearRegression().fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = regression_metrics(y_test, preds)

        # --- 메트릭 (단일/여러 개) ---
        mlflow.log_metrics(metrics)

        # --- step 이 있는 메트릭 (학습 곡선처럼 UI 에서 그래프로 보임) ---
        for step, frac in enumerate(np.linspace(0.2, 1.0, 9)):
            n = int(len(X_train) * frac)
            m = LinearRegression().fit(X_train[:n], y_train[:n])
            mlflow.log_metric("learning_curve_r2", r2_score(y_test, m.predict(X_test)), step=step)

        # --- Artifact: 텍스트 / dict(JSON·YAML) ---
        mlflow.log_text("이 run 은 diabetes 데이터로 학습한 선형회귀입니다.\n", "notes/readme.txt")
        coef = dict(zip(X.columns, model.coef_.round(4).tolist()))
        mlflow.log_dict({"intercept": float(model.intercept_), "coef": coef}, "model_info/coefficients.json")
        mlflow.log_dict({"features": list(X.columns)}, "model_info/features.yaml")

        # --- Artifact: matplotlib figure ---
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y_test, preds, alpha=0.6)
        lim = [y_test.min(), y_test.max()]
        ax.plot(lim, lim, "r--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        mlflow.log_figure(fig, "plots/actual_vs_pred.png")
        plt.close(fig)

        # --- Artifact: numpy 이미지 ---
        img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        mlflow.log_image(img, "plots/random_image.png")

        # --- Artifact: 테이블 (UI 의 Evaluation/Artifacts 탭에서 표로 보임) ---
        table = pd.DataFrame({"actual": y_test.values[:20], "pred": preds[:20]})
        mlflow.log_table(table, artifact_file="tables/sample_predictions.json")

        # --- Artifact: 일반 파일 / 디렉터리 ---
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "test_predictions.csv")
            pd.DataFrame({"actual": y_test, "pred": preds}).to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path, artifact_path="data")
            os.makedirs(os.path.join(tmp, "extra"))
            with open(os.path.join(tmp, "extra", "config.json"), "w") as f:
                json.dump({"lr": "n/a", "note": "directory upload example"}, f)
            mlflow.log_artifacts(os.path.join(tmp, "extra"), artifact_path="extra")

        # --- 모델 로깅: signature + input_example ---
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",  # MLflow 3.x (구버전은 artifact_path="model")
            signature=signature,
            input_example=X_train.head(3),
        )
        print(f"  run_id    : {run.info.run_id}")
        print(f"  model_uri : {model_info.model_uri}")
        print(f"  metrics   : { {k: round(v, 3) for k, v in metrics.items()} }")
        return run.info.run_id, model_info.model_uri


# ---------------------------------------------------------------------------
# 4. 하이퍼파라미터 탐색 (Nested runs)
# ---------------------------------------------------------------------------
@section("4. 하이퍼파라미터 탐색 (Nested runs)")
def hyperparam_search():
    grid = [
        {"n_estimators": n, "max_depth": d} for n in (50, 100) for d in (3, 6, None)
    ]
    best = {"rmse": float("inf")}
    with mlflow.start_run(run_name="rf_grid_search") as parent:
        mlflow.set_tag("stage", "tuning")
        mlflow.log_param("search_space_size", len(grid))
        for i, params in enumerate(grid):
            with mlflow.start_run(run_name=f"trial_{i}", nested=True) as child:
                mlflow.log_params(params)
                rf = RandomForestRegressor(random_state=42, **params).fit(X_train, y_train)
                m = regression_metrics(y_test, rf.predict(X_test))
                mlflow.log_metrics(m)
                print(f"  trial_{i} {params} → rmse={m['rmse']:.2f}")
                if m["rmse"] < best["rmse"]:
                    best = {"rmse": m["rmse"], "params": params, "run_id": child.info.run_id, "model": rf}

        # 부모 run 에 최적 결과 기록 + 최적 모델 저장
        mlflow.log_params({f"best_{k}": v for k, v in best["params"].items()})
        mlflow.log_metric("best_rmse", best["rmse"])
        mlflow.set_tag("best_child_run_id", best["run_id"])
        info = mlflow.sklearn.log_model(
            best["model"],
            name="best_model",
            signature=infer_signature(X_train, best["model"].predict(X_train)),
            input_example=X_train.head(3),
            **tree_model_kwargs(),
        )
        print(f"  ★ best: {best['params']} rmse={best['rmse']:.2f}")
        return parent.info.run_id, info.model_uri


# ---------------------------------------------------------------------------
# 5. Autolog
# ---------------------------------------------------------------------------
@section("5. Autolog (sklearn)")
def autolog_demo():
    # fit() 만 호출하면 파라미터/메트릭/모델이 자동 기록됨
    mlflow.sklearn.autolog(log_input_examples=True, log_models=True, silent=True)
    with mlflow.start_run(run_name="ridge_autolog") as run:
        Ridge(alpha=0.5).fit(X_train, y_train)
    mlflow.sklearn.autolog(disable=True)  # 이후 섹션에 영향 없도록 끄기
    logged = mlflow.get_run(run.info.run_id)
    print(f"  자동 기록 params : {list(logged.data.params)[:5]} ...")
    print(f"  자동 기록 metrics: {list(logged.data.metrics)}")
    return run.info.run_id


# ---------------------------------------------------------------------------
# 6. 모델 평가 (mlflow.models.evaluate)
# ---------------------------------------------------------------------------
@section("6. 모델 평가 (mlflow.models.evaluate)")
def evaluate_demo(model_uri):
    eval_df = pd.concat([X_test, y_test], axis=1)
    with mlflow.start_run(run_name="evaluate_best_model"):
        result = mlflow.models.evaluate(
            model=model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )
    show = {k: round(v, 3) for k, v in result.metrics.items() if k in
            ("mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "r2_score")}
    print(f"  평가 메트릭: {show}")
    print(f"  평가 artifacts: {list(result.artifacts)}")
    return result


# ---------------------------------------------------------------------------
# 7. Custom PyFunc 모델
# ---------------------------------------------------------------------------
class ClippedRegressor(mlflow.pyfunc.PythonModel):
    """sklearn 모델을 감싸 예측값을 [lower, upper] 범위로 자르고, 등급 컬럼을 추가하는 사용자 정의 모델."""

    def __init__(self, lower=25.0, upper=346.0):
        self.lower = lower
        self.upper = upper

    def load_context(self, context):
        import joblib

        self.model = joblib.load(context.artifacts["sk_model"])

    def predict(self, context, model_input, params=None):
        threshold = (params or {}).get("high_risk_threshold", 200.0)
        pred = np.clip(self.model.predict(model_input), self.lower, self.upper)
        return pd.DataFrame({"prediction": pred, "high_risk": pred >= threshold})


@section("7. Custom PyFunc 모델")
def pyfunc_demo():
    import joblib

    base = LinearRegression().fit(X_train, y_train)
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name="custom_pyfunc") as run:
        path = os.path.join(tmp, "lr.joblib")
        joblib.dump(base, path)
        sample_out = pd.DataFrame({"prediction": [150.0], "high_risk": [False]})
        info = mlflow.pyfunc.log_model(
            name="clipped_model",
            python_model=ClippedRegressor(),
            artifacts={"sk_model": path},
            signature=infer_signature(X_train.head(), sample_out, params={"high_risk_threshold": 200.0}),
            input_example=X_train.head(3),
            pip_requirements=["scikit-learn", "pandas", "numpy", "joblib"],
        )
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    out = loaded.predict(X_test.head(5), params={"high_risk_threshold": 180.0})
    print(out.to_string(index=False))
    return info.model_uri


# ---------------------------------------------------------------------------
# 8. Model Registry
# ---------------------------------------------------------------------------
@section("8. Model Registry (등록 / 버전 / alias)")
def registry_demo(baseline_uri, best_uri):
    # (1) 등록: 같은 이름으로 두 번 등록 → v1, v2
    v1 = mlflow.register_model(baseline_uri, REGISTERED_MODEL_NAME)
    v2 = mlflow.register_model(best_uri, REGISTERED_MODEL_NAME)
    print(f"  등록: {REGISTERED_MODEL_NAME} v{v1.version}(baseline), v{v2.version}(best RF)")

    # (2) 설명 / 태그
    client.update_registered_model(REGISTERED_MODEL_NAME, description="Diabetes 진행도 회귀 모델")
    client.set_registered_model_tag(REGISTERED_MODEL_NAME, "task", "regression")
    client.update_model_version(REGISTERED_MODEL_NAME, v1.version, description="LinearRegression baseline")
    client.update_model_version(REGISTERED_MODEL_NAME, v2.version, description="RandomForest (grid search best)")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, v2.version, "validation", "passed")

    # (3) Alias (구 Stage 기능 대체): champion / challenger
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", v2.version)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", v1.version)
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    print(f"  @champion → v{mv.version} / @challenger → v{v1.version}")

    # (4) alias 로 모델 로드
    champion = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
    print(f"  champion 예측(3건): {np.round(champion.predict(X_test.head(3)), 1)}")

    # (5) 버전 검색
    for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"):
        full = client.get_model_version(REGISTERED_MODEL_NAME, v.version)  # aliases 는 get 으로 조회
        print(f"    - v{v.version} aliases={list(full.aliases)} run_id={v.run_id[:8]}")


# ---------------------------------------------------------------------------
# 9. Run 검색
# ---------------------------------------------------------------------------
@section("9. Run 검색 (search_runs / MlflowClient)")
def search_demo():
    # pandas DataFrame 으로 반환
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="metrics.rmse < 70 and attributes.status = 'FINISHED'",
        order_by=["metrics.rmse ASC"],
        max_results=5,
    )
    cols = [c for c in ("run_id", "tags.mlflow.runName", "metrics.rmse", "metrics.r2") if c in runs.columns]
    print(runs[cols].to_string(index=False) if not runs.empty else "  (조건에 맞는 run 없음)")

    # MlflowClient: 태그로 검색, 메트릭 히스토리 조회
    tuned = client.search_runs(
        [experiment.experiment_id], filter_string="tags.stage = 'tuning'", max_results=1
    )
    if tuned:
        print(f"  tuning run: {tuned[0].info.run_name}, best_rmse={tuned[0].data.metrics.get('best_rmse'):.2f}")
    base = client.search_runs([experiment.experiment_id], filter_string="tags.stage = 'baseline'", max_results=1)
    if base:
        hist = client.get_metric_history(base[0].info.run_id, "learning_curve_r2")
        print(f"  learning_curve_r2 history: {[round(m.value, 3) for m in hist]}")
        arts = [a.path for a in client.list_artifacts(base[0].info.run_id)]
        print(f"  baseline artifacts: {arts}")


# ---------------------------------------------------------------------------
# 10. 모델 로드 & 추론
# ---------------------------------------------------------------------------
@section("10. 모델 로드 & 추론")
def load_and_predict(model_uri):
    # (a) flavor 그대로 로드 (sklearn 객체)
    sk_model = mlflow.sklearn.load_model(model_uri)
    print(f"  sklearn 로드: {type(sk_model).__name__}")
    # (b) 범용 pyfunc 로 로드 (어떤 프레임워크든 predict() 하나로)
    py_model = mlflow.pyfunc.load_model(model_uri)
    print(f"  pyfunc 예측: {np.round(py_model.predict(X_test.head(3)), 1)}")
    print(f"  signature  : inputs={len(py_model.metadata.signature.inputs.inputs)}개 컬럼")
    # (c) REST 서빙 방법 안내
    print("\n  [REST 서빙] 다른 터미널에서:")
    print(f'    mlflow models serve -m "models:/{REGISTERED_MODEL_NAME}@champion" -p 5001 --env-manager local')
    print("  호출 예:")
    body = {"dataframe_split": X_test.head(2).to_dict(orient="split")}
    body["dataframe_split"].pop("index", None)
    print(f"    POST http://127.0.0.1:5001/invocations  body={json.dumps(body)[:120]}...")


# ---------------------------------------------------------------------------
# 11. Tracing
# ---------------------------------------------------------------------------
@mlflow.trace(name="preprocess", span_type="PARSER")
def preprocess(record: dict) -> pd.DataFrame:
    return pd.DataFrame([record])[list(X.columns)]


@mlflow.trace(name="predict", span_type="CHAIN")
def traced_predict(model, record: dict) -> dict:
    frame = preprocess(record)
    with mlflow.start_span(name="model_inference", span_type="TOOL") as span:
        span.set_inputs({"n_rows": len(frame)})
        t0 = time.perf_counter()
        value = float(model.predict(frame)[0])
        span.set_outputs({"prediction": value})
        span.set_attribute("latency_ms", (time.perf_counter() - t0) * 1000)
    return {"prediction": value, "high_risk": value >= 200}


@section("11. Tracing (@mlflow.trace / start_span)")
def tracing_demo(model_uri):
    model = mlflow.pyfunc.load_model(model_uri)
    for rec in X_test.head(3).to_dict(orient="records"):
        print(f"  {traced_predict(model, rec)}")
    traces = mlflow.search_traces(locations=[experiment.experiment_id], max_results=3)
    print(f"  최근 trace {len(traces)}건 조회됨 → UI 의 'Traces' 탭에서 span 트리 확인")


# ---------------------------------------------------------------------------
# 12. Prompt Registry (MLflow 3.x, GenAI)
# ---------------------------------------------------------------------------
@section("12. Prompt Registry (mlflow.genai)")
def prompt_demo():
    import mlflow.genai

    name = "diabetes_report_prompt"
    p1 = mlflow.genai.register_prompt(
        name=name,
        template="환자 데이터 {{features}} 에 대한 예측값은 {{prediction}} 입니다. 한 줄로 요약하세요.",
        commit_message="initial",
        tags={"lang": "ko"},
    )
    p2 = mlflow.genai.register_prompt(
        name=name,
        template="[위험도 리포트] 입력 {{features}} → 예측 {{prediction}}. 위험도와 권고사항을 2줄로 작성하세요.",
        commit_message="add risk advice",
    )
    mlflow.genai.set_prompt_alias(name, alias="production", version=p2.version)
    prompt = mlflow.genai.load_prompt(f"prompts:/{name}@production")
    print(f"  v{p1.version}, v{p2.version} 등록 / @production → v{prompt.version}")
    print(f"  format: {prompt.format(features='bmi=0.06', prediction=210.5)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"MLflow {mlflow.__version__} | tracking: {TRACKING_URI} | experiment: {EXPERIMENT_NAME}")

    baseline = basic_tracking() or (None, None)
    tuned = hyperparam_search() or (None, None)
    autolog_demo()
    if tuned[1]:
        evaluate_demo(tuned[1])
    pyfunc_demo()
    if baseline[1] and tuned[1]:
        registry_demo(baseline[1], tuned[1])
    search_demo()
    if tuned[1]:
        load_and_predict(tuned[1])
        tracing_demo(tuned[1])
    prompt_demo()

    print(f"\n{'=' * 70}\n실행 요약\n{'=' * 70}")
    for k, v in RESULTS.items():
        print(f"  [{'✓' if v == 'OK' else '✗'}] {k}" + ("" if v == "OK" else f"  → {v}"))
    print(f"\nMLflow UI: {TRACKING_URI if TRACKING_URI.startswith('http') else 'mlflow ui 실행 후 http://127.0.0.1:5000'}")
