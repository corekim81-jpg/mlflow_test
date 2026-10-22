@echo off
chcp 65001 >nul
REM mlflow_test 프로젝트 가상환경(.venv) 생성 및 패키지 설치
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    set PY=py -3
) else (
    set PY=python
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] .venv 생성 중...
    %PY% -m venv .venv || (echo Python을 찾을 수 없습니다. Python 3.10+ 설치 후 다시 실행하세요. & pause & exit /b 1)
) else (
    echo [1/3] .venv 가 이미 존재합니다.
)

echo [2/3] pip 업그레이드...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [3/3] requirements.txt 설치...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo 완료! 활성화:  .venv\Scripts\activate
echo MLflow UI 실행: mlflow ui
pause
