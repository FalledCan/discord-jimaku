@echo off
rem Build DiscordJimaku.exe  (output: dist\DiscordJimaku\)
cd /d "%~dp0"
if not exist .venv (
  py -3.12 -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
.venv\Scripts\pyinstaller.exe --noconfirm --clean --windowed --name DiscordJimaku ^
  --paths src ^
  --collect-all faster_whisper ^
  --collect-all ctranslate2 ^
  --collect-binaries nvidia.cublas ^
  --collect-binaries nvidia.cudnn ^
  --collect-binaries onnxruntime ^
  --collect-all sherpa_onnx ^
  --collect-all soxr ^
  src\main.py
echo.
echo Done: dist\DiscordJimaku\DiscordJimaku.exe
