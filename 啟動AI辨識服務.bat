@echo off
chcp 65001 >nul
title 萬壽社區 AI 生物辨識服務

echo.
echo  ╔══════════════════════════════════╗
echo  ║   萬壽社區 AI 生物辨識服務啟動中   ║
echo  ╚══════════════════════════════════╝
echo.

:: 檢查 Ollama 是否在執行
echo  [1/3] 檢查 Ollama 服務...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if %ERRORLEVEL% NEQ 0 (
    echo  ▶ 啟動 Ollama...
    start "" "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe"
    timeout /t 3 /nobreak >nul
) else (
    echo  ✓ Ollama 已在執行中
)

:: 切換到網站資料夾
echo  [2/3] 切換到網站資料夾...
cd /d "%USERPROFILE%\OneDrive\萬壽社區發展協會\wanshou-limpet"
if %ERRORLEVEL% NEQ 0 (
    echo  ✗ 找不到網站資料夾！
    pause
    exit
)
echo  ✓ 資料夾就緒

:: 啟動 Python 服務
echo  [3/3] 啟動 AI 辨識服務...
echo.
echo  ════════════════════════════════════
echo   服務啟動成功！請勿關閉此視窗
echo   網站網址：開啟瀏覽器輸入下方網址
echo   http://127.0.0.1:5500/wanshou-limpet/species.html
echo  ════════════════════════════════════
echo.
echo   按 Ctrl+C 可停止服務
echo.

python server.py
pause