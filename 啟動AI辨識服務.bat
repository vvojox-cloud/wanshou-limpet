@echo off
chcp 65001 >nul
title 萬壽社區 AI 生物辨識服務

echo.
echo  ╔══════════════════════════════════════╗
echo  ║   萬壽社區 AI 生物辨識服務啟動中     ║
echo  ╚══════════════════════════════════════╝
echo.

:: 檢查 Ollama
echo  [1/4] 檢查 Ollama 服務...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if %ERRORLEVEL% NEQ 0 (
    echo  ▶ 啟動 Ollama...
    start "" "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe"
    timeout /t 3 /nobreak >nul
) else (
    echo  ✓ Ollama 已在執行中
)

:: 切換資料夾
echo  [2/4] 切換到網站資料夾...
cd /d "%USERPROFILE%\OneDrive\萬壽社區發展協會\wanshou-limpet"
if %ERRORLEVEL% NEQ 0 (
    echo  ✗ 找不到網站資料夾！
    pause
    exit
)
echo  ✓ 資料夾就緒

:: 啟動 Python 服務（背景執行）
echo  [3/4] 啟動 AI 辨識服務...
start "AI辨識服務" cmd /k "python server.py"
timeout /t 3 /nobreak >nul
echo  ✓ AI 服務啟動中

:: 啟動 ngrok
echo  [4/4] 啟動 ngrok 網路穿透...
echo.
echo  ════════════════════════════════════════
echo   請等待下方出現公開網址（https://...）
echo   複製網址後貼到網站設定頁面
echo   手機用行動網路也能辨識！
echo  ════════════════════════════════════════
echo.
ngrok http 5000