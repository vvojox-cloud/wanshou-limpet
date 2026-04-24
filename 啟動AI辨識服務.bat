@echo off
chcp 65001 >nul
title 萬壽社區 AI 生物辨識服務

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║     萬壽社區 AI 生物辨識服務啟動中       ║
echo  ║     中角灣笠螺採集體驗活動專用           ║
echo  ╚══════════════════════════════════════════╝
echo.

:: ═══════════════════════════════════════
:: 步驟 1：確認 Ollama 服務
:: ═══════════════════════════════════════
echo  [1/5] 檢查 Ollama AI 引擎...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if %ERRORLEVEL% NEQ 0 (
    echo  ▶ 啟動 Ollama AI 引擎...
    start "" "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe"
    timeout /t 5 /nobreak >nul
    echo  ✓ Ollama 啟動完成
) else (
    echo  ✓ Ollama 已在執行中
)

:: ═══════════════════════════════════════
:: 步驟 2：預載 Gemma 12B 模型
:: ═══════════════════════════════════════
echo  [2/5] 預載 Gemma 12B 模型（首次約需 30 秒）...
start /B "" ollama run gemma3:12b "你好" >nul 2>&1
timeout /t 8 /nobreak >nul
echo  ✓ Gemma 12B 模型就緒

:: ═══════════════════════════════════════
:: 步驟 3：切換到網站資料夾
:: ═══════════════════════════════════════
echo  [3/5] 切換到網站資料夾...
cd /d "%USERPROFILE%\OneDrive\萬壽社區發展協會\wanshou-limpet"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ✗ 找不到網站資料夾！
    echo  請確認路徑是否正確：
    echo  %USERPROFILE%\OneDrive\萬壽社區發展協會\wanshou-limpet
    pause
    exit
)
echo  ✓ 資料夾就緒

:: ═══════════════════════════════════════
:: 步驟 4：啟動 Flask AI 橋接服務
:: ═══════════════════════════════════════
echo  [4/5] 啟動 Gemma AI 橋接服務...
start "🤖 Gemma AI 服務" cmd /k "title Gemma AI 橋接服務 && python server.py"
timeout /t 4 /nobreak >nul
echo  ✓ AI 橋接服務啟動中

:: ═══════════════════════════════════════
:: 步驟 5：啟動 ngrok 網路穿透
:: ═══════════════════════════════════════
echo  [5/5] 啟動 ngrok 行動網路穿透...
start "🌐 ngrok 穿透服務" cmd /k "title ngrok 行動網路穿透 && ngrok http 5000"
timeout /t 5 /nobreak >nul
echo  ✓ ngrok 啟動中

:: ═══════════════════════════════════════
:: 完成畫面
:: ═══════════════════════════════════════
echo.
echo  ╔══════════════════════════════════════════╗
echo  ║           ✅ 所有服務啟動完成！          ║
echo  ╠══════════════════════════════════════════╣
echo  ║                                          ║
echo  ║  🤖 Gemma 12B AI   → 已就緒             ║
echo  ║  🌐 ngrok 穿透     → 請查看穿透視窗     ║
echo  ║  📱 手機行動網路   → 可以使用           ║
echo  ║                                          ║
echo  ╠══════════════════════════════════════════╣
echo  ║  接下來步驟：                            ║
echo  ║  1. 查看 ngrok 視窗的 Forwarding 網址   ║
echo  ║  2. 複製 https://xxxx.ngrok-free.dev    ║
echo  ║  3. 開啟主持人工具貼上網址              ║
echo  ║  4. 產生 QR Code 讓遊客掃碼             ║
echo  ║                                          ║
echo  ╚══════════════════════════════════════════╝
echo.
echo  正在開啟主持人工具頁面...
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:5500/wanshou-limpet/admin.html"

echo.
echo  ⚠️  請保持此視窗開啟，關閉會停止所有服務
echo.
pause