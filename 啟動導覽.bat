@echo off
chcp 65001
echo ==============================================
echo   🌊 萬壽社區笠螺導覽 - 雙 AI 系統啟動程序 🌊
echo ==============================================

echo [1/2] 正在解除跨域限制並啟動 Gemma 3:12B 大腦...
:: 設定環境變數，允許網頁連線到 Ollama
set OLLAMA_ORIGINS="*"
:: 在背景啟動 Ollama 服務
start /B ollama serve

echo [2/2] 正在開啟導覽網頁...
:: 請把下面的 index.html 換成你電腦裡網頁檔案的實際路徑
:: 例如：start "" "C:\Users\rabbi\OneDrive\萬壽社區發展協會\wanshou-limpet\index.html"
start "" "index.html"

echo 啟動完成！請在網頁上進行拍照辨識。
pause