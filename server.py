from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import re

app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"

SYSTEM_PROMPT = """你是一個台灣潮間帶生物辨識專家，專門辨識台灣北部礁岸的海洋生物。
請辨識照片中的生物，只用以下JSON格式回答，不要有其他文字：
{"name":"生物中文名稱","latinName":"學名","confidence":"高或中或低","description":"簡短說明50字內","canHarvest":false,"danger":false,"advice":"建議30字內"}

台灣北部礁岸常見物種外觀特徵：
- 珠螺：黑褐色圓錐形螺，殼口內側有明顯藍綠色珍珠光澤，口蓋厚重圓滑，常見於低潮線水坑及礁石縫隙
- 釋迦黑鐘螺：黑色，表面有顆粒突起如釋迦果皮，無珍珠光澤，口蓋圓形石灰質
- 草蓆鐘螺：墨綠黑色，表面顆粒排列整齊如草蓆編織，比釋迦黑鐘螺更扁平
- 滑圓蜑螺：黑色殼面十分平滑，有橘黃色小斑紋，半球形低矮，口蓋石灰質白色
- 花斑蜑螺：黑亮，口蓋白色帶黑斑，比滑圓蜑螺更圓更黑
- 台灣笠螺：扁平帽狀殼，無螺旋結構，緊貼附著礁石，俗稱丕仔，可採集
- 疣岩螺：黑白相間明顯疣狀突起，細長紡錘形，肉食性
- 玉黍螺：小型1-2cm，灰黃黑色多樣，高潮帶礁石縫隙常見
- 藤壺：白色火山錐形，固著礁石完全不移動，退潮後關閉殼口
- 織錦芋螺：圓錐形，棕白色網格花紋華麗，有毒絕對勿碰
- 黃寶螺：光滑如瓷器，橢圓形，淺黃白色，活體有黑色外套膜覆蓋
- 石蟳：螃蟹類，螯強而有力，扁平身體，躲於礁石水坑
- 燒酒海膽：球形，短棘刺，灰綠色，低潮線附近礁石上
- 陽燧足：五條細長腕可靈活彎曲，躲於石塊下方陰暗處

辨識重點提示：
- 看到圓錐形黑螺且殼口有珍珠光澤 → 珠螺
- 看到圓錐形黑螺且殼口無光澤有顆粒 → 釋迦黑鐘螺或草蓆鐘螺
- 看到半球形黑螺非常平滑 → 蜑螺類
- 看到扁平帽狀緊貼礁石 → 笠螺
- 看到網格花紋圓錐螺 → 織錦芋螺（危險）

照片不清楚時name填無法辨識，confidence填低。
只回傳JSON一行，不要markdown，不要換行。"""

@app.route('/identify', methods=['POST'])
def identify():
    try:
        data = request.json
        image_base64 = data.get('image', '')

        if not image_base64:
            return jsonify({"error": "沒有收到圖片"}), 400

        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]

        payload = {
            "model": MODEL,
            "prompt": "辨識照片中的潮間帶生物，注意觀察殼口光澤、外形、顆粒紋路等細節，只回傳JSON。",
            "images": [image_base64],
            "system": SYSTEM_PROMPT,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        result = response.json()
        raw_text = result.get('response', '').strip()

        print("=== AI 原始回應 ===")
        print(raw_text)
        print("==================")

        # 方法1：直接解析
        try:
            parsed = json.loads(raw_text)
            return jsonify(parsed)
        except:
            pass

        # 方法2：找 { } 之間的內容
        match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return jsonify(parsed)
            except:
                pass

        # 方法3：移除 markdown 後再試
        cleaned = re.sub(r'```json|```', '', raw_text).strip()
        try:
            parsed = json.loads(cleaned)
            return jsonify(parsed)
        except:
            pass

        # 都失敗回傳原始文字
        return jsonify({
            "name": "辨識完成",
            "latinName": "",
            "confidence": "低",
            "description": raw_text[:100] if raw_text else "無回應",
            "canHarvest": False,
            "danger": False,
            "advice": "AI回應格式異常，請重試"
        })

    except Exception as e:
        print("錯誤：", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": MODEL})

if __name__ == '__main__':
    print("=================================")
    print("  萬壽社區 AI 生物辨識服務啟動中")
    print("  網址：http://localhost:5000")
    print("=================================")
    app.run(host='0.0.0.0', port=5000, debug=False)