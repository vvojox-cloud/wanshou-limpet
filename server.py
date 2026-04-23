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
- 花笠螺（篦仔）：扁平帽狀，完全無螺旋，緊貼礁石，殼面圓潤黃綠褐色，無明顯放射紋，可採集，帶苦味回甘
- 斗笠螺（海鋼盔）：扁平帽狀，完全無螺旋，緊貼礁石，殼面有清楚放射條紋，灰褐色，可採集，鮮甜爽脆
- 笠螺辨識關鍵：完全扁平如帽子，無螺旋結構，緊貼在礁石表面不動
- 珠螺：圓錐螺旋形，黑褐色，殼口內側有明顯藍綠色珍珠光澤，口蓋厚重
- 釋迦黑鐘螺：圓錐螺旋形，黑色，表面有顆粒突起如釋迦果皮，無珍珠光澤
- 草蓆鐘螺：圓錐螺旋形，墨綠黑色，表面顆粒排列整齊如草蓆
- 滑圓蜑螺：半球形極扁，黑色殼面平滑，有橘黃色小斑紋
- 花斑蜑螺：半球形，黑亮，口蓋白色帶黑斑
- 疣岩螺：細長紡錘形，黑白相間疣狀突起
- 藤壺：白色火山錐形，固著礁石完全不移動，無法爬行
- 織錦芋螺：圓錐形，棕白色網格花紋，有毒絕對勿碰
- 黃寶螺：橢圓形，光滑如瓷器，淺黃白色
- 石蟳：螃蟹類，有螯，扁平身體

辨識重點：
- 扁平帽狀緊貼礁石 → 笠螺（花笠螺或斗笠螺），絕對不是藤壺或鐘螺
- 有放射條紋的帽狀 → 斗笠螺
- 無放射條紋的帽狀 → 花笠螺
- 白色火山形固著不動 → 藤壺
- 螺旋錐形 → 鐘螺或珠螺類

照片不清楚時name填無法辨識，confidence填低。
只回傳JSON一行，不要markdown，不要換行。"""


def parse_response(raw_text):
    try:
        return json.loads(raw_text)
    except:
        pass

    match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    cleaned = re.sub(r'```json|```', '', raw_text).strip()
    try:
        return json.loads(cleaned)
    except:
        pass

    return {
        "name": "辨識完成",
        "latinName": "",
        "confidence": "低",
        "description": raw_text[:100] if raw_text else "無回應",
        "canHarvest": False,
        "danger": False,
        "advice": "AI回應格式異常，請重試"
    }


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
            "prompt": "辨識照片中的潮間帶生物，注意觀察形狀特徵，只回傳JSON。",
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

        parsed = parse_response(raw_text)
        return jsonify(parsed)

    except Exception as e:
        print("錯誤：", str(e))
        return jsonify({
            "name": "辨識失敗",
            "latinName": "",
            "confidence": "低",
            "description": str(e)[:80],
            "canHarvest": False,
            "danger": False,
            "advice": "請確認啟動AI辨識服務.bat 有開著"
        })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": MODEL})


if __name__ == '__main__':
    print("=================================")
    print("  萬壽社區 AI 生物辨識服務啟動中")
    print("  模式：本機 Gemma 12B 🖥️")
    print("  網址：http://localhost:5000")
    print("=================================")
    app.run(host='0.0.0.0', port=5000, debug=False)