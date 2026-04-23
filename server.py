from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import re
import base64
import os
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:12b"

# 照片資料庫路徑
SPECIES_IMG_DIR = "images/species"

# 物種對應表（檔名 → 物種資訊）
SPECIES_DB = {
    "limpet-taiwan": {
        "name": "花笠螺", "latinName": "Cellana toreuma",
        "category": "笠螺類", "canHarvest": True, "danger": False,
        "description": "殼面圓潤黃綠褐色，無明顯放射紋，帶苦味回甘，在地稱苦丕仔",
        "advice": "採集殼徑3cm以上，春季3-5月禁採"
    },
    "limpet-oriental": {
        "name": "斗笠螺", "latinName": "Cellana grata",
        "category": "笠螺類", "canHarvest": True, "danger": False,
        "description": "殼面有清楚放射條紋，灰褐色，肉質鮮甜爽脆，俗稱丕仔",
        "advice": "採集殼徑3cm以上，春季3-5月禁採"
    },
    "nerita-smooth-front": {
        "name": "滑圓蜑螺", "latinName": "Nerita ocellata",
        "category": "蜑螺類", "canHarvest": False, "danger": False,
        "description": "黑色殼面平滑，有橘黃色小斑紋，半球形低矮",
        "advice": "請勿採集，觀察即可"
    },
    "nerita-smooth-back": {
        "name": "滑圓蜑螺", "latinName": "Nerita ocellata",
        "category": "蜑螺類", "canHarvest": False, "danger": False,
        "description": "黑色殼面平滑，有橘黃色小斑紋，半球形低矮",
        "advice": "請勿採集，觀察即可"
    },
    "nerita-japonica": {
        "name": "花斑蜑螺", "latinName": "Nerita japonica",
        "category": "蜑螺類", "canHarvest": False, "danger": False,
        "description": "殼色黑亮，口蓋白色帶黑斑，比滑圓蜑螺更圓更黑",
        "advice": "請勿採集，靜靜觀察即可"
    },
    "turbo-black": {
        "name": "釋迦黑鐘螺", "latinName": "Omphalius nigerrimus",
        "category": "鐘螺類", "canHarvest": False, "danger": False,
        "description": "黑色圓錐螺旋形，表面有顆粒突起如釋迦果皮，無珍珠光澤",
        "advice": "請勿採集，觀察即可"
    },
    "turbo-green": {
        "name": "草蓆鐘螺", "latinName": "Tegula nigerrima",
        "category": "鐘螺類", "canHarvest": False, "danger": False,
        "description": "墨綠黑色圓錐螺旋形，表面顆粒排列整齊如草蓆",
        "advice": "請勿採集，為礁岸藻類控制重要物種"
    },
    "turbo-pearl": {
        "name": "珠螺", "latinName": "Turbo bruneus",
        "category": "鐘螺類", "canHarvest": False, "danger": False,
        "description": "黑褐色圓錐螺旋形，殼口內側有明顯藍綠色珍珠光澤",
        "advice": "請勿採集，可輕拿觀察後放回"
    },
    "cowrie-yellow": {
        "name": "黃寶螺", "latinName": "Cypraea moneta",
        "category": "寶螺類", "canHarvest": False, "danger": False,
        "description": "橢圓形，殼面光滑如瓷器，淺黃白色",
        "advice": "請勿採集，稀少種請特別保護"
    },
    "cowrie-money": {
        "name": "貨幣寶螺", "latinName": "Cypraea moneta",
        "category": "寶螺類", "canHarvest": False, "danger": False,
        "description": "橢圓形，殼面白色至淺黃色，光滑無紋",
        "advice": "請勿採集，稀少種請特別保護"
    },
    "thais-clavigera": {
        "name": "疣岩螺", "latinName": "Thais clavigera",
        "category": "岩螺類", "canHarvest": False, "danger": False,
        "description": "細長紡錘形，黑白相間疣狀突起，肉食性螺類",
        "advice": "請勿採集，殼口銳利小心割傷"
    },
    "purpura": {
        "name": "桃羅螺", "latinName": "Purpura persica",
        "category": "岩螺類", "canHarvest": False, "danger": False,
        "description": "殼面灰藍色帶白色斑點，殼質厚重",
        "advice": "請勿採集，具歷史文化價值"
    },
    "conus-textile": {
        "name": "織錦芋螺", "latinName": "Conus textile",
        "category": "芋螺類", "canHarvest": False, "danger": True,
        "description": "圓錐形，棕白色網格花紋，含神經毒素，曾有致死案例",
        "advice": "⚠️ 絕對禁止觸碰！立即告知講師"
    },
}

# ═══════════════════════════════════════════
# 圖片特徵提取（用於相似度比對）
# ═══════════════════════════════════════════
def extract_features(img):
    """把圖片縮小成 64x64，轉成特徵向量"""
    img = img.convert('RGB').resize((64, 64))
    arr = np.array(img).flatten().astype(float)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def load_db_features():
    """載入資料庫所有照片的特徵（支援每種多張）"""
    db = {}
    for key in SPECIES_DB.keys():
        features_list = []
        # 載入 key.jpg、key-2.jpg、key-3.jpg ... key-20.jpg
        candidates = [key] + [f"{key}-{i}" for i in range(2, 21)]
        for candidate in candidates:
            for ext in ['.jpg', '.jpeg', '.png']:
                path = os.path.join(SPECIES_IMG_DIR, candidate + ext)
                if os.path.exists(path):
                    try:
                        img = Image.open(path)
                        features_list.append(extract_features(img))
                        print(f"✓ 載入：{candidate}{ext}")
                    except Exception as e:
                        print(f"✗ 載入失敗：{candidate} → {e}")
        if features_list:
            # 多張照片取平均特徵值
            db[key] = np.mean(features_list, axis=0)
    return db


def find_best_match(query_features, db_features, threshold=0.85):
    """找出最相似的物種"""
    best_key = None
    best_score = 0

    for key, features in db_features.items():
        score = cosine_similarity(
            query_features.reshape(1, -1),
            features.reshape(1, -1)
        )[0][0]
        if score > best_score:
            best_score = score
            best_key = key

    return best_key, float(best_score)


# 啟動時載入資料庫
print("載入生物照片資料庫...")
DB_FEATURES = load_db_features()
print(f"資料庫載入完成，共 {len(DB_FEATURES)} 種生物")

# ═══════════════════════════════════════════
# AI 第一階段：判斷大類
# ═══════════════════════════════════════════
STAGE1_PROMPT = """你是台灣潮間帶生物辨識專家。
請看這張照片，只判斷生物的大類，用JSON回答：
{"category":"大類名稱","shape":"形狀描述","confidence":"高或中或低"}

大類選項（只能選一個）：
- 笠螺類：扁平帽狀，完全無螺旋，緊貼礁石
- 蜑螺類：半球形，非常扁平，殼面平滑或有斑紋
- 鐘螺類：圓錐螺旋形，黑色或墨綠色
- 寶螺類：橢圓形，光滑如瓷器
- 岩螺類：紡錘形或梨形，有突起
- 芋螺類：圓錐形，有網格花紋，有毒
- 甲殼類：螃蟹或藤壺
- 棘皮動物：海膽或陽燧足
- 無法判斷：照片不清楚

只回傳JSON一行，不要markdown。"""


def ai_classify_category(image_base64):
    """讓 AI 判斷生物大類"""
    payload = {
        "model": MODEL,
        "prompt": "請判斷照片中生物的大類，只回傳JSON。",
        "images": [image_base64],
        "system": STAGE1_PROMPT,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    result = response.json()
    raw = result.get('response', '{}').strip()
    print(f"=== 第一階段（大類判斷）===\n{raw}\n")

    try:
        return json.loads(raw)
    except:
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {"category": "無法判斷", "confidence": "低"}


# ═══════════════════════════════════════════
# 主要辨識路由
# ═══════════════════════════════════════════
@app.route('/identify', methods=['POST'])
def identify():
    try:
        data = request.json
        image_base64 = data.get('image', '')

        if not image_base64:
            return jsonify({"error": "沒有收到圖片"}), 400

        # 處理 base64 前綴
        img_data = image_base64
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        # 解碼圖片
        img_bytes = base64.b64decode(img_data)
        query_img = Image.open(BytesIO(img_bytes))
        query_features = extract_features(query_img)

        # ── 第一階段：AI 判斷大類 ──
        category_result = ai_classify_category(img_data)
        category = category_result.get('category', '無法判斷')
        ai_confidence = category_result.get('confidence', '低')

        if category == '無法判斷' or ai_confidence == '低':
            return jsonify({
                "name": "無法辨識",
                "latinName": "",
                "confidence": "低",
                "description": "照片不夠清晰，請重新拍攝。建議從側面45度角拍攝，讓生物佔畫面一半以上。",
                "canHarvest": False,
                "danger": False,
                "advice": "靠近拍攝，側面角度效果最好"
            })

        # ── 第二階段：圖片相似度比對 ──
        # 只比對同大類的物種
        category_keys = {
            k for k, v in SPECIES_DB.items()
            if v.get('category') == category
        }

        # 先在同大類中比對
        filtered_db = {
            k: v for k, v in DB_FEATURES.items()
            if k in category_keys
        }

        # 如果同類沒有照片，用全部比對
        if not filtered_db:
            filtered_db = DB_FEATURES

        best_key, similarity = find_best_match(query_features, filtered_db)

        print(f"=== 第二階段（圖片比對）===")
        print(f"大類：{category}")
        print(f"最佳比對：{best_key}，相似度：{similarity:.2%}")
        print(f"===========================")

        # ── 相似度判斷 ──
        if similarity >= 0.90:
            # 高度相似，直接給答案
            species = SPECIES_DB[best_key]
            confidence = "高"
        elif similarity >= 0.80:
            # 中度相似，給答案但標記為中信心度
            species = SPECIES_DB[best_key]
            confidence = "中"
        else:
            # 相似度不足，回傳大類資訊
            return jsonify({
                "name": f"疑似{category}（無法精確比對）",
                "latinName": "",
                "confidence": "低",
                "description": f"AI判斷為{category}，但與資料庫照片相似度僅{similarity:.0%}，建議對照圖鑑確認。",
                "canHarvest": False,
                "danger": False,
                "advice": "請翻開圖鑑比對，或請講師協助確認"
            })

        return jsonify({
            "name": species["name"],
            "latinName": species["latinName"],
            "confidence": confidence,
            "description": species["description"],
            "canHarvest": species["canHarvest"],
            "danger": species["danger"],
            "advice": species["advice"],
            "similarity": f"{similarity:.0%}"
        })

    except Exception as e:
        print(f"錯誤：{str(e)}")
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
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "db_count": len(DB_FEATURES)
    })


if __name__ == '__main__':
    print("=================================")
    print("  萬壽社區 AI 生物辨識服務啟動中")
    print("  模式：AI大類判斷 + 圖片比對 🔍")
    print("  網址：http://localhost:5000")
    print("=================================")
    app.run(host='0.0.0.0', port=5000, debug=False)