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
SPECIES_IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "species")

SPECIES_DB = {
    "花笠螺":    {"latinName": "Cellana toreuma",     "category": "笠螺類", "canHarvest": True,  "danger": False, "description": "殼面圓潤黃綠褐色，無明顯放射紋，帶苦味回甘，在地稱苦丕仔。",   "advice": "採集殼徑3cm以上，春季3-5月禁採。"},
    "斗笠螺":    {"latinName": "Cellana grata",        "category": "笠螺類", "canHarvest": True,  "danger": False, "description": "殼面有清楚放射條紋，灰褐色，肉質鮮甜爽脆，俗稱丕仔。",       "advice": "採集殼徑3cm以上，春季3-5月禁採。"},
    "蜑螺":      {"latinName": "Nerita sp.",            "category": "蜑螺類", "canHarvest": False, "danger": False, "description": "黑褐色半球形，殼面平滑，是中角灣礁岸最常見的螺類之一。",     "advice": "請勿採集，觀察即可。"},
    "珠螺":      {"latinName": "Turbo bruneus",         "category": "鐘螺類", "canHarvest": False, "danger": False, "description": "殼口內側有美麗的藍綠色珍珠光澤，口蓋厚重圓滑。",             "advice": "請勿採集，可輕拿觀察後放回。"},
    "黑鐘螺":    {"latinName": "Omphalius nigerrimus",  "category": "鐘螺類", "canHarvest": False, "danger": False, "description": "黑色圓錐螺旋形，表面有顆粒突起，口蓋內側帶珍珠光澤。",       "advice": "請勿採集，注意礁石濕滑。"},
    "草蓆鐘螺":  {"latinName": "Tegula nigerrima",      "category": "鐘螺類", "canHarvest": False, "danger": False, "description": "墨綠黑色，表面顆粒排列整齊如草蓆，以礁石藻類為食。",         "advice": "請勿採集，為礁岸生態重要物種。"},
    "岩螺":      {"latinName": "Thais sp.",             "category": "岩螺類", "canHarvest": False, "danger": False, "description": "菱形或紡錘形，殼表有明顯疣狀突起，在地稱辣螺，肉食性。",     "advice": "請勿採集，殼口銳利小心割傷。"},
    "織錦芋螺":  {"latinName": "Conus textile",         "category": "芋螺類", "canHarvest": False, "danger": True,  "description": "棕白色網格花紋，含神經毒素，曾有致死案例！",                 "advice": "⚠️ 絕對禁止觸碰！立即告知講師！"},
    "黃寶螺":    {"latinName": "Cypraea moneta",        "category": "寶螺類", "canHarvest": False, "danger": False, "description": "殼表光滑如瓷器，淺黃白色，歷史上曾用作貨幣，中角灣稀有種。", "advice": "請勿採集，稀少種請特別保護。"},
    "阿拉伯寶螺": {"latinName": "Mauritia arabica",     "category": "寶螺類", "canHarvest": False, "danger": False, "description": "殼面有獨特棕色網格花紋，腹面有明顯細齒紋路，中角灣稀客。",   "advice": "請勿採集，觀賞後輕放回原位。"},
}

# 阿超預設說話（Gemma 失敗時備用）
CHAOSHAO_DEFAULT = {
    "花笠螺":    "你看你看！這是我們的「苦丕仔」！黃綠色殼面，沒有放射紋。苦中帶甘，老饕才懂這味道！3公分以上才能採喔！",
    "斗笠螺":    "這個厲害了！「丕仔」來啦！你看那個放射條紋多漂亮。肉質Q脆甜甜的，第一次採集的朋友最適合！",
    "蜑螺":      "哎呦！這是蜑螺啦，中角灣礁岸到處都是！殼面平滑黑黑的，是我們這邊的老鄰居，請觀察就好！",
    "珠螺":      "你看你看！把牠翻過來，殼口有沒有藍綠色珍珠光澤？漂亮吧！大自然送的珠寶，看看就好，放回去！",
    "黑鐘螺":    "這是黑鐘螺！黑色錐形，表面有小顆粒，感覺像釋迦皮。礁岸很常見的好鄰居，請勿採集！",
    "草蓆鐘螺":  "哎！草蓆鐘螺！你看那個顆粒排列，真的很像草蓆編出來的。牠是礁岸清道夫，負責吃藻類，很重要！",
    "岩螺":      "這是辣螺！菱形的殼，有疣狀突起。注意喔，殼口很銳利，不小心會割傷手，觀察就好！",
    "織錦芋螺":  "🚨 危險！織錦芋螺！有神經毒素！花紋再漂亮也不能碰！立刻告知講師！保持距離！",
    "黃寶螺":    "哎呦！黃寶螺！古代用來當錢幣的！現在很稀有了，在中角灣看到算你運氣好！輕輕放回去！",
    "阿拉伯寶螺": "你看你看！阿拉伯寶螺！殼面的花紋像阿拉伯文字，很神奇吧！中角灣的稀客，觀察後放回原位！",
}

GEMMA_PERSONALITY_PROMPT = """你是「萬壽社區的海洋學家阿超」，在金山中角灣研究潮間帶生物超過30年，說話幽默親切，喜歡用台語俗稱，常說「哎呦」、「你看你看」、「這個厲害了」。

這張照片已經確認是：{species_name}（{latin_name}）

請用阿超的幽默口吻，說一句介紹這個生物的話，包含在地俗稱和一個有趣知識點或注意事項，50字以內，口語化。

同時回傳在地俗稱（如果有的話）。

只用以下JSON格式回答，不要其他文字：
{{"local_name": "在地俗稱（沒有就填空字串）", "personality_msg": "阿超說的話"}}"""


def extract_features(img):
    img = img.convert('RGB').resize((64, 64))
    arr = np.array(img).flatten().astype(float)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def load_db_features():
    db = {}
    for key in SPECIES_DB.keys():
        features_list = []
        candidates = [key] + [f"{key}-{i}" for i in range(2, 21)]
        for candidate in candidates:
            for ext in ['.jpg', '.jpeg', '.png']:
                path = os.path.join(SPECIES_IMG_DIR, candidate + ext)
                if os.path.exists(path):
                    try:
                        img = Image.open(path)
                        features_list.append(extract_features(img))
                        print(f"  ✓ 載入：{candidate}{ext}")
                    except Exception as e:
                        print(f"  ✗ 失敗：{candidate} → {e}")
        if features_list:
            db[key] = np.mean(features_list, axis=0)
    return db


def find_best_match(query_features, db_features):
    best_key, best_score = None, 0
    for key, features in db_features.items():
        score = cosine_similarity(
            query_features.reshape(1, -1),
            features.reshape(1, -1)
        )[0][0]
        if score > best_score:
            best_score = score
            best_key = key
    return best_key, float(best_score)


print("\n載入生物照片資料庫...")
DB_FEATURES = load_db_features()
print(f"✓ 資料庫載入完成，共 {len(DB_FEATURES)} 種生物\n")


def gemma_get_personality(image_base64, species_name, latin_name):
    """只讓 Gemma 負責產生阿超的說話，不負責修改辨識結果"""
    try:
        payload = {
            "model": MODEL,
            "prompt": GEMMA_PERSONALITY_PROMPT.format(
                species_name=species_name,
                latin_name=latin_name
            ),
            "images": [image_base64],
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        result = response.json()
        raw = result.get('response', '{}').strip()
        print(f"  Gemma 阿超回應：{raw[:200]}")

        try:
            data = json.loads(raw)
            return data.get('local_name', ''), data.get('personality_msg', '')
        except:
            match = re.search(r'\{.*?\}', raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get('local_name', ''), data.get('personality_msg', '')
                except:
                    pass

        # 如果 JSON 解析失敗，直接把回應當成阿超的話
        clean = re.sub(r'```json|```|\{.*?\}', '', raw, flags=re.DOTALL).strip()
        if clean and len(clean) > 5:
            return '', clean[:100]

    except Exception as e:
        print(f"  Gemma 錯誤：{e}")

    return '', ''


@app.route('/identify', methods=['POST'])
def identify():
    try:
        data = request.json
        image_base64 = data.get('image', '')
        if not image_base64:
            return jsonify({"error": "沒有收到圖片"}), 400

        img_data = image_base64
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        # ── 圖片比對（主要辨識）──
        img_bytes = base64.b64decode(img_data)
        query_img = Image.open(BytesIO(img_bytes))
        query_features = extract_features(query_img)
        best_key, similarity = find_best_match(query_features, DB_FEATURES)

        print(f"\n[辨識請求]")
        print(f"  圖片比對：{best_key}（相似度 {similarity:.1%}）")

        # 相似度太低
        if similarity < 0.70:
            return jsonify({
                "name": "無法辨識",
                "latinName": "",
                "confidence": "低",
                "description": "照片不夠清晰或角度不佳，建議從側面45度角重新拍攝。",
                "canHarvest": False,
                "danger": False,
                "advice": "靠近拍攝，讓生物佔畫面一半以上",
                "similarity": f"{similarity:.0%}",
                "method": "圖片比對",
                "local_name": "",
                "personality_msg": "哎呦！這張照片太模糊了啦！阿超我看了30年的貝類，這個真的認不出來。靠近一點，側面45度角拍，保證給你準確答案！"
            })

        # ── 圖片比對結果確定，Gemma 只負責說阿超的話 ──
        final_name = best_key
        confidence = "高" if similarity >= 0.90 else "中" if similarity >= 0.80 else "低"
        species = SPECIES_DB[final_name]

        print(f"  Gemma 產生阿超說話中...")
        local_name, personality_msg = gemma_get_personality(
            img_data, final_name, species['latinName']
        )

        # Gemma 失敗時用預設說話
        if not personality_msg:
            personality_msg = CHAOSHAO_DEFAULT.get(final_name, f"這是{final_name}，{species['description']}")
            print(f"  使用預設阿超說話")

        return jsonify({
            "name": final_name,
            "latinName": species["latinName"],
            "confidence": confidence,
            "description": species["description"],
            "canHarvest": species["canHarvest"],
            "danger": species["danger"],
            "advice": species["advice"],
            "similarity": f"{similarity:.0%}",
            "method": "圖片比對 + 阿超解說",
            "local_name": local_name,
            "personality_msg": personality_msg
        })

    except Exception as e:
        print(f"  錯誤：{e}")
        return jsonify({
            "name": "辨識失敗",
            "latinName": "",
            "confidence": "低",
            "description": str(e)[:80],
            "canHarvest": False,
            "danger": False,
            "advice": "請確認啟動AI辨識服務.bat 有開著",
            "local_name": "",
            "personality_msg": "哎呦！阿超我今天狀況不太好，請重新試試看！"
        })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "db_count": len(DB_FEATURES),
        "mode": "圖片比對辨識 + Gemma阿超解說"
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  萬壽社區 AI 生物辨識服務啟動中")
    print("  模式：圖片比對 + 阿超人格解說")
    print("  網址：http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)