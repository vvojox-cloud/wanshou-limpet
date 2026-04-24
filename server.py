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
SPECIES_IMG_DIR = "images/species"

SPECIES_DB = {
    "花笠螺":    {"latinName": "Cellana toreuma",     "category": "笠螺類", "canHarvest": True,  "danger": False, "description": "殼面圓潤黃綠褐色，無明顯放射紋，帶苦味回甘，在地稱篦仔。",     "advice": "採集殼徑3cm以上，春季3-5月禁採。"},
    "斗笠螺":    {"latinName": "Cellana grata",        "category": "笠螺類", "canHarvest": True,  "danger": False, "description": "殼面有清楚放射條紋，灰褐色，肉質鮮甜爽脆，俗稱海鋼盔。",     "advice": "採集殼徑3cm以上，春季3-5月禁採。"},
    "蜑螺":      {"latinName": "Nerita sp.",            "category": "蜑螺類", "canHarvest": False, "danger": False, "description": "黑褐色半球形，殼面平滑，是中角灣礁岸最常見的螺類之一。",     "advice": "請勿採集，觀察即可。"},
    "珠螺":      {"latinName": "Turbo bruneus",         "category": "鐘螺類", "canHarvest": False, "danger": False, "description": "殼口內側有美麗的藍綠色珍珠光澤，口蓋厚重圓滑。",             "advice": "請勿採集，可輕拿觀察後放回。"},
    "黑鐘螺":    {"latinName": "Omphalius nigerrimus",  "category": "鐘螺類", "canHarvest": False, "danger": False, "description": "黑色圓錐螺旋形，表面有顆粒突起，口蓋內側帶珍珠光澤。",       "advice": "請勿採集，注意礁石濕滑。"},
    "草蓆鐘螺":  {"latinName": "Tegula nigerrima",      "category": "鐘螺類", "canHarvest": False, "danger": False, "description": "墨綠黑色，表面顆粒排列整齊如草蓆，以礁石藻類為食。",         "advice": "請勿採集，為礁岸生態重要物種。"},
    "岩螺":      {"latinName": "Thais sp.",             "category": "岩螺類", "canHarvest": False, "danger": False, "description": "菱形或紡錘形，殼表有明顯疣狀突起，肉食性螺類。",             "advice": "請勿採集，殼口銳利小心割傷。"},
    "織錦芋螺":  {"latinName": "Conus textile",         "category": "芋螺類", "canHarvest": False, "danger": True,  "description": "棕白色網格花紋，含神經毒素，曾有致死案例！",                 "advice": "⚠️ 絕對禁止觸碰！立即告知講師！"},
    "黃寶螺":    {"latinName": "Cypraea moneta",        "category": "寶螺類", "canHarvest": False, "danger": False, "description": "殼表光滑如瓷器，淺黃白色，歷史上曾用作貨幣。",             "advice": "請勿採集，稀少種請特別保護。"},
    "阿拉伯寶螺": {"latinName": "Mauritia arabica",     "category": "寶螺類", "canHarvest": False, "danger": False, "description": "殼面有獨特棕色網格花紋，腹面有明顯細齒紋路。",             "advice": "請勿採集，觀賞後輕放回原位。"},
}

GEMMA_PROMPT = """你是「萬壽社區的海洋學家阿超」，在金山中角灣研究潮間帶生物超過30年，說話風格幽默親切，喜歡用在地台語俗稱介紹生物，偶爾會說「哎呦」、「你看你看」、「這個厲害了」之類的語氣詞。

照片比對系統判斷這可能是：{candidate}（相似度 {similarity}%）。

請根據照片，用以下JSON格式回答：
{{"confirmed": true或false, "name": "最終物種名稱", "confidence": "高或中或低", "reason": "判斷理由15字內", "local_name": "在地俗稱或台語名（如果有的話）", "personality_msg": "用阿超的幽默口吻說一句話介紹這個生物，包含在地名稱和一個有趣的知識點或注意事項，50字以內，口語化，像在跟遊客聊天"}}

中角灣10種貝類在地知識：
- 花笠螺：在地叫「苦丕仔」，帶微苦回甘，是老饕最愛的「大人味」，春季3-5月禁採
- 斗笠螺：在地叫「丕仔」，肉質Q脆鮮甜，初學者首選，春季3-5月禁採
- 蜑螺：礁岸最常見，殼很平滑，請勿採集
- 珠螺：殼口有漂亮珍珠光澤，像天然珠寶，請勿採集
- 黑鐘螺：黑色錐形，表面有顆粒，請勿採集
- 草蓆鐘螺：顆粒排列像草蓆，礁岸清道夫，請勿採集
- 岩螺：在地叫辣螺，微辣口感，肉食性，會鑽孔吃藤壺，殼口銳利注意
- 織錦芋螺：超危險！有毒會致死！發現立即遠離告知講師！
- 黃寶螺：古代貨幣，稀有珍貴，請勿採集
- 阿拉伯寶螺：殼面像阿拉伯文字花紋，稀有，請勿採集

只回傳JSON，不要其他文字，不要markdown。"""


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


def gemma_verify(image_base64, candidate, similarity):
    try:
        payload = {
            "model": MODEL,
            "prompt": GEMMA_PROMPT.format(
                candidate=candidate,
                similarity=f"{similarity:.0%}"
            ),
            "images": [image_base64],
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        result = response.json()
        raw = result.get('response', '{}').strip()
        print(f"  Gemma 回應：{raw[:200]}")
        try:
            return json.loads(raw)
        except:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
    except Exception as e:
        print(f"  Gemma 錯誤：{e}")
    return None


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

        img_bytes = base64.b64decode(img_data)
        query_img = Image.open(BytesIO(img_bytes))
        query_features = extract_features(query_img)
        best_key, similarity = find_best_match(query_features, DB_FEATURES)

        print(f"\n[辨識請求]")
        print(f"  圖片比對：{best_key}（相似度 {similarity:.1%}）")

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
                "personality_msg": "哎呦！這張照片太模糊了啦！阿超我看了30年的貝類，這個真的認不出來。靠近一點，側面45度角拍，我保證給你一個準確的答案！"
            })

        print(f"  啟動 Gemma 12B 深度驗證...")
        gemma_result = gemma_verify(img_data, best_key, similarity)

        personality_msg = ""
        local_name = ""

        if gemma_result:
            final_name = gemma_result.get('name', best_key)
            confidence = gemma_result.get('confidence', '中')
            personality_msg = gemma_result.get('personality_msg', '')
            local_name = gemma_result.get('local_name', '')
            method = "AI雙重驗證" if gemma_result.get('confirmed') else "AI修正"
            print(f"  最終結果：{final_name}（{confidence}）")
        else:
            final_name = best_key
            confidence = "高" if similarity >= 0.90 else "中" if similarity >= 0.80 else "低"
            method = "圖片比對"

        species = SPECIES_DB.get(final_name, SPECIES_DB.get(best_key))
        if not species:
            species = {"latinName": "", "canHarvest": False, "danger": False,
                      "description": "中角灣潮間帶生物", "advice": "請對照圖鑑確認"}

        return jsonify({
            "name": final_name,
            "latinName": species["latinName"],
            "confidence": confidence,
            "description": species["description"],
            "canHarvest": species["canHarvest"],
            "danger": species["danger"],
            "advice": species["advice"],
            "similarity": f"{similarity:.0%}",
            "method": method,
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
            "personality_msg": ""
        })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "db_count": len(DB_FEATURES),
        "mode": "圖片比對 + Gemma12B雙重驗證 + 阿超人格"
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  萬壽社區 AI 生物辨識服務啟動中")
    print("  模式：圖片比對 + Gemma 12B + 阿超人格")
    print("  網址：http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)