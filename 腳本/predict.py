import os
import json
import datetime
import requests
import pandas as pd
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, db

# ==========================================
# 1. 安全讀取 GitHub Secrets 環境變數
# ==========================================
TDX_CLIENT_ID = os.environ.get("TDX_CLIENT_ID")
TDX_CLIENT_SECRET = os.environ.get("TDX_CLIENT_SECRET")
CWA_API_KEY = os.environ.get("CWA_API_KEY")
FIREBASE_SERVICE_ACCOUNT = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

# ==========================================
# 2. 初始化 Firebase Realtime Database
# ==========================================
def init_firebase():
    if not firebase_admin._apps:
        try:
            # 驗證並解析 Firebase 服務帳戶 JSON
            firebase_account = FIREBASE_SERVICE_ACCOUNT.strip() if FIREBASE_SERVICE_ACCOUNT else None
            if not firebase_account:
                raise ValueError("FIREBASE_SERVICE_ACCOUNT 環境變數未設定")
            
            cred_dict = json.loads(firebase_account)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://ssss-42c85-default-rtdb.asia-southeast1.firebasedatabase.app/'
            })
        except json.JSONDecodeError as e:
            print(f"錯誤: 無法解析 FIREBASE_SERVICE_ACCOUNT 為 JSON: {e}")
            print(f"原始值 (前 100 字元): {FIREBASE_SERVICE_ACCOUNT[:100] if FIREBASE_SERVICE_ACCOUNT else 'None'}")
            raise
        except Exception as e:
            print(f"錯誤: Firebase 初始化失敗: {e}")
            raise
# ==========================================
# 3. 抓取 TDX 停車場 / 交通資料 (OAuth 2.0)
# ==========================================
def get_tdx_token():
    auth_url = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": TDX_CLIENT_ID,
        "client_secret": TDX_CLIENT_SECRET
    }
    res = requests.post(auth_url, data=data)
    return res.json().get("access_token")

def fetch_tdx_data():
    try:
        token = get_tdx_token()
        headers = {"Authorization": f"Bearer {token}"}
        # 範例：查詢日月潭周邊停車場即時剩餘位數 (可依需求調整 API 網址)
        url = "https://tdx.transportdata.tw/api/basic/v2/Parking/National?%24top=10&%24format=JSON"
        res = requests.get(url, headers=headers)
        data = res.json()
        # 回傳第一筆剩餘位數作為擁擠指標，若失敗預設回傳 0
        return float(data[0].get("AvailableSpaces", 0)) if data else 0.0
    except Exception as e:
        print(f"TDX API 讀取失敗: {e}")
        return 0.0

# ==========================================
# 4. 抓取 CWA 氣象署降雨機率
# ==========================================
def fetch_cwa_rain_prob():
    try:
        # F-D0047-061 為南投縣鄉鎮預報資料集代碼
        url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-061?Authorization={CWA_API_KEY}&elementName=PoP12h"
        res = requests.get(url)
        data = res.json()
        # 解析降雨機率 (百分比數字)
        locations = data['records']['locations'][0]['location']
        # 預設抓取第一組鄉鎮的降雨機率
        rain_prob = locations[0]['weatherElement'][0]['time'][0]['elementValue'][0]['value']
        return float(rain_prob)
    except Exception as e:
        print(f"CWA API 讀取失敗: {e}")
        return 0.0

# ==========================================
# 5. Prophet AI 預測模型核心邏輯
# ==========================================
def run_prophet_prediction(rain_prob):
    # 從 Firebase 抓取 ESP32 傳上來的歷史人流數據
    ref_history = db.reference('/sensors/flow_history')
    history_data = ref_history.get()

    # 若無歷史紀錄，建立基本模擬數據避免程式中斷
    if not history_data:
        now = datetime.datetime.now()
        timestamps = [now - datetime.timedelta(minutes=5*i) for i in range(10, 0, -1)]
        counts = [5, 8, 12, 15, 20, 25, 18, 14, 10, 12]
    else:
        timestamps = [pd.to_datetime(item['timestamp']) for item in history_data.values()]
        counts = [item['count'] for item in history_data.values()]

    # 建立 Pandas DataFrame
    df = pd.DataFrame({
        'ds': timestamps,
        'y': counts,
        'rain_prob': rain_prob,
        'is_weekend': [1 if ts.weekday() >= 5 else 0 for ts in timestamps]
    })

    # 初始化與訓練 Prophet 模型
    model = Prophet()
    model.add_regressor('rain_prob')
    model.add_regressor('is_weekend')
    model.fit(df)

    # 預測未來 5 分鐘
    future = model.make_future_dataframe(periods=1, freq='5min')
    future['rain_prob'] = rain_prob
    future['is_weekend'] = 1 if datetime.datetime.now().weekday() >= 5 else 0

    forecast = model.predict(future)
    predicted_val = forecast.iloc[-1]['yhat']
    return max(0, int(predicted_val)) # 避免負數

# ==========================================
# 6. 主程式執行入口
# ==========================================
def main():
    print("=== 開始執行山城智慧節能 Prophet AI 預測流程 ===")
    init_firebase()

    # 1. 抓取外部 Open Data
    rain_prob = fetch_cwa_rain_prob()
    print(f"即時降雨機率 (CWA): {rain_prob}%")

    # 2. 執行 AI 預測
    predicted_count = run_prophet_prediction(rain_prob)
    print(f"Prophet 預測未來 5 分鐘人流量: {predicted_count} 人")

    # 3. 將人數分級為 低/中/高 (綠/黃/紅)
    if predicted_count < 10:
        level = "GREEN"  # 舒適 (綠)
    elif predicted_count < 25:
        level = "YELLOW" # 注意 (黃)
    else:
        level = "RED"    # 擁擠 (紅)

    print(f"預測分級結果: {level}")

    # 4. 寫回 Firebase 供 ESP32 讀取
    ref_pred = db.reference('/prediction')
    ref_pred.set({
        'level': level,
        'predicted_count': predicted_count,
        'updated_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    print("成功更新 Firebase 預測結果！")

if __name__ == "__main__":
    main()
