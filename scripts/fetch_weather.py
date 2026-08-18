Python
import os
import requests

def fetch_weather(location="南投縣"):
    api_key = os.environ.get("CWA_API_KEY")
    if not api_key:
        raise ValueError("未設定 CWA_API_KEY 環境變數")
        
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-D0047-005"
    params = {
        "Authorization": api_key,
        "locationName": location,
        "format": "JSON"
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    
    # 預設範例結構，後續可依需求擷取具體降雨機率與溫度
    return {"rain_prob": 20.0, "temp": 26.5}
