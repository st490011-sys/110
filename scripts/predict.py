Python
import pandas as pd
from prophet import Prophet

def build_and_predict(history_df, weather_info=None, periods=12):
    # 若無歷史數據，預設建立基礎序列
    if history_df.empty:
        history_df = pd.DataFrame({
            "ds": pd.date_range(end=pd.Timestamp.now(), periods=24, freq="h"),
            "y": [10] * 24,
            "rain_prob": [20.0] * 24
        })
    else:
        if weather_info and "rain_prob" in weather_info:
            history_df["rain_prob"] = weather_info["rain_prob"]

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    
    if "rain_prob" in history_df.columns:
        m.add_regressor("rain_prob")

    m.fit(history_df)

    future = m.make_future_dataframe(periods=periods, freq="5min")
    if "rain_prob" in history_df.columns:
        future["rain_prob"] = weather_info.get("rain_prob", 0.0) if weather_info else 0.0

    forecast = m.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

