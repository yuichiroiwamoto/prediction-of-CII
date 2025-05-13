
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# モデルとスケーラーの読み込み
model = xgb.Booster()
model.load_model("ts_model.json")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# 特徴量の定義（順番は学習時と同じに！）
feature_names = [
    'glucose',
    'time_lag',
    'HCO3-',
    'lag1',
    '年齢',
    '体重',
    'insulin_flow_rate',
    'drip_flow_rate'
]

st.title("時系列予測モデル：ΔGlu予測")

# ファイルアップロード（任意）
uploaded_file = st.file_uploader("CSVファイルをアップロード（任意）", type="csv")

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    X = scaler_X.transform(input_df[feature_names])
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    pred_scaled = model.predict(dmatrix)
    pred_dGlu = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    input_df["予測ΔGlu (mg/dL)"] = pred_dGlu
    input_df["予測血糖値 (mg/dL)"] = input_df["glucose"] + pred_dGlu
    st.write("📊 予測結果：")
    st.dataframe(input_df)
else:
    st.subheader("手入力")

    input_values = {}
    input_values["glucose"] = st.number_input("血糖値 (mg/dL)", step=1.0)
    input_values["time_lag"] = st.number_input("再評価時点 (hour)", step=0.5)
    input_values["HCO3-"] = st.number_input("重炭酸イオン(ガス) (mEq/L)", step=0.1)
    input_values["lag1"] = st.number_input("前回のΔGlu (mg/dL、初回は0)", step=1.0)
    input_values["年齢"] = st.number_input("年齢 (歳)", step=1.0)
    input_values["体重"] = st.number_input("体重 (kg)", step=0.1)
    input_values["insulin_flow_rate"] = st.number_input("インスリン流量 (units/hour)", step=0.1)
    input_values["drip_flow_rate"] = st.number_input("点滴流量 (mL/hour)", step=1.0)

    if st.button("予測実行"):
        X_input = pd.DataFrame([input_values])[feature_names]
        X_scaled = scaler_X.transform(X_input)
        dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
        pred_scaled = model.predict(dmatrix)
        pred_dGlu = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        st.success(f"✅ 予測ΔGlu: {pred_dGlu:.2f} mg/dL")
        st.info(f"➡️ 予測血糖値: {input_values['glucose'] + pred_dGlu:.2f} mg/dL")
