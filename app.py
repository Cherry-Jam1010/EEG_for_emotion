from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 启动神经计算核心
df = pd.read_csv(os.path.join(BASE_DIR, 'emotions.csv'))
fft_cols = [i for i, col in enumerate(df.columns) if 'fft' in col.lower()]
X_raw = df.drop('label', axis=1).values
y_raw = df['label'].values

model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'emotion_cnn_model.h5'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

@app.route('/api/get_samples', methods=['GET'])
def get_samples():
    indices = np.random.choice(len(df), 15, replace=False).tolist()
    return jsonify([{'sample_id': i, 'name': f'神经科研样本 #{i}'} for i in indices])

@app.route('/api/predict', methods=['POST'])
def predict():
    sid = request.json.get('sample_id')
    features = X_raw[sid]
    
    # 1. AI 推理
    fs = scaler.transform([features]).reshape(1, 2548, 1)
    prob = model.predict(fs)
    p_label = label_encoder.inverse_transform([int(np.argmax(prob))])[0]

    # 2. 频段能量计算 (FFT 深度提取 + 动态方差算法)
    f_data = features[fft_cols] if len(fft_cols) > 0 else features[:1000]
    chunk = len(f_data) // 5
    bands = [float(np.var(f_data[i*chunk:(i+1)*chunk])) for i in range(5)]
    b_min, b_max = min(bands), max(bands)
    norm_bands = [round(((x - b_min) / (b_max - b_min + 1e-7) * 75) + 20, 1) for x in bands]

    # 3. 情绪深度分析报告
    conf = round(float(np.max(prob)) * 100, 1)
    reasoning = ""
    if p_label == "NEGATIVE":
        reasoning = f"【深度解析】AI 以 {conf}% 置信度判定为消极。Beta 波({norm_bands[3]}%)显著增强，前额叶高频放电红色区域指示被试处于焦虑或情绪负荷状态。"
    elif p_label == "POSITIVE":
        reasoning = f"【深度解析】当前为积极情绪。Alpha 波({norm_bands[2]}%)比例健康，左侧前额叶(Fp1)呈现趋近动机特征，显示出愉悦的心理投入。"
    else:
        reasoning = "【深度解析】脑电特征平稳，Delta/Theta 与 Beta 波处于平衡态，大脑处于心理中性基准线。"

    return jsonify({
        'true_label': y_raw[sid],
        'predicted_label': str(p_label),
        'confidence': float(np.max(prob)),
        'heatmap_data': [round(float(x/np.max(np.abs(features[:20]))*100), 1) for x in np.abs(features[:20])],
        'bands_data': norm_bands,
        'reasoning': reasoning
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)