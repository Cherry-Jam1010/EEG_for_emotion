import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import joblib

# 1. 读取数据
print("正在读取数据集...")
df = pd.read_csv(r'E:\Grade_2_2\ai\hw1_plus\data\emotions.csv')

# 分离特征 (X) 和标签 (y)
# 最后一列是 'label'，前面 2548 列是特征
X = df.drop('label', axis=1).values
y = df['label'].values

# 2. 标签编码 (将字符串转换为数字)
# NEGATIVE -> 0, NEUTRAL -> 1, POSITIVE -> 2
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# 保存 LabelEncoder，后端解析时会用到
joblib.dump(label_encoder, 'label_encoder.pkl') 

# 3. 数据集划分 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. 数据标准化 (对于神经网络极其重要！)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 保存 StandardScaler，因为网站后端接收到新数据时，必须使用相同的规则进行缩放
joblib.dump(scaler, 'scaler.pkl')

# 5. 重塑数据维度以适应 1D CNN
# 1D CNN 的输入要求是 3D 张量: (样本数, 步长/特征数, 通道数)
# 我们这里有 2548 个特征，作为 1 个通道输入
features_count = X_train_scaled.shape[1]
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], features_count, 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], features_count, 1)

# 6. 构建 1D CNN 模型
print("正在构建 1D CNN 模型...")
model = Sequential()

# 第一层卷积
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(features_count, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2)) # 防止过拟合

# 第二层卷积
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

# 展平层
model.add(Flatten())

# 全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# 输出层 (3个神经元对应3种情绪，使用 softmax 输出概率)
model.add(Dense(3, activation='softmax'))

# 7. 编译模型
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 8. 训练模型
print("开始训练模型...")
history = model.fit(X_train_reshaped, y_train, 
                    epochs=20, # 你可以根据准确率调整训练轮数
                    batch_size=32, 
                    validation_data=(X_test_reshaped, y_test))

# 9. 评估并保存模型
test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
print(f"模型在测试集上的准确率: {test_acc:.4f}")

# 保存为 H5 格式，供 Flask 后端调用
model.save('emotion_cnn_model.h5')
print("模型和预处理工具已成功保存！")