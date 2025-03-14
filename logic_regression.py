import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os

# 读取训练、验证、测试集
data_path = './codabench_dataset'
train = pd.read_csv(os.path.join(data_path, 'train.csv'))
val = pd.read_csv(os.path.join(data_path, 'val.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))

# 文本特征提取（使用 Comment 字段）
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(train['Comment'])
X_val = vectorizer.transform(val['Comment'])

# 标签编码
le = LabelEncoder()
y_train = le.fit_transform(train['Ground_Truth_Label'])
y_val = le.transform(val['Ground_Truth_Label'])

# 训练 Logistic Regression 模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 验证集预测
val_preds = model.predict(X_val)

# 评估
acc = accuracy_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds, average='weighted')

print(f"Logistic Regression Accuracy: {acc:.3f}")
print(f"Logistic Regression F1 Score: {f1:.3f}")

# 如需用于测试集预测：
X_test = vectorizer.transform(test['Comment'])
test_preds = model.predict(X_test)
test_labels = le.inverse_transform(test_preds)

# 生成 submission 文件
submission = pd.DataFrame({
    'ID': test['ID'],
    'Predicted_Label': test_labels
})
submission.to_csv('./outputs/logreg_submission.csv', index=False)
print("✅ Submission file saved to ./outputs/logreg_submission.csv")
