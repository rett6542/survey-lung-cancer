import pandas as pd
import numpy as np
import streamlit as st
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




st.title("預測有無肺癌的可能")

#載入資料
lung_cancer = pd.read_csv('survey lung cancer.csv', encoding='big5')

#將物件屬性轉為數值
label_encoder = preprocessing.LabelEncoder()
lung_cancer["GENDER"] = label_encoder.fit_transform(lung_cancer["GENDER"])
lung_cancer["LUNG_CANCER"] = label_encoder.fit_transform(lung_cancer["LUNG_CANCER"])

#建立數據集
X = lung_cancer.copy()
X.drop(columns=['LUNG_CANCER'], inplace=True)
y = lung_cancer["LUNG_CANCER"]

#切割數據集
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=1)


#建立Logistic模型及訓練
logistic = linear_model.LogisticRegression()
logistic.fit(XTrain, yTrain)
yPred_log = logistic.predict(XTest)

#預判分數(90%)
report = classification_report(yTest, yPred_log)
confusion = confusion_matrix(yTest, yPred_log)

# print(report)
# print(confusion)

# 預測有無肺癌
#建立一個輸入的表單
with st.form(key='my_form'):

    n_GENDER = st.text_input("輸入性別，男填1、女填2",1)
    n_AGE = st.text_input("輸入年齡",20)
    n_SMOKING = st.text_input("輸入抽菸習慣，無填1、有填2",1)
    n_YELLOW_FINGERS = st.text_input("輸入有無黃手指，無填1、有填2",1)
    n_ANXIETY = st.text_input("輸入有無焦慮，無填1、有填2",1)
    n_PEER_PRESSURE = st.text_input("輸入有無壓力，無填1、有填2",1)
    n_CHRONIC_DISEASE = st.text_input("輸入有無慢性病，無填1、有填2",1)
    n_FATIGUE = st.text_input("輸入有無疲勞，無填1、有填2",1)
    n_ALLERGY = st.text_input("輸入有無ALLERGY，無填1、有填2",1)
    n_WHEEZING = st.text_input("輸入有無WHEEZING，無填1、有填2",1)
    n_ALCOHOL_CONSUMING = st.text_input("輸入有無ALCOHOL_CONSUMING，無填1、有填2",1)
    n_COUGHING = st.text_input("輸入有無COUGHING，無填1、有填2",1)
    n_SHORTNESS_OF_BREATH = st.text_input("輸入有無SHORTNESS_OF_BREATH，無填1、有填2",1)
    n_SWALLOWING_DIFFICULTY = st.text_input("輸入有無SWALLOWING_DIFFICULTY，無填1、有填2",1)
    n_CHEST_PAIN = st.text_input("輸入有無CHEST_PAIN，無填1、有填2",1)
    #建立一個按鍵做總匯入
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    new_lung_cancer = pd.DataFrame({
        "GENDER": [n_GENDER], "AGE": [n_AGE], "SMOKING": [n_SMOKING], "YELLOW_FINGERS": [n_YELLOW_FINGERS],
        "ANXIETY": [n_ANXIETY], "PEER_PRESSURE": [n_PEER_PRESSURE], "CHRONIC DISEASE": [n_CHRONIC_DISEASE],
        "FATIGUE ": [n_FATIGUE], "ALLERGY ": [n_ALLERGY], "WHEEZING": [n_WHEEZING], "ALCOHOL CONSUMING": [n_ALCOHOL_CONSUMING],
        "COUGHING": [n_COUGHING], "SHORTNESS OF BREATH": [n_SHORTNESS_OF_BREATH], "SWALLOWING DIFFICULTY": [n_SWALLOWING_DIFFICULTY],
        "CHEST PAIN": [n_CHEST_PAIN]})
    predicted_lung_cancer = logistic.predict(new_lung_cancer)
    st.write("預測結果，0為無肺癌可能、1為有肺癌可能")
    st.write(predicted_lung_cancer)

# plt.figure(figsize=(8,8))
# sns.heatmap(confusion,annot=True)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()
# print(report)

fig, ax = plt.subplots()
from sklearn import metrics
auc = metrics.roc_auc_score(yTest, yPred_log)

false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(yTest, yPred_log)

fig.set_size_inches(6, 6) 
ax.axis('equal')  # 將 x 和 y 軸的比例設置為相等
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title("AUC & ROC Curve")  # 設置標題
ax.plot(false_positive_rate, true_positive_rate,'--' )
ax.fill_between(false_positive_rate, true_positive_rate, facecolor='lightblue', alpha=0.7)
ax.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
ax.set_xlabel("False Positive Rate")  # 設置 x 軸標籤
ax.set_ylabel("True Positive Rate")  # 設置 y 軸標籤


st.pyplot(fig)