import pandas as pd
import numpy as np
import streamlit as st
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split




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
st.write("輸入性別，男填1、女填2")
n_GENDER = st.text_input("性別",1)

st.write("輸入年齡")
n_AGE = st.text_input("年齡",20)

st.write("輸入抽菸習慣，無填1、有填2")
n_SMOKING = st.text_input("抽菸習慣",1)

st.write("輸入有無黃手指，無填1、有填2")
n_YELLOW_FINGERS = st.text_input("黃手指",1)

st.write("輸入有無焦慮，無填1、有填2")
n_ANXIETY = st.text_input("焦慮",1)

st.write("輸入有無壓力，無填1、有填2")
n_PEER_PRESSURE = st.text_input("壓力",1)

st.write("輸入有無慢性病，無填1、有填2")
n_CHRONIC_DISEASE = st.text_input("慢性病",1)

st.write("輸入有無疲勞，無填1、有填2")
n_FATIGUE = st.text_input("疲勞",1)

st.write("輸入有無ALLERGY，無填1、有填2")
n_ALLERGY = st.text_input("ALLERGY",1)

st.write("輸入有無WHEEZING，無填1、有填2")
n_WHEEZING = st.text_input("WHEEZING",1)

st.write("輸入有無ALCOHOL_CONSUMING，無填1、有填2")
n_ALCOHOL_CONSUMING = st.text_input("ALCOHOL_CONSUMING",1)

st.write("輸入有無COUGHING，無填1、有填2")
n_COUGHING = st.text_input("COUGHING",1)

st.write("輸入有無SHORTNESS_OF_BREATH，無填1、有填2")
n_SHORTNESS_OF_BREATH = st.text_input("SHORTNESS_OF_BREATH",1)

st.write("輸入有無SWALLOWING_DIFFICULTY，無填1、有填2")
n_SWALLOWING_DIFFICULTY = st.text_input("SWALLOWING_DIFFICULTY",1)

st.write("輸入有無CHEST_PAIN，無填1、有填2")
n_CHEST_PAIN = st.text_input("CHEST_PAIN",1)

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