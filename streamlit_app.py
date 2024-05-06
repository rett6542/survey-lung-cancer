import pandas as pd
import numpy as np
import streamlit as st
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


st.title("預測有無肺癌的可能")
st.write("(採用Logistic模型，準確率90%)")

#載入資料
lung_cancer_old = pd.read_csv('survey lung cancer.csv', encoding='big5')
lung_cancer = lung_cancer_old.copy()

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
logistic = linear_model.LogisticRegression(max_iter=500)
logistic.fit(XTrain, yTrain)
yPred_log = logistic.predict(XTest)



# 預測有無肺癌
#建立一個輸入的表單
with st.form(key='my_form'):

    n_GENDER = st.selectbox("輸入性別",["男","女"])
    n_AGE = st.slider("輸入年齡", 0, 150)
    n_SMOKING = st.selectbox("輸入抽菸習慣",["無","有"])
    n_YELLOW_FINGERS = st.selectbox("有無黃手指",["無","有"])
    n_ANXIETY = st.selectbox("有無焦慮感",["無","有"])
    n_PEER_PRESSURE = st.selectbox("有無精神壓力",["無","有"])
    n_CHRONIC_DISEASE = st.selectbox("有無慢性病",["無","有"])
    n_FATIGUE = st.selectbox("有無疲勞感",["無","有"])
    n_ALLERGY = st.selectbox("有無過敏",["無","有"])
    n_WHEEZING = st.selectbox("有無氣喘",["無","有"])
    n_ALCOHOL_CONSUMING = st.selectbox("有無飲酒習慣",["無","有"])
    n_COUGHING = st.selectbox("有無咳嗽",["無","有"])
    n_SHORTNESS_OF_BREATH = st.selectbox("有無呼吸短促",["無","有"])
    n_SWALLOWING_DIFFICULTY = st.selectbox("有無吞嚥困難",["無","有"])
    n_CHEST_PAIN = st.selectbox("有無胸痛",["無","有"])
    #建立一個按鍵做總匯入
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    new_lung_cancer = pd.DataFrame({
        "GENDER": [n_GENDER], "AGE": [n_AGE], "SMOKING": [n_SMOKING], "YELLOW_FINGERS": [n_YELLOW_FINGERS],
        "ANXIETY": [n_ANXIETY], "PEER_PRESSURE": [n_PEER_PRESSURE], "CHRONIC DISEASE": [n_CHRONIC_DISEASE],
        "FATIGUE ": [n_FATIGUE], "ALLERGY ": [n_ALLERGY], "WHEEZING": [n_WHEEZING], "ALCOHOL CONSUMING": [n_ALCOHOL_CONSUMING],
        "COUGHING": [n_COUGHING], "SHORTNESS OF BREATH": [n_SHORTNESS_OF_BREATH], "SWALLOWING DIFFICULTY": [n_SWALLOWING_DIFFICULTY],
        "CHEST PAIN": [n_CHEST_PAIN]})
    new_lung_cancer["GENDER"] = np.where(new_lung_cancer["GENDER"] == "男" ,1 , 2)
    new_lung_cancer["SMOKING"] = np.where(new_lung_cancer["SMOKING"] == "無" ,1 , 2)
    new_lung_cancer["YELLOW_FINGERS"] = np.where(new_lung_cancer["YELLOW_FINGERS"] == "無" ,1 , 2)
    new_lung_cancer["ANXIETY"] = np.where(new_lung_cancer["ANXIETY"] == "無" ,1 , 2)
    new_lung_cancer["PEER_PRESSURE"] = np.where(new_lung_cancer["PEER_PRESSURE"] == "無" ,1 , 2)
    new_lung_cancer["CHRONIC DISEASE"] = np.where(new_lung_cancer["CHRONIC DISEASE"] == "無" ,1 , 2)
    new_lung_cancer["FATIGUE "] = np.where(new_lung_cancer["FATIGUE "] == "無" ,1 , 2)
    new_lung_cancer["ALLERGY "] = np.where(new_lung_cancer["ALLERGY "] == "無" ,1 , 2)
    new_lung_cancer["WHEEZING"] = np.where(new_lung_cancer["WHEEZING"] == "無" ,1 , 2)
    new_lung_cancer["ALCOHOL CONSUMING"] = np.where(new_lung_cancer["ALCOHOL CONSUMING"] == "無" ,1 , 2)
    new_lung_cancer["COUGHING"] = np.where(new_lung_cancer["COUGHING"] == "無" ,1 , 2)
    new_lung_cancer["SHORTNESS OF BREATH"] = np.where(new_lung_cancer["SHORTNESS OF BREATH"] == "無" ,1 , 2)
    new_lung_cancer["SWALLOWING DIFFICULTY"] = np.where(new_lung_cancer["SWALLOWING DIFFICULTY"] == "無" ,1 , 2)
    new_lung_cancer["CHEST PAIN"] = np.where(new_lung_cancer["CHEST PAIN"] == "無" ,1 , 2)
    predicted_lung_cancer = logistic.predict(new_lung_cancer)
    st.write("預測結果:")
    predicted_lung_cancer = np.where(predicted_lung_cancer == 0 ,"無肺癌可能" , "有肺癌可能")
    st.write(predicted_lung_cancer)

if st.checkbox('顯示肺癌數據集資料'):
    lung_cancer_old

if st.checkbox('顯示模型的分類報告'):
    report = classification_report(yTest, yPred_log)
    #使用st.text_area控制文本的格式，字不會跑掉。
    st.text_area("分類報告", report)

if st.checkbox('顯示模型的混淆矩陣熱圖'):
    #計算混淆矩陣數值
    confusion = confusion_matrix(yTest, yPred_log)

    fig1, ax1 = plt.subplots()
    fig1.set_size_inches(8, 8)
    sns.heatmap(confusion, annot=True, ax=ax1)  # 將ax1作為參數傳遞以使用相同的axes物件
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    st.pyplot(fig1)

if st.checkbox('顯示模型的ROC曲線及AUC分數'):
    #計算AUC分數
    auc = metrics.roc_auc_score(yTest, yPred_log)
    #false_positive_rate（FP/N）、true_positive_rate（TP/P）、thresholds（閾值）在二元分類中，分類器根據閾值將樣本分為正類(P)和負類(N)
    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(yTest, yPred_log)

    fig2, ax2 = plt.subplots()
    fig2.set_size_inches(6, 6) 
    ax2.axis('equal')  # 將 x 和 y 軸的比例設置為相等
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title("AUC & ROC Curve")  # 設置標題
    ax2.plot(false_positive_rate, true_positive_rate,'--' )
    ax2.fill_between(false_positive_rate, true_positive_rate, facecolor='lightblue', alpha=0.7)
    #0.95, 0.為位置參數
    ax2.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    ax2.set_xlabel("False Positive Rate")  # 設置 x 軸標籤
    ax2.set_ylabel("True Positive Rate")  # 設置 y 軸標籤
    st.pyplot(fig2)

#增加超連結
st.markdown("資料來源:[kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)")

#增加側邊佈局
option = st.sidebar.selectbox('選取特徵',['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
       'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
       'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
       'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER'])
                              
st.sidebar.text('顯示在不同特徵下的人數長條圖')
fig3, ax3 = plt.subplots()
fig3.set_size_inches(6, 6) 
sns.countplot(data=lung_cancer_old,x=option,ax=ax3)
st.sidebar.pyplot(fig3)

st.sidebar.text('顯示在肺癌與否情況下，不同特徵的人數長條圖')
fig4, ax4 = plt.subplots()
fig4.set_size_inches(6, 6) 
sns.countplot(data=lung_cancer_old,x=option,ax=ax4,hue='LUNG_CANCER')
st.sidebar.pyplot(fig4)