# 用来创建最终展示效果的streamlit web page
# 显示输入框，提交按钮，
# 后台调用模型，做预测
# 显示预测的情感分类结果

# 文档调用方法：streamlit run streamlit_web.py
import streamlit as st
import jieba
import joblib
from flask import request
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# web部分
st.title("请输入反馈意见")

st.divider()

txt = st.text_input(label="评论",
              placeholder="请输入评论 ..."
              )

btn = st.button("提交")

if btn:
    if txt:
        print("提交内容:")
        print(txt)   #后台显示
        st.write("您反馈的内容是：",txt) #前台显示
        st.success("提交成功！")
    else :
        st.toast("请输入内容！")
    

if txt:
	# 1、读取停用词文件
	stopwords = []
	stop_words_path = './stop_words.txt'
	with open(stop_words_path, 'r', encoding='utf-8') as file:
		for line in file:
			# 去除每行末尾的换行符并添加到停用词列表中
			stopwords.append(line.strip())

	np.set_printoptions(threshold=np.inf)
	# 2、读数据 # 使用jieba进行分词
	X = []
	words = ' '.join([word for word in jieba.cut(txt) if word not in stopwords])
	X.append(words)
	print(X)

	# 3、文本向量化
	# 返回的类型是scipy.sparse._csr.csr_matrix，是一个稀疏矩阵
	# 使用CountVectorizer进行文本向量化
	# vectorizer = CountVectorizer()
	
	vectorizer = joblib.load("./models/CountVectorizer")
	X = vectorizer.transform(X)
	X = X.toarray()
	# print(X)
	# print(vectorizer.vocabulary_)

	# 4、预处理：推理时，需要做跟训练一样的预处理

	_mean = X.mean(axis=0)
	_std = X.std(axis=0) + 1e-9
	X_pre = (X - _mean) / _std
	print(_mean)
	# print(_std)
	# X_pre = X

	# print(X_pre)

	# 5、加载模型（模型以单例的模式常驻内存）
	knn = joblib.load("./models/RandomForestClassifier-Counter")

	# 6、推理
	y_pred = knn.predict(X_pre)
	if y_pred == 1:
		print("预测结果：好评")
	else:
		print("预测结果：差评")