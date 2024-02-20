from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import jieba
from scipy.sparse import csr_matrix

class OneHotVectorizer(object):
    # 用于生成词典，并向量化文本
    def fit_transform(self, raw_documents):
        # 生词标记为"<UNK>"，测试集有可能会有生词
        words_set = {"<UNK>"}
        for raw_document in raw_documents:
            words_set.update(raw_document.split())

        # 构建列表
        words_list = list(words_set)

        # 构建字典
        self.words_dict = {word:idx for idx, word in enumerate(words_list)}
        
        # 字典长度
        self.dict_len = len(words_set)
        
        # 数据向量化 - 词袋模型
        X = []
        for sentence in raw_documents:
            x = [0] * self.dict_len
            for word in set(sentence.split()):
                idx = self.words_dict[word] if word in self.words_dict else self.words_dict["<UNK>"]
                x[idx] = 1
            X.append(x)
        # 将列表转换为 CSR 格式的稀疏矩阵
        sparse_matrix = csr_matrix(X)

        # 打印稀疏矩阵
        return sparse_matrix
    
    # 用于利用已有的词典，向量化文本
    def transform(self, raw_documents):
        X = []
        for sentence in raw_documents:
            x = [0] * self.dict_len
            for word in set(sentence.split()):
                idx = self.words_dict[word] if word in self.words_dict else self.words_dict["<UNK>"]
                x[idx] = 1
            X.append(x)
            
        # 将列表转换为 CSR 格式的稀疏矩阵
        sparse_matrix = csr_matrix(X)

        # 打印稀疏矩阵
        return sparse_matrix    




# 数据集对象
class DataSet(object):
    def __init__(self, X, y, vectorizer):
        self.X = X
        self.y = y
        self.vectorizer = vectorizer
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)

# 推理模型
class PredictModel(object):
    def __init__(self, model, classification_data):
        self.model = model
        self.model_name = type(model).__name__
        self.classification_data = classification_data
        
        # 预处理数据
        self._mean = self.classification_data.X_train.mean(axis=0)
        self._std = self.classification_data.X_train.std(axis=0) + 1e-9
        self.X_train_pre = (self.classification_data.X_train - self._mean) / self._std
        self.X_test_pre = (self.classification_data.X_test - self._mean) / self._std
        # self.X_train_pre = self.classification_data.X_train
        # self.X_test_pre = self.classification_data.X_test

    def fit(self):
        start = time.perf_counter()
        
        # 训练
        self.model.fit(X=self.X_train_pre, y=self.classification_data.y_train)
        
        end = time.perf_counter() 
        # 训练耗时（单位为秒）
        self.train_duration = round(end - start, 2)

    def predict(self):
        start = time.perf_counter()

        # 推理
        self.y_pred = self.model.predict(X=self.X_test_pre)
        
        end = time.perf_counter() 
        # 推理耗时（单位为秒）
        self.pred_duration = round(end - start, 2)

        # 评估
        self._evaluation()
        
    def _evaluation(self):
        # 模型评估
        self.acc = (self.y_pred == self.classification_data.y_test).mean()
            
    def get_eval(self):
        return self.acc 
    
    def save(self):
        joblib.dump(value=self.model, filename="./models/"+self.model_name+"-"+self.classification_data.vectorizer)
        filename = "./models/"+self.model_name+"-"+self.classification_data.vectorizer + "-mean.txt"
        np.savetxt(filename, self._mean)
        filename = "./models/"+self.model_name+"-"+self.classification_data.vectorizer + "-std.txt"
        np.savetxt(filename, self._std)
        
        
    
# 读数据
# input
# folder_path: 数据目录
# vectorizer: 向量化算法
def Read_comments_from_file(folder_path, stop_words_path):
    X = []
    y = []
    stopwords = []
    # 读取停用词文件
    with open(stop_words_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除每行末尾的换行符并添加到停用词列表中
            stopwords.append(line.strip())
            
    # 读取数据文件
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir == 'neg':
                label = 0
            elif dir == 'pos':
                label = 1
            else:
                continue
            for file in os.listdir(os.path.join(folder_path, dir)):
                if file.endswith('.txt'):
                    file_path = os.path.join(folder_path, dir, file)
                    try:
                        with open(file_path, 'r', encoding='gb2312', errors='ignore') as f:
                            content = f.read()
                            stripcontent = content.strip().replace(" ", "").replace('\n', '').replace('\t', '')
                        
                            if stripcontent == "": 
                                # print(file_path + " is empty")
                                continue
                            content = content.strip().replace('\n', '').replace('\t', '')
                            # 使用jieba进行分词
                            words = ' '.join([word for word in jieba.cut(content) if word not in stopwords])
                            X.append(words)
                            y.append(label)
                    except:
                        # print(file_path + " has exception")
                        continue
                # else:
                #     file_path = os.path.join(folder_path, dir, file)
                #     print(file_path + " is not txt")

    return X, y

# 测试用返回模型
def Get_test_model():
    models = []
    models.append(RandomForestClassifier(criterion='gini',n_estimators=100))
    return models
    

# 创建分类模型
def Make_model_classifier():
    models = []
    
    # 创建LinearClassifier模型
#    linear_classifier = LinearClassifier()
#    models.append(linear_classifier)

    # 创建KNeighborsClassifier模型
    models.append(KNeighborsClassifier(n_neighbors=5, weights='distance'))
    
    # 创建logisticRegressionClassifier
    models.append(LogisticRegression(C=1))
    
    # 创建DecisionTreeClassifier模型
    models.append(DecisionTreeClassifier(criterion='entropy',max_depth=100, random_state=1))

    # 创建SVC模型
    models.append(SVC(kernel='linear',C=0.1))
    
    # 创建随机森林模型
    models.append(RandomForestClassifier(criterion='gini',n_estimators=100))

    # 创建AdaBoostClassifier模型
    models.append(AdaBoostClassifier(n_estimators=200, learning_rate=1.0))

    # 创建GradientBoostingClassifier模型
    models.append(GradientBoostingClassifier(n_estimators=200, learning_rate=1.0))

    # 创建XGBClassifier模型
    models.append(XGBClassifier(n_estimators=200, learning_rate=1.0))
    
    # 创建LGBMClassifier模型
    models.append(LGBMClassifier(n_estimators=200, learning_rate=1.0))
    
    return models
def Get_model_short_name():
    short_name = []
    short_name.append("KN")
    short_name.append("LG")
    short_name.append("DT")
    short_name.append("SVC")
    short_name.append("RF")
    short_name.append("AdaB")
    short_name.append("GB")
    short_name.append("XGB")
    short_name.append("LGBM")
    return short_name

def Result_analysis(predict_models):
    ret = {}
    x = []
    y = {}
    y["train_duration"]=[]
    y["pred_duration"]=[]
    y["acc"]=[]
    for model in predict_models:
        x.append(model.model_name)
        y["train_duration"].append(model.train_duration)
        y["pred_duration"].append(model.pred_duration)
        y["acc"].append(model.acc)
    ret["x"] = x
    ret["y"] = y
    return ret
        
def Plot_analysis(sample_analysis):
    plt.figure() 
    x_line_text = Get_model_short_name()
    
    x = sample_analysis["x"]
    y = sample_analysis["y"]
    
    x_index = np.arange(9)
    
    plt.title("train_duration")
    plt.bar(x_index-0.2, y["train_duration"][:9], width=0.2, color='y', label="Counter")
    plt.bar(x_index, y["train_duration"][9:18], width=0.2, color='c', label="TFIDF")
    plt.bar(x_index+0.2, y["train_duration"][18:], width=0.2, color='m', label="OntHot")
    plt.xticks(x_index, x_line_text)
    plt.grid(True)
    plt.autoscale(enable=True, axis='y')
    plt.legend()
    plt.show()

    plt.title("pred_duration")
    plt.bar(x_index-0.2, y["pred_duration"][:9], width=0.2, color='y', label="Counter")
    plt.bar(x_index, y["pred_duration"][9:18], width=0.2, color='c', label="TFIDF")
    plt.bar(x_index+0.2, y["pred_duration"][18:], width=0.2, color='m', label="OntHot")
    plt.xticks(x_index, x_line_text)
    plt.grid(True)
    plt.autoscale(enable=True, axis='y')
    plt.legend()
    plt.show()

    plt.title("acc")
    plt.bar(x_index-0.2, y["acc"][:9], width=0.2, color='y', label="Counter")
    plt.bar(x_index, y["acc"][9:18], width=0.2, color='c', label="TFIDF")
    plt.bar(x_index+0.2, y["acc"][18:], width=0.2, color='m', label="OntHot")
    plt.xticks(x_index, x_line_text)
    plt.grid(True)
    plt.autoscale(enable=True, axis='y')
    plt.legend()
    plt.show()
    
    
