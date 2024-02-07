# test function
def Hello_world():
    print("Hello World")
    
    
# 读数据, todo
import os
import numpy as np


def Read_comments_from_file(file_path = "./user_comments/"):
    X = []
    y = [] 

    # 读取语料库
    file_list = os.listdir(file_path)
    for file in file_list:
        new_path = file_path + file + "/"
        file_names = os.listdir(new_path)
        for file_name in file_names:
            if file_name.endswith(".txt"):
                with open(file=new_path + file_name, mode='r', encoding='gbk', errors='ignore') as f:
                    content = f.read().replace('\n', '').replace('\t', '').replace(' ', '')
                    X.append(content)
                    label = 1 if file == "pos" else 0
                    y.append(label)
    return X, y