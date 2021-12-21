#-*- coding: utf-8 -*-
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
from skimage import feature as skft
import numpy as np
import joblib
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix,classification_report,precision_score
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import pandas as pd
from glob import glob # FILE OPERATIONS
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import svm
import torch
# settings for LBP
radius = 3
n_points = 8 * radius

BASE_DATA_FOLDER = "./plant-seedlings-classification"
TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "train")

images_per_class = {}
for class_folder_name in os.listdir(TRAin_DATA_FOLDER):
    class_folder_path = os.path.join(TRAin_DATA_FOLDER, class_folder_name)
    class_label = class_folder_name
    images_per_class[class_label] = []
    for image_path in glob(os.path.join(class_folder_path, "*.png")):
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        images_per_class[class_label].append(image_bgr)

# print(type(images_per_class))
# 变成灰度图片
def rgb2gray(im):
    im = skimage.color.rgb2gray(im)
    im = skimage.transform.resize(im, (128, 64))
    return im

print(images_per_class.keys())
data_x=[]
data_l=[]
for index in images_per_class.keys():
    for image in tqdm(images_per_class[index]):
        image=rgb2gray(image)
        data_x.append(image)
        data_l.append(index)
print(np.shape(data_l),np.shape(data_x))

x_train,x_test,y_train,y_test=train_test_split(data_x,data_l,test_size=0.01,random_state=5)
print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))

# print(len(x_train))


### 获取图片LBP特征

def lbp_texture(train_data,test_data):

    # 生成训练，测试样本的直方图
    train_hist=np.zeros((len(x_train),12))
    test_hist=np.zeros((len(x_test),12))

    # 遍历训练样本的照片得到每张照片的LBP特征，归一化直方图并保存到train_hist中
    for i in np.arange(len(x_train)):

        # 得到LBP特征，p是依赖点个数，R为半径，method=['default','uniforn','nri_uniform','ror','var]
        # 可以修改P，R，method方法，得到不同准确率
        lbp=skft.local_binary_pattern(train_data[i],P=10,R=4,method='uniform')              #
        max_bins=int(lbp.max()+1)

        # 生成LBP特征归一化后直方图，作为SVC分类数据
        train_hist[i],_=np.histogram(lbp,density=True,bins=max_bins,range=(0,max_bins))


    for i in np.arange(len(x_test)):
        lbp = skft.local_binary_pattern(test_data[i], P=10, R=4, method='uniform')
        max_bins = int(lbp.max() + 1)
        test_hist[i], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))

    return train_hist,test_hist


# 得到训练数据，测试数据LBP特征直方图
x_train,x_test=lbp_texture(x_train,x_test)
print(x_train.shape)
print(x_test.shape)


# 设置SVC分类器，C为惩罚参数类型浮点数，kernel为核函数，用于非线性分类，gamma是rbf内核系数
# 可以设置不同的C，kernel函数，gamma值得到不同准确率，'ovo'用于多分类

# # clf=svm.SVC(kernel='rbf',decision_function_shape='ovr',C=1)
# clf.fit(x_train,y_train)
#
# # 保存训练模型clf.pkl
# joblib.dump(clf, '../experience4/clf.pkl')
# # 使用模型进行预测
# p_test=clf.predict(x_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

x_train, y_train= make_classification(n_samples=4700, n_features=12,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=2, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

print(clf.feature_importances_)



# print('decision_function:\n', clf.decision_function(x_train))
# print(precision_score(y_test, p_test, average='macro'))
# print(recall_score(y_test, p_test, average='macro'))


# # 训练准确率
# print(accuracy_score(y_test, p_test))
#
# path_to_test = './plant-seedlings-classification/test/*.png'
# pics = glob(path_to_test)
# testimages = []
# tests = []
# count=1
# num = len(pics)
# for i in pics:
#     print(str(count)+'/'+str(num),end='\r')
#     tests.append(i.split('/')[-1])
#     testimages.append(cv2.resize(cv2.imread(i),(128,64)))
#     count = count + 1
# test=[]
# for image in tqdm(testimages):
#     image = rgb2gray(image)
#     test.append(image)
# test_hist=np.zeros((len(test),12))
# for i in np.arange(len(x_train)):
#     # 得到LBP特征，p是依赖点个数，R为半径，method=['default','uniforn','nri_uniform','ror','var]
#     # 可以修改P，R，method方法，得到不同准确率
#     lbp = skft.local_binary_pattern(test[i], P=10, R=4, method='uniform')  #
#     max_bins = int(lbp.max() + 1)
#
#     # 生成LBP特征归一化后直方图，作为SVC分类数据
#     test_hist[i], _ = np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
# p_test=clf.predict(x_test)
# # 提交
# train_data = pd.read_csv(r'./sample_submission.csv')
# file = train_data['file']
# submission = pd.DataFrame()
# submission['file'] = file
# submission['species'] = p_test
# submission.to_csv(r'submission_lbp_svm.csv', index=False)