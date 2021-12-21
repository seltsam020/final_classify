import cv2
import os
import pandas as pd # EFFICIENT DATA STRUCTURES
import numpy as np
from tqdm import tqdm
from libsvm.svmutil import *
from sklearn import svm
'''
首先考虑hog+SVM
从文件中读取hog信息
hog+knn+f1score.py中提取得到已存为文件

'''
all=np.loadtxt("train_hogfeature.csv",dtype=np.float,delimiter=',',skiprows=1)
lable=np.loadtxt("label.csv",dtype=np.str,delimiter=',',skiprows=1)


# '''
# 考虑提取sift特征
# '''
# from glob import glob
# import skimage.color
# import skimage.feature
# import skimage.io
# import skimage.transform
# BASE_DATA_FOLDER = "./plant-seedlings-classification"
# TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "train")
# images_per_class = {}
# for class_folder_name in os.listdir(TRAin_DATA_FOLDER):
#     class_folder_path = os.path.join(TRAin_DATA_FOLDER, class_folder_name)
#     class_label = class_folder_name
#     images_per_class[class_label] = []
#     for image_path in glob(os.path.join(class_folder_path, "*.png")):
#         image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
#         images_per_class[class_label].append(image_bgr)
#
# # print(type(images_per_class))
# # 变成灰度图片
# def rgb2gray(im):
#     im = skimage.color.rgb2gray(im)
#     im = skimage.transform.resize(im, (128, 64))
#     return im
#
# print(images_per_class.keys())
# data_x=[]
# for index in images_per_class.keys():
#     for image in tqdm(images_per_class[index]):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         sift = cv2.xfeatures2d.SIFT_create()
#         # 找出关键点
#         kp = sift.detect(gray, None)
#         kp, des = sift.compute(gray, kp)
#         # print(np.shape(fd))
#         data_x.append(des)
# data_x=np.array(data_x)
# print(np.shape(data_x))
# lable=np.loadtxt("label.csv",dtype=np.str,delimiter=',',skiprows=1)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all, lable, random_state=1, train_size=0.7)
print(np.shape(x_train),np.shape(y_train),np.shape(x_test),np.shape(y_test))

from sklearn.model_selection import GridSearchCV
parameters={'kernel':['linear','rbf','sigmoid','poly'],'C':np.linspace(0.1,20,50),'gamma':np.linspace(0.1,20,20)}
# clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
# clf = svm.SVC(kernel='linear', decision_function_shape='ovr')
clf = svm.SVC()
model = GridSearchCV(clf,parameters,cv=5,scoring='accuracy')
# clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
model.fit(x_train, y_train.ravel())

print(model.best_params_)
model.score(x_test,y_test)


from sklearn.metrics import f1_score


print(clf.score(x_train, y_train))
# y_hat = clf.predict(x_train)
print(clf.score(x_test, y_test))
y_hat = clf.predict(x_test)
# show_accuracy(y_hat, y_test, '测试集')
print(f1_score (y_test, y_hat, labels=None, pos_label=1, average='macro', sample_weight=None))
print(f1_score (y_test, y_hat, labels=None, pos_label=1, average='weighted', sample_weight=None))
#
# #读取test  hog特征
# test_hog=np.loadtxt("test_hogfeature.csv",dtype=np.float,delimiter=',',skiprows=1)
# print(np.shape(test_hog))
#
# predictions = clf.predict(test_hog)
# print(predictions)
#
# #提交
# train_data = pd.read_csv(r'./sample_submission.csv')
# file = train_data['file']
# submission = pd.DataFrame()
# submission['file'] = file
# submission['species'] = predictions
# submission.to_csv(r'submission_hog_svm_rbf.csv', index=False)























