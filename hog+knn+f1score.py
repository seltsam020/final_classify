import numpy as np # MATRIX OPERATIONS
import pandas as pd # EFFICIENT DATA STRUCTURES
import matplotlib.pyplot as plt # GRAPHING AND VISUALIZATIONS
import math # MATHEMATICAL OPERATIONS
import cv2 # IMAGE PROCESSING - OPENCV
from glob import glob # FILE OPERATIONS
from skimage.feature import hog
import os
from sklearn.decomposition import PCA
import skimage.color
import skimage.feature
import skimage.io
import skimage.transform
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np

# GLOBAL VARIABLES
seed = 7
BASE_DATA_FOLDER = "./plant-seedlings-classification"
TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "plant-seedlings-classification/train")

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
#
# print(images_per_class.keys())
data_x=[]
data_l=[]
for index in images_per_class.keys():
    for image in tqdm(images_per_class[index]):
        image=rgb2gray(image)
        fd = hog(image,orientations=8, pixels_per_cell=(16, 8), cells_per_block=(4, 4))
        # print(np.shape(fd))
        data_x.append(fd)
        data_l.append(index)
print(np.shape(data_l),np.shape(data_x))

#执行命令之前，这个保存到文件要关闭
np.savetxt("train_hogfeature.csv",data_x,delimiter=",",header='feature',comments="")
np.savetxt("label.csv",data_l,delimiter=",",header='label',comments="",fmt='%s')

import sklearn.model_selection as sk_model_selection
all=np.loadtxt("train_hogfeature.csv",dtype=np.float,delimiter=',',skiprows=1)
lable=np.loadtxt("label.csv",dtype=np.str,delimiter=',',skiprows=1)

# print(np.shape(all),np.shape(lable))
# k_range = range(1,50)
# cv_scores = []		#用来放每个模型的结果值
# for n in k_range:
#     knn = KNeighborsClassifier(n)   #knn模型，这里一个超参数可以做预测，当多个超参数时需要使用另一种方法GridSearchCV
#     scores = sk_model_selection.cross_val_score(knn,\all, lable,cv=10,scoring='accuracy')  #cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值，具体使用参考下面。
#     cv_scores.append(scores.mean())
# plt.plot(k_range,cv_scores)
# plt.xlabel('K')
# plt.ylabel('Accuracy')		#通过图像选择最好的参数
# plt.show()

model = KNeighborsClassifier(n_neighbors=5,n_jobs=1) #KNN分类
accs=sk_model_selection.cross_val_score(model, all, y=lable, scoring=None,cv=10, n_jobs=1)
print('交叉验证结果:',accs)

# model = KNeighborsClassifier(n_neighbors=5,n_jobs=1)
# model.fit(all,lable)
# # from sklearn.metrics import f1_score



# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score
# data_train, data_test, label_train, label_test = train_test_split(all, lable, test_size=0.2,random_state=7)
# mod = KNeighborsClassifier(n_neighbors=3)
# mod.fit(data_train, label_train)
# predictions = mod.predict(data_test)
# # print(np.shape(predictions),np.shape(label_test))
# print(f1_score (label_test, predictions, labels=None, pos_label=1, average='macro', sample_weight=None))
# print(f1_score (label_test, predictions, labels=None, pos_label=1, average='weighted', sample_weight=None))
# #
# ########提取testhog 特征
# # path_to_test = './plant-seedlings-classification/test/*.png'
# # pics = glob(path_to_test)
# # testimages = []
# # tests = []
# # count=1
# # num = len(pics)
# # for i in pics:
# #     print(str(count)+'/'+str(num),end='\r')
# #     tests.append(i.split('/')[-1])
# #     testimages.append(cv2.resize(cv2.imread(i),(128,64)))
# #     count = count + 1
# # hog_test=[]
# # for image in tqdm(testimages):
# #     image = rgb2gray(image)
# #     fd = hog(image,orientations=8, pixels_per_cell=(16, 8), cells_per_block=(4, 4))
# #     hog_test.append(fd)
# # print(np.shape(hog_test))
# # np.savetxt("test_hogfeature.csv",hog_test,delimiter=",",header='feature',comments="")
#
# #读取test  hog特征
# test_hog=np.loadtxt("test_hogfeature.csv",dtype=np.float,delimiter=',',skiprows=1)
# print(np.shape(test_hog))
#
# predictions = mod.predict(test_hog)
# print(predictions)
#
# #提交
# train_data = pd.read_csv(r'./sample_submission.csv')
# file = train_data['file']
# submission = pd.DataFrame()
# submission['file'] = file
# submission['species'] = predictions
# submission.to_csv(r'submission_hog_knn5.csv', index=False)
#
