#-*- coding: utf-8 -*-
import os
import time
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from skimage.feature import hog
import pandas as pd
import cv2
import numpy as np
import skimage.transform
import skimage.color
from glob import glob
from tqdm import tqdm

import joblib
'''
main.py
一些常用函数  如读取，存储图片 读取，存储模型
图形改变size等
其余文件基本需from main import *

'''

def load_train_data():
    BASE_DATA_FOLDER = ".\\plant-seedlings-classification"
    TRAin_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "train")
    images_per_class = {}
    for class_folder_name in os.listdir(TRAin_DATA_FOLDER):
        class_folder_path = os.path.join(TRAin_DATA_FOLDER, class_folder_name)
        class_label = class_folder_name
        images_per_class[class_label] = []
        for image_path in tqdm(glob(os.path.join(class_folder_path, "*.png"))) :
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            images_per_class[class_label].append(image_bgr)
        print(class_folder_name, '读取完成！', class_label, '->', len(images_per_class[class_folder_name]))
    # print(images_per_class.popitem())

    return images_per_class

def load_test_data():#读取测试集
    TEST_FOLDER=".\\plant-seedlings-classification\\test"
    test_data = []
    for image_path in tqdm(glob(os.path.join(TEST_FOLDER, '*.png'))):
        image = cv2.imread(image_path)
        test_data.append(image)
    print('测试集：{}   读取完成！'.format(len(test_data)))
    return test_data

def imaged(im):
    im = skimage.color.rgb2gray(im)
    im = skimage.transform.resize(im, (128, 64))
    return im


def create_mask_for_plant(image):
    # bgr转化为hsv
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    # lower_hsv = np.array([25, 40, 40])
    # upper_hsv = np.array([80, 255, 255])

    # 二值化
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 形态学开操作 先腐蚀后膨胀
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def segment_plant(image):#背景去色
    mask = create_mask_for_plant(image)
    # 利用掩膜进行图像混合
    # 求交集 、 掩膜提取图像
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

if __name__ == '__main__':
    '''
    测试用，存储照片训练集测试集信息
    '''
    # alldata=load_train_data()
    # pre_image=load_test_data()
    # with open('alldata.pkl', 'wb') as f:
    #     joblib.dump(alldata, f)
    # with open('pre_image.pkl', 'wb') as d:
    #     joblib.dump(pre_image, d)

    pass