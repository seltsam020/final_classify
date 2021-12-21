import numpy as np
import cv2 as cv
import glob
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

image_list = []
all_des = []
print("GRABBING IMAGES")
for filename in glob.glob(r'D:\\PyCharm 2021.2.1\\experience4\\all_image\\*.png'
): #读取所有图片
    im = cv.imread(filename)
    image_list.append(im)
print(np.shape(image_list))
print("GETTING DESCRIPTORS")
for img in image_list: # 计算特征
    grey= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kp, des = cv.SIFT.create().detectAndCompute(grey, None)
    all_des.append(des)
all_des=np.array(all_des, dtype=np.ndarray) #changes descriptor list to array
all_des=np.concatenate(all_des, axis=0)
print(np.shape(all_des))
print('CLASSIFY BEGIN ,IT WILL TAKE A FEW MINS...')
km = KMeans(n_clusters=12, random_state=0).fit(all_des)
print(km)

# 分类
# BG = "Black-grass"
# CL = "Charlock"
# CV = "Cleavers"
# CC = "Common Chickweed"
# CW = "Common wheat"
# FH = "Fat Hen"
# LS = "Loose Silky-bent"
# MA = "Maize"
# SM = "Scentless Mayweed"
# SP = "Shepherds Purse"
# SFC = "Small-flowered Cranesbill"
# SB = "Sugar beet"

def readTrainingImages():
    root_dir = 'D:/PyCharm 2021.2.1/experience4/plant-seedlings-classification/train/'
    folders = glob.glob(root_dir + '*')

    training_images = []
    training_labels = []

    for folder in folders:
        for image in glob.glob(folder+'/*.png'):
            training_images.append(cv.imread(image))
            if 'Black-grass' in folder:
                training_labels.append('Black-grass')
            elif 'Charlock' in folder:
                training_labels.append('Charlock')
            elif 'Cleavers' in folder:
                training_labels.append('Cleavers')
            elif 'Common Chickweed' in folder:
                training_labels.append('Common Chickweed')
            elif 'Fat Hen' in folder:
                training_labels.append('Fat Hen')
            elif 'Loose Silky-bent' in folder:
                training_labels.append('Loose Silky-bent')
            elif 'Maize' in folder:
                training_labels.append('Maize')
            elif 'Scentless Mayweed' in folder:
                training_labels.append('Scentless Mayweed')
            elif 'Shepherds Purse' in folder:
                training_labels.append('Shepherds Purse')
            elif 'Small-flowered Cranesbill' in folder:
                training_labels.append('Small-flowered Cranesbill')
            elif 'Common wheat' in folder:
                training_labels.append('Common wheat')
            elif 'Sugar beet' in folder:
                training_labels.append('Sugar beet')
    print(np.shape(training_labels),np.shape(training_images))
    return training_images,training_labels

#测试集
def readTestingImages():
    root_dir = 'D:/PyCharm 2021.2.1/experience4/plant-seedlings-classification/test/'

    testing_images =[]
    testing_labels =[]
    for image in glob.glob(root_dir+'/*.png'):
            testing_images.append(cv.imread(image))
            testing_labels.append('Black-grass')
    return testing_images,testing_labels


'''
    Converts all training images to normalised histograms.
'''
def getTrainingHistograms(training_images, training_labels, km):
    X_train = []
    y_train = training_labels

    for image in training_images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #Convert to grayscale 
        kp, descript = cv.SIFT.create().detectAndCompute(gray, None) # Gets keypoints and descriptors for image (to be replaced with tylers code)
        histogram, bin_edges=np.histogram(km.predict(descript),bins=5) #Histogram as feature vector
        #print(histogram, bin_edges,sep='\n')
        X_train.append(histogram)

    return X_train, y_train


'''
    Converts all testing images to normalised histograms.
'''
def getTestingHistograms(testing_images, testing_labels):
    X_test = []
    y_test = testing_labels
    for image in testing_images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Convert to grayscale 
        kp, descript = cv.SIFT.create().detectAndCompute(gray, None) # Gets keypoints and descriptors for image (to be replaced with tylers code)
        test_histogram, bin_edges=np.histogram(km.predict(descript),bins=5)
        
        X_test.append(test_histogram)

    return X_test, y_test


'''
    Classifies every histogram from the testing image set with a class prediction.
    Prints the confusion matrix, error rate and classification errors for each class.
'''
def getClassifications(X_train, y_train, X_test, y_test):
    print('start classify')
    k_nn = KNeighborsClassifier(n_neighbors=3)
    k_nn.fit(X_train, y_train)
    predictions= k_nn.predict(X_test).tolist()
    print(type(predictions),type(y_test))
    #accuracy = k_nn.score(X_test, y_test)
    classes = ["Black-grass","Charlock","Cleavers","Common Chickweed", "Common wheat", "Fat Hen","Loose Silky-bent","Maize","Scentless Mayweed","Shepherds Purse","Small-flowered Cranesbill","Sugar beet"]
    # print("\nERROR RATE: " + str(1-accuracy))
    print(y_test[-3:], predictions[-3:])
    print(np.shape(predictions))
    print(confusion_matrix(y_test[-3:], predictions[-3:]))
    train_data = pd.read_csv(r'../experience4/sample_submission.csv')
    file = train_data['file']
    submission = pd.DataFrame()
    submission['file'] = file
    submission['species'] = predictions
    submission.to_csv(r'submission_diot.csv', index=False)
    print('finish3')



# Reads images into arrays.
training_images, training_labels=readTrainingImages()
testing_images, testing_labels=readTestingImages()
print(np.shape(training_labels),np.shape(training_images))
print('completed1')
# Creates histograms.
X_train, y_train = getTrainingHistograms(training_images, training_labels, km)
X_test, y_test = getTestingHistograms(testing_images, testing_labels)
print('completed2')
# Classifies the histograms.
print(y_test[-1])
pca = PCA(n_components=2)
print('start3')
print(np.shape(X_train),np.shape(y_train),np.shape(X_test),np.shape(y_test))
new_X_train=pca.fit_transform(X_train)
new_X_test=pca.fit_transform(X_test)
print(np.shape(new_X_train),np.shape(new_X_test))
# getClassifications(new_X_train,y_train,new_X_test,y_test)

from sklearn.model_selection import train_test_split
from sklearn import svm
clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
clf.fit(new_X_train, y_train)
y_hat = clf.predict(new_X_test)
print(np.shape(y_hat))
train_data = pd.read_csv(r'../experience4/sample_submission.csv')
file = train_data['file']
submission = pd.DataFrame()
submission['file'] = file
submission['species'] = y_hat
submission.to_csv(r'submission_diot_svm.csv', index=False)
print('finish3')
print('completed3')
