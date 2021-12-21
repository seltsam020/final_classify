#-*- coding: utf-8 -*-
import time
import numpy as np
from main import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def HOG_UP(allimage,testimage):#所有的HOG操作
    print("========================>训练集hog获取")
    data_x = []
    label_x= []
    for index in allimage.keys():
        for image in tqdm(allimage[index]):
            image = segment_plant(image)
            img=imaged(image)
            hog_feature =hog(img,orientations=8, pixels_per_cell=(16, 8), cells_per_block=(4, 4))
            data_x.append(hog_feature)
            label_x.append(index)
            # print(np.shape(hog_feature))


    print('共有数据{0}组，标签{1}组'.format(len(data_x), len(label_x)))
    #划分训练集
    data_train, data_test, label_train, label_test = train_test_split(data_x, label_x, test_size=0.3,random_state=1)
    print('划分data_train, data_test, label_train, label_test分别有：\n')
    print(np.shape(data_train),np.shape(data_test),np.shape(label_train),np.shape(label_test))
    print("========================>测试集hog获取")

    pre_x=[]
    for image in tqdm(testimage):
        image = segment_plant(image)
        img = imaged(image)
        hog_feature = hog(img, orientations=8, pixels_per_cell=(16, 8), cells_per_block=(4, 4))
        pre_x.append(hog_feature)
    print("========================>测试集hog获取完毕")
    return data_train, data_test, label_train, label_test ,pre_x

def PAC_hog_to_128(data):
    pca = PCA(n_components=128)  # 自动选择特征个数  'mle'
    print("**************************")
    print('进行降维')
    pca.fit(data)
    new_data= pca.transform(data)
    print("降维后shape:{0}".format(new_data.shape))
    print("降维完成")
    return new_data

if __name__ == '__main__':
    # with open('alldata.pkl', 'rb') as f:
    #     allimage=joblib.load(f)
    # with open('pre_image.pkl', 'rb') as d:
    #     testimage=joblib.load(d)
    # data_train, data_test, label_train, label_test ,pre_x=HOG_UP(allimage,testimage)
    # new_data_train=PAC_hog_to_128(data_train)
    # new_data_test=PAC_hog_to_128(data_test)
    # # 考虑到训练一次时间较长，这里把new_data_train, new_data_test, label_train, label_test ,pre_x存入文件
    # with open('new_data_train_hog.pkl', 'wb') as f:
    #     joblib.dump(data_train, f)
    # with open('new_data_test_hog.pkl', 'wb') as f:
    #     joblib.dump(data_test, f)
    # with open('label_train_hog.pkl', 'wb') as f:
    #     joblib.dump(label_train, f)
    # with open('label_test_hog.pkl', 'wb') as f:
    #     joblib.dump(label_test, f)
    # with open('pre_x_hog.pkl', 'wb') as f:
    #     joblib.dump(pre_x, f)
    ##########################################以上便得到了所有需要的的hog特征并存于对应文件
    with open('new_data_train_hog.pkl', 'rb') as f:
         new_data_train =joblib.load(f)
    with open('new_data_test_hog.pkl', 'rb') as f:
         new_data_test=joblib.load(f)
    with open('label_train_hog.pkl', 'rb') as f:
         label_train=joblib.load(f)
    with open('label_test_hog.pkl', 'rb') as f:
         label_test=joblib.load(f)
    with open('pre_x_hog.pkl', 'rb') as f:
         pre_x=joblib.load(f)
    print("cnahsfjas")
    # model = svm.SVC(kernel='rbf', probability=True, gamma='auto', C=1)#|0.18
    model1=OneVsOneClassifier(svm.SVC(kernel='rbf',probability=True,C=5))#|0.55  rbf0.665  c=0.5 0.57    c=5 0.69
    model2 = OneVsRestClassifier(svm.SVC(kernel='rbf',probability=True))#0.52  rbf 0.67
    # model3=  SGDClassifier(tol=1e-3)#0.52
    # model4=OneVsOneClassifier(LogisticRegression(solver="sag",C=5,max_iter=100)) #0.54  0.57
    model5 = XGBClassifier( objective='multi：softmax')#0.62
    # model=model1*0.5+model1*0.3+model5*0.2
    # model.fit(new_data_train, label_train)
    # print("训练集准确率=====>>>>>>", model.score(new_data_train, label_train))
    # print("验证集准确率=====>>>>>>", model.score(new_data_test, label_test))
    model1.fit(new_data_train, label_train)
    model2.fit(new_data_train, label_train)
    model5.fit(new_data_train, label_train)
    predictions1 = model1.predict(new_data_test)
    predictions2 = model2.predict(new_data_test)
    predictions5 = model5.predict(new_data_test)
    # print(f1_score(label_test, predictions1, labels=None, pos_label=1, average='macro', sample_weight=None))
    # print(f1_score(label_test, predictions2, labels=None, pos_label=1, average='macro', sample_weight=None))
    # print(f1_score(label_test, predictions5, labels=None, pos_label=1, average='macro', sample_weight=None))
    '''
    0.642087801910837
    0.5943484050974138
    0.5661203177960191
    '''

    prefinal=[]
    for i in range(len(new_data_test)):
        pre1 = predictions1[i]
        pre2 = predictions2[i]
        pre5 = predictions5[i]
        if(pre2==pre5):
            prefinal.append(pre2)
        else:
            prefinal.append(pre1)
    print(f1_score(label_test, prefinal, labels=None, pos_label=1, average='macro', sample_weight=None))#0.6354
    pass