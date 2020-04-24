# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:40:06 2020

@author: jiwandhono
"""

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
plt.style.use('ggplot')

path = 'Pengujian/Normalisasi New'

data = pd.read_excel(path+'\Fitur Tekstur dan Target(Norm).xlsx') #( path+'\Fitur Warna dan Tekstur.xlsx')

#cek index
data.iloc[:,3:6]


col1 = 3
col2 = 18
neighbor = 1
fitur = 'Warna Tekstur'

path_data = 'Pengujian/Pengujian Warna Tekstur/Data/'+fitur+'/'
path_write = 'Pengujian/Pengujian Tekstur/F5K1/'+fitur+'/'
excel = 'Hasil Akurasi Fitur ' + fitur + '.xlsx'
excel_cm = 'Confusion Matrix Fitur ' + fitur + '.xlsx'

def KNN(neighbor,col1,col2):
    hasil = []
    predict = []
    true_target = []
    for i in range(0,5):
        train = pd.read_excel(path_data + 'train ke- '+ str(i) + '.xlsx')
        test = pd.read_excel(path_data + 'test ke- '+ str(i) + '.xlsx')
        train = train.drop(['Unnamed: 0'], axis = 1)
        test = test.drop(['Unnamed: 0'], axis = 1)
        
        KNN = KNeighborsClassifier(n_neighbors = neighbor, metric = 'manhattan')    
        KNN.fit(train.iloc[:,col1:col2], train.iloc[:,-1])
        predicted = KNN.predict(test.iloc[:,col1:col2])
        
        predict.extend(predicted)
        true_target.extend(test.iloc[:,-1])
        #Hasil Akurasi Tiap Fold
        hasil.append(metrics.accuracy_score(test.iloc[:,-1], predicted[:]))
        
    mean_hasil = np.mean(hasil)
    final = metrics.accuracy_score(true_target,predict)
    
    zip_final = list(zip(predict, true_target))
    df_final = pd.DataFrame(zip_final, columns = ['Predict', 'True Target'])
    df_final['Correction'] = np.where((df_final['Predict'] != df_final['True Target']), False, True)
    df_final['accuracy'] = final
    df_final.to_excel(path_write + excel)
    
    return df_final,true_target, predict

df_final,true_target,predict = KNN(neighbor,col1,col2)

def conf_mat(true_target,predict):
    label = ['Ape', 'DadarGulung','Lumpur','PutuAyu','Soes','BikaAmbon',
             'Bolu', 'CumCum', 'Getas','Lapis','Pukis']
    labels = [0,1,2,3,4,5,6,7,8,9,10]
    index = ['Actual','Predicted']
    
    #Confusion Matrix
    cm = confusion_matrix(true_target,predict)
    df_conf = pd.DataFrame(cm,index=[i for i in label] ,columns = [i for i in label])
    
    #Precision, recall, f1score, support
    hasil_report = classification_report(true_target,predict,target_names=label, output_dict=True)
    # hasil_report2 = precision_recall_fscore_support(true_target, predict,labels=labels ,average=None)
    
    df_report = pd.DataFrame(hasil_report).transpose()
    
    #Concate dataframe df_conf and df_report
    df_final_report = pd.concat([df_conf,df_report], axis=1, sort=False)
    
    #Write
    df_final_report.to_excel(path_write + excel_cm)
    
    return df_final_report

df_final_report = conf_mat(true_target, predict)


    # plt.figure(figsize = (10,7),dpi=72)
    # plt.title('Confussion Matrix')
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    # sns.set(font_scale=1) #label size
    # conf = sns.heatmap(df_conf, annot=True, fmt='g', cmap =cmap, linewidths=0.3) #Belum jadi
    # plt.imshow(conf)
    # a = conf.get_figure()
    # a.savefig('C:/Users/jiwandhono/Belajar Python/Skripsi Citra/aa.png')
    
    # print(precision_recall_fscore_support(true_target, predict,labels=labels, average='macro'))
    # print('\n')
    # print(precision_recall_fscore_support(true_target, predict,labels=labels ,average='micro'))
    # print('\n')
    # print(precision_recall_fscore_support(true_target, predict,labels=labels ,average=None))
    # print('\n')
    # print(classification_report(true_target,predict,target_names=label))
# a = classification_report(true_target,predict,target_names=label, output_dict=True)
# df_a = pd.DataFrame(a).transpose()
    # print('\n')
# result = pd.concat([df_conf,df_a], axis=1, sort=False)
    

# hasil, mean_hasil = KNN(neighbor, col1,col2)
# print("Hasil Akurasi Tiap Fold :")
# print(hasil)
# print('\n')
# print("Hasil Mean Akurasi :")
# print(mean_hasil)


# plt.plot(range_k, hasil,color='red', marker='o', markerfacecolor='blue',
    #              markersize=10)
    # objects = ('iter 1', 'iter 2', 'iter 3', 'iter 4', 'iter 5')
    # plt.bar( objects, final_result)
    # plt.title('Pengujian Fitur Warna')
    # plt.xlabel('Iterasi')
    # plt.ylabel('Akurasi')
    # plt.show()

#Hasil yang benar adalah listnya berisi 45, karena 1 iterasi ada 9 bilai, 9 x 5 = 45
# range_k = range(1,11)

# def best_k(k,col1,col2):
#     final_result = []
#     hasil_k_iter = []
#     best_k = []
#     for i in range(0,5):
#         train = pd.read_excel(path_data + 'train ke- '+ str(i) + '.xlsx')
#         test = pd.read_excel(path_data + 'test ke- '+ str(i) + '.xlsx')
#         train = train.drop(['Unnamed: 0'], axis = 1)
#         test = test.drop(['Unnamed: 0'], axis = 1)
#         hasil = []
#         for k in range_k:
#             KNN = KNeighborsClassifier(n_neighbors = k, metric = 'manhattan')    
#             KNN.fit(train.iloc[:,col1:col2], train.iloc[:,-1])
#             predicted = KNN.predict(test.iloc[:,col1:col2])
            
#             #Hasil Akurasi Tiap iter
#             hasil_k_iter.append(metrics.accuracy_score(test.iloc[:,-1], predicted[:]))
#             hasil.append(metrics.accuracy_score(test.iloc[:,-1], predicted[:]))
        
#         #find best k index for 1 iteration
#         best = max(hasil)
#         best_k.append([i for i, j in enumerate(hasil) if j == best])
#         final_result.append(np.mean(hasil))
#     accuracy = np.mean(final_result)
#     # best_k = [int(i) for [i] in best_k]
    
#     df = pd.DataFrame({'a':range(50)})
#     df['K_All_iter'] = pd.Series(hasil_k_iter, index = df.index[:len(hasil_k_iter)])
#     df['Best_K_All_iter'] = pd.Series(best_k, index = df.index[:len(best_k)])
#     df['Accuracy_All_iter'] = pd.Series(final_result, index = df.index[:len(final_result)])
#     df['Rata2 Accuracy'] = pd.Series(accuracy)
    
#     df.to_excel('Hasil Pengujian Fitur Warna(AllFitur).xlsx')
    
#     # return accuracy
    
#     plt.plot(range_k, hasil,color='red', marker='o', markerfacecolor='blue',
#                  markersize=10)
#     objects = ('iter 1', 'iter 2', 'iter 3', 'iter 4', 'iter 5')
#     plt.bar( objects, final_result)
#     plt.title('Pengujian Fitur Warna')
#     plt.xlabel('Iterasi')
#     plt.ylabel('Akurasi')
#     plt.show()



# print('\n')
# print("Accuracy KNN:",metrics.accuracy_score(test.iloc[:,-1], predicted[:]))
# print('\n')
# print('CV scores : ' + '\n')
# cv_scores = cross_val_score(KNN, train.iloc[:,2:17], train.iloc[:,-1], cv=KF)
# print(cv_scores)
# print('cv_scores mean:{}'.format(np.mean(cv_scores)))

# KF = KFold(n_splits = 5, shuffle = True, random_state = 7 )

# KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'manhattan')
    
# KNN.fit(dataLatih.iloc[:,2:17], targetLatih)
    
# predicted = KNN.predict(dataUji.iloc[:,1:16])

# def normalization(data,col1,col2):
#     scaler = MinMaxScaler(feature_range = (0,1), copy = True)
#     data.iloc[:,col1:col2] = scaler.fit_transform(data.iloc[:,col1:col2])
        
#     return pd.DataFrame(data)
# # normalization(data,col1,col2)

# #move your body
# kfold = 5
# rs = 5
# def split_data(data,kfold,rs):
#     KF = KFold(n_splits = kfold, shuffle = True, random_state = rs )
#     i = 0
#     for train, test in KF.split(data):
#         data_train = pd.DataFrame(data.iloc[train])
#         data_test = pd.DataFrame(data.iloc[test])
        
#         data_train.to_excel(path_data +"train ke- " + str(i) + ".xlsx")
#         data_test.to_excel(path_data +"test ke- " + str(i) + ".xlsx")
#         i += 1
        
# split_data(data,kfold,rs)





