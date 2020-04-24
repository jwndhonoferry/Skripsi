# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:21:49 2020

@author: jiwandhono
"""

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


path = 'Pengujian/Normalisasi New'

data = pd.read_excel(path+'/Fitur Warna Tekstur dan Target(Norm) EnergiSkewMean.xlsx') #( path+'\Fitur Warna dan Tekstur.xlsx')

# data = pd.read_excel('Feature Latih Warna Ape New.xlsx')


col1 = 3
col2 = 18
data.iloc[:,col1:col2]

def normalization(data,col1,col2):
    scaler = MinMaxScaler(feature_range = (0,1), copy = True)
    data.iloc[:,col1:col2] = scaler.fit_transform(data.iloc[:,col1:col2])
    data = pd.DataFrame(data)
    return data.to_excel("Fitur Warna Tekstur dan Target(Norm).xlsx")
normalization(data,col1,col2)

#move your body
kfold = 5
rs = 5
path_data = 'Pengujian/Pengujian Warna Tekstur/Campur/'#Data/AllFiturTekstur/'
def split_data(data,kfold,rs):
    KF = KFold(n_splits = kfold, shuffle = True, random_state = rs )
    i = 0
    for train, test in KF.split(data):
        data_train = pd.DataFrame(data.iloc[train])
        data_test = pd.DataFrame(data.iloc[test])
        
        data_train.to_excel(path_data +"train ke- " + str(i) + ".xlsx")
        data_test.to_excel(path_data +"test ke- " + str(i) + ".xlsx")
        i += 1
        

split_data(data,kfold,rs)

