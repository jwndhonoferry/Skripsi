# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:00:59 2019

@author: jiwandhono
"""
from PIL import Image as PImage
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import cv2
import datetime
from statistics import mode

img = cv2.imread('Preprocessing\PutuAyu_crop_.jpg')

path = 'Pengujian\Analisis Visualisasi Gambar\Tekstur Mirip/'
Ape = cv2.imread(path+'Ape_crop_1.jpg')
DadarGulung = cv2.imread(path+'DadarGulung_crop_1.jpg')
Lumpur = cv2.imread(path+'Lumpur_crop_1.jpg')
PutuAyu = cv2.imread(path+'PutuAyu_crop_1.jpg')
Soes = cv2.imread(path+'Soes_crop_1.jpg')
BikaAmbon = cv2.imread(path+'BikaAmbon_crop_1.jpg')
Bolu = cv2.imread(path+'Bolu_crop_1.jpg')
CumCum = cv2.imread(path+'CumCum_crop_1.jpg')
Getas = cv2.imread(path+'Getas_crop_1.jpg')
Lapis = cv2.imread(path+'Lapis_crop_1.jpg')
Pukis = cv2.imread(path+'Pukis_crop_1.jpg')

imgApe, listApe = FN_LBP(Ape)
imgDadar, listDadar = FN_LBP(DadarGulung)
imgLumpur, listLumpur = FN_LBP(Lumpur)
imgPutuAyu, listPutuAyu = FN_LBP(PutuAyu)
imgSoes, listSoes= FN_LBP(Soes)
imgBikaAmbon, listBikaAmbon= FN_LBP(BikaAmbon)
imgBolu, listBolu= FN_LBP(Bolu)
imgCumCum, listCumCum= FN_LBP(CumCum)
imgGetas, listGetas= FN_LBP(Getas)
imgLapis, listLapis= FN_LBP(Lapis)
imgPukis, listPukis= FN_LBP(Pukis)

show(imgApe, 'Ape')
show(imgDadar, 'DadarGulung')
show(imgLumpur, 'Lumpur')
show(imgPutuAyu, 'PutuAyu')
show(imgSoes, 'Soes')
show(imgBikaAmbon, 'BikaAmbon')
show(imgBolu, 'Bolu')
show(imgCumCum, 'CumCum')
show(imgGetas, 'Getas')
show(imgLapis, 'Lapis')
show(imgPukis, 'Pukis')


# img = cv2.imread('Bolu1.jpg')
# imgG = cv2.cvtColor(imgG, cv2.COLOR_BGR2GRAY)
# pixelArray = np.random.randint(150, 230, size=9).reshape(3,3)

def biner(image,t_tengah, x,y):
    nilai_biner = []
    if image[x][y] >= t_tengah:
        nilai_biner.append(1)
    else:
        nilai_biner.append(0)
    return nilai_biner

def at_most(List_nilai):
    return(mode(List_nilai))

def circular_fnlbp(nilai_bin):
    pangkat_ketetanggan = [0,1,2,3,4,5,6,7]
    list_jumlah = []
    for i in range(0,8):
        jumlah = 0
        nilai_bin.insert(8, nilai_bin.pop(0))
        print(nilai_bin)
        for j in range(0, len(nilai_bin)):
            jumlah += nilai_bin[j] * math.pow(2,pangkat_ketetanggan[j])
            print(jumlah)
        list_jumlah.insert(0,jumlah)
    print(list_jumlah)
    return min(list_jumlah)

def FN_LBP(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_FNLBP = np.zeros_like(img)
    pixelArray = np.array(img)
    img_FNLBP1 = []
    for x in range(1, len(pixelArray)-1):
        for y in range(1, len(pixelArray[x])-1):
            nilai_bin = []
            barisan_bin = []
            #Operator
            
            #Center
            center_atas_kiri = pixelArray[x-1,y-1]
            center_tengah_atas = pixelArray[x-1,y]
            center_atas_kanan = pixelArray[x-1, y+1]
            center_kiri_tengah = pixelArray[x,y-1]
            # center_tengah = pixelArray[x,y]
            center_kanan_tengah = pixelArray[x, y+1]
            center_kiri_bawah = pixelArray[x+1, y-1]
            center_tengah_bawah = pixelArray[x+1, y]
            center_kanan_bawah = pixelArray[x+1,y+1]
            
            #Atas Kiri
            tetangga_kanan =  biner(pixelArray,center_atas_kiri, x-1,y)
            tetangga_diagonal_kanan = biner(pixelArray, center_atas_kiri, x,y)            
            tetangga_bawah = biner(pixelArray, center_atas_kiri, x, y-1)
            # barisan_bin.append(biner(pixelArray, center_atas_kiri, x-1,y))
            # barisan_bin.append(biner(pixelArray, center_atas_kiri, x,y))
            # barisan_bin.append(biner(pixelArray, center_atas_kiri, x,y-1))
            
            barisan_bin.append(tetangga_kanan)
            barisan_bin.append(tetangga_diagonal_kanan)
            barisan_bin.append(tetangga_bawah)
            barisan_bin = [int(i) for [i] in barisan_bin]
            nilai_bin.append(at_most(barisan_bin))
            
            #reset
            barisan_bin.clear()
            
            #Tengah Atas
            barisan_bin.append(biner(pixelArray, center_tengah_atas, x-1,y-1))
            barisan_bin.append(biner(pixelArray, center_tengah_atas, x,y-1))
            barisan_bin.append(biner(pixelArray, center_tengah_atas, x,y))
            barisan_bin.append(biner(pixelArray, center_tengah_atas, x,y+1))
            barisan_bin.append(biner(pixelArray, center_tengah_atas, x-1,y+1))
            barisan_bin = [int(i) for [i] in barisan_bin]
            nilai_bin.append(at_most(barisan_bin))
            
            #reset
            barisan_bin.clear()
            
            #Kanan Atas
            barisan_bin.append(biner(pixelArray,center_atas_kanan, x-1,y))
            barisan_bin.append(biner(pixelArray, center_atas_kanan, x,y))
            barisan_bin.append(biner(pixelArray, center_atas_kanan, x, y+1))
            barisan_bin = [int(i) for [i] in barisan_bin]
            nilai_bin.append(at_most(barisan_bin))
            
            #reset
            barisan_bin.clear()
            
            #Tengah Kanan
            barisan_bin.append(biner(pixelArray, center_kanan_tengah, x-1, y+1))
            barisan_bin.append(biner(pixelArray, center_kanan_tengah, x-1,y))
            barisan_bin.append(biner(pixelArray, center_kanan_tengah, x,y))
            barisan_bin.append(biner(pixelArray, center_kanan_tengah, x+1,y))
            barisan_bin.append(biner(pixelArray, center_kanan_tengah, x+1,y+1))
            barisan_bin = [int(i) for [i] in barisan_bin]
            nilai_bin.append(at_most(barisan_bin))
            
            #reset
            barisan_bin.clear()
            
            #Bawah Kanan
            barisan_bin.append(biner(pixelArray, center_kanan_bawah, x+1,y))
            barisan_bin.append(biner(pixelArray, center_tengah_bawah, x,y))
            barisan_bin.append(biner(pixelArray, center_tengah_bawah, x,y+1))
            barisan_bin = [int(i) for [i] in barisan_bin]
            nilai_bin.append(at_most(barisan_bin))
            
            #reset
            barisan_bin.clear()
            
            #Bawah Tengah
            barisan_bin.append(biner(pixelArray, center_tengah_bawah, x+1,y-1))
            barisan_bin.append(biner(pixelArray, center_tengah_bawah, x,y-1))
            barisan_bin.append(biner(pixelArray, center_tengah_bawah, x,y))
            barisan_bin.append(biner(pixelArray, center_tengah_bawah, x,y+1))
            barisan_bin.append(biner(pixelArray, center_tengah_bawah, x+1,y+1))
            barisan_bin = [int(i) for [i] in barisan_bin]
            nilai_bin.append(at_most(barisan_bin))
            
            #reset
            barisan_bin.clear()
            
            #Bawah kiri
            #gunakan .extend
            barisan_bin.append(biner(pixelArray, center_kiri_bawah, x,y-1))
            barisan_bin.append(biner(pixelArray, center_kiri_bawah, x,y))
            barisan_bin.append(biner(pixelArray, center_kiri_bawah, x+1,y))
            barisan_bin = [int(i) for [i] in barisan_bin]
            nilai_bin.append(at_most(barisan_bin))
            
            #reset
            barisan_bin.clear()
            
            #Tengah Kiri
            barisan_bin.append(biner(pixelArray, center_kiri_tengah,x-1,y-1))
            barisan_bin.append(biner(pixelArray, center_kiri_tengah,x-1,y))
            barisan_bin.append(biner(pixelArray, center_kiri_tengah,x,y))
            barisan_bin.append(biner(pixelArray, center_kiri_tengah,x+1,y))
            barisan_bin.append(biner(pixelArray, center_kiri_tengah,x+1,y-1))            
            barisan_bin = [int(i) for [i] in barisan_bin]
            nilai_bin.append(at_most(barisan_bin))
            
            # #Tengah
            # barisan_bin.append(biner(pixelArray, center_tengah,x-1,y-1))
            # barisan_bin.append(biner(pixelArray, center_tengah,x-1,y))
            # barisan_bin.append(biner(pixelArray, center_tengah,x-1,y+1))
            # barisan_bin.append(biner(pixelArray, center_tengah,x,y+1))
            # barisan_bin.append(biner(pixelArray, center_tengah,x+1,y+1))
            # barisan_bin.append(biner(pixelArray, center_tengah,x+1,y))
            # barisan_bin.append(biner(pixelArray, center_tengah,x+1,y-1))
            # barisan_bin = [int(i) for [i] in barisan_bin]
            # nilai_bin.append(at_most(barisan_bin))
            #ketetanggan = [] inisialisasi di luar loop
            # ketetanggaan.append(barisan_bin[:])
            
            #reset
            barisan_bin.clear()
            
            total = int(circular_fnlbp(nilai_bin))
            img_FNLBP1.append(total)
            img_FNLBP[x][y] = total
    return  img_FNLBP, img_FNLBP1

def mean_FNLBP(listLBP):
    # h,w = np.shape(img_FNLBP)
    L = max(listLBP)
    length = len(listLBP)
    # listCount = []
    prob = 0
    mean = 0
    for i in range(0, L+1):
        num = listLBP.count(i)
        # listCount.append([i,num])
        prob_i = num/length
        prob += prob_i
        #Mean
        mean += i * prob_i
    return mean

def std_FNLBP(listLBP):
    # h,w = np.shape(img_FNLBP)
    L = max(listLBP)
    length = len(listLBP)
    m = mean_FNLBP(listLBP)
    std = 0
    for i in range(0, L+1):
        num = listLBP.count(i)
        prob_i = num/length
        #Standar Deviasi
        std += ((i - m)**2) * prob_i
    totalstd = np.sqrt(std)
    varian_n = std / (L-1)**2
    return totalstd, varian_n

def skew_FNLBP(listLBP):
    # h,w = np.shape(img_FNLBP)
    L = max(listLBP)
    length = len(listLBP)
    m = mean_FNLBP(listLBP)
    skewness = 0
    for i in range(0, L+1):
        num = listLBP.count(i)
        prob_i = num/length
        #Skewness
        skewness += ((i - m)**3) * prob_i
    skewness = skewness / (L-1)**2
    return skewness

def energi_FNLBP(listLBP):
    # h,w = np.shape(img_FNLBP)
    L = max(listLBP)
    length = len(listLBP)
    energi = 0
    for i in range(0, L+1):
        num = listLBP.count(i)
        prob_i = num/length
        #Energi
        energi += abs(pow(prob_i,2))
    return energi

def smoothness(varian_n):
    # standar_dev,varian_n = std_FNLBP(listLBP)
    smooth = 1 - (1/(1+(varian_n)**2))
    return smooth

def show(imgLBP, nama):
    # imgLBP = PImage.fromarray(imgLBP)
    # imgLBP.show()
    nama = str(nama)
    cv2.imshow(nama, imgLBP)

mulai = 1
akhir = 181
def get_feature(jajan):
    FeatureName = ['jajan','MeanFNLBP', 'stdFNLBP','skewFNLBP', 'energiFNLBP' ,'smoothness']#,'skewFNLBP''energiFNLBP','smoothnessFNLBP']
    df_feature = pd.DataFrame(columns = FeatureName)
    time = datetime.datetime.now()
    path = 'E:\jajanan pasar baru\FERRY\jajanan_sudah_crop/'+ jajan + '/'
    
    for i in range(mulai,akhir):
        img =  cv2.imread(path + jajan +'_crop_' + str(i) + '.jpg')
        print('Citra ' + jajan + ' ke-' + str(i))
        
        Image_FNLBP, list_FNLBP = FN_LBP(img)
        
        #Mean
        mean_lbp = mean_FNLBP(list_FNLBP)
        
        #STD
        std_lbp, varian_n = std_FNLBP(list_FNLBP)
        
        #Skew
        skew_lbp = skew_FNLBP(list_FNLBP)
        
        #Energi
        energi_lbp = energi_FNLBP(list_FNLBP)
        
        #Smoothness
        smoothness_lbp = smoothness(varian_n)
        
        jajanan = jajan + ' ke - ' + str(i)
        df_feature.loc[i] = [jajanan, mean_lbp, std_lbp,skew_lbp,energi_lbp,smoothness_lbp]
        df_feature.to_excel('Feature Latih Tekstur FNLBP ' +jajan+' Neww.xlsx')
        
        time_out = datetime.datetime.now()
        Real_time = time_out - time
        print('Finish => '+str(Real_time))
        
        list_FNLBP[:].clear()
        #del img_FNLBP[:]

if __name__ == '__main__':
    get_feature('Ape')
    get_feature('DadarGulung')
    get_feature('Lumpur')
    get_feature('PutuAyu')
    get_feature('Soes')
    get_feature('BikaAmbon')
    get_feature('Bolu')
    get_feature('CumCum')
    get_feature('Getas')
    get_feature('Lapis')
    get_feature('Pukis')

# def stdLBP(listLBP):
#     lengthSTD = len(listLBP)
#     m = meanLBP(listLBP)
#     totalSTD = 0
#     for i in range(lengthSTD):
#         totalSTD += (listLBP[i] - m)**2
#     hasil = totalSTD/lengthSTD
#     return math.sqrt(hasil)

# def skewnessLBP(listLBP):
#     lengthSkew = len(listLBP)
#     mSkew = meanLBP(listLBP)
#     totalSkew1 = 0
#     for i in range(lengthSkew):
#         totalSkew1 += (listLBP[i] - mSkew)**3
#     hasil = totalSkew1/lengthSkew
#     return np.cbrt(hasil) #math.pow(hasil,1/3) #akar 3, menjadi pangkat 1/3

# def energiLBP(listLBP):
    
#     return 0