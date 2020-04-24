# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:17:55 2019

@author: jiwandhono
"""
from PIL import Image as PImage
import numpy as np
import pandas as pd
import math
import datetime
import cv2

# https://programmingdesignsystems.com/color/color-models-and-color-spaces/index.html
#def mainHSV():
img = PImage.open('Bolu1.jpg')
# img = PImage.open(path+'\PutuAyu_crop_.jpg')
# imRGB = im.convert('RGB')
# imHSV = img.convert('HSV')

# hsvPIL = np.uint8(imHSV)

# img = cv2.imread('Bolu1.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# pixelImage = list(imRGB.getdata())
# pixelImage = [[255,120,100]]
#pixelImage = [[1,2,3],[4,5,6],[7,8,9],[11,12,13]]
#pixelImage = np.random.random_integers(1,255,(10,3))
#pixelImage = np.asarray(pixelImage).tolist()
#pixelImage = tuple(pixelImage[0])


def convertRGB_HSV(image):
    pixelImage = list(image.getdata())
    newPixelHSV = []
    newPixelH = []
    newPixelS = []
    newPixelV = []
    newPixel = []
    for pixel in pixelImage:
        R = pixel[0]/255
        G = pixel[1]/255
        B = pixel[2]/255
        
        maxPix = max(R,G,B)
        minPix = min(R,G,B)
        #Value
        V = maxPix
        delta = V - minPix
        #Saturation
        if(V > 0):
            S = delta/V
        else:
            S = 0
        #Hue
        if(S == 0):
            H = 0
        elif(V == R):
            H = (60/360) * (((G-B)/delta) % 6)
        elif(V == G):
            H = (60/360) * (2 + ((B-R)/delta))
        elif(V == B):
            H = (60/360) * (4 + ((R-G)/delta))
        
        newPixel = [H,S,V]
        newPixelH.append(newPixel[0])
        newPixelS.append(newPixel[1])
        newPixelV.append(newPixel[2])
        newPixelHSV.append(newPixel)
        
    return newPixelHSV,newPixelH,newPixelS,newPixelV    

def convert_values(h,s,v):
    new_h = [i * 360 for i in h]
    new_h = [round(i) for i in new_h]
    
    
    new_s = ["{:.2}".format(float(i)) for i in s]
    new_s = [float(i) * 100 for i in new_s]
    new_s = [int(i) for i in new_s]
    
    new_v = ["{:.2}".format(float(i)) for i in v]
    new_v = [float(i) * 100 for i in new_v]
    new_v = [int(i) for i in new_v]
    
    merge_hsv = [(new_h[i], new_s[i], new_v[i]) for i in range(len(new_h))]
    merge_hsv = [list(i) for i in merge_hsv]
    return merge_hsv, new_h,new_s,new_v

def display_hsv(img,hsv):
    image_out = PImage.new('HSV', img.size)
    for i in range(len(hsv)):
        for j in range(len(hsv[i])):
            hsv[i][j] = round(float(hsv[i][j]))
            
    hsv = [tuple(i) for i in hsv]
    image_out.putdata(hsv)
    # image_out.show()
    image_out = np.uint8(image_out)
    return cv2.imshow('HSV', image_out), image_out

def mean(listWarna):
    length = len(listWarna)
    total = 0
    # for i in range(length):
    #     total +=  listWarna[i]
    total = sum(listWarna)
    rata2 = total/length
    return rata2

def std(listWarna):
    lengthSTD = len(listWarna)
    m = mean(listWarna)
    totalSTD = 0
    for i in range(lengthSTD):
        totalSTD += (listWarna[i] - m)**2
    hasil = totalSTD/lengthSTD
    return math.sqrt(hasil)

def skewness(listWarna):
    lengthSkew = len(listWarna)
    mSkew = mean(listWarna)
    totalSkew1 = 0
    for i in range(lengthSkew):
        totalSkew1 += (listWarna[i] - mSkew)**3
    hasil = totalSkew1/lengthSkew
    return np.cbrt(hasil) #akar 3, menjadi pangkat 1/3

#Periksa Kurtosis
def kurtosis(listWarna):
    lengthKurt = len(listWarna)
    mKurt = mean(listWarna)
    totalKurt1 = 0
    for i in range(lengthKurt):
        totalKurt1 += (listWarna[i] - mKurt)**4
    hasil = totalKurt1/lengthKurt
    hasil = math.pow(hasil,1/4)
    return hasil - 3

def show(nama,imgHSV):
    imgHSV = np.uint8(imgHSV)
    cv2.namedWindow(nama, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nama, 700,700)
    return cv2.imshow(nama,imgHSV)

def get_feature(path,jajan,mulai,akhir):
    FeatureName = ['jajan','Mean H', 'Mean S', 'Mean V', 'STD H','STD S', 'STD V',
                   'SKEW H', 'SKEW S', 'SKEW V', 'Kurt H','Kurt S','Kurt V' ]
    df_feature = pd.DataFrame(columns = FeatureName)
    time = datetime.datetime.now()
    
    for i in range(mulai,akhir):
        img =  PImage.open(path + jajan +'/' + jajan + '_crop_' + str(i) + '.jpg')
        print('Citra ' + jajan + ' ke-' + str(i))
        
        image_HSV, pixelH,pixelS,pixelV = convertRGB_HSV(img)
        
        merge, new_h,new_s,new_v = convert_values(pixelH,pixelS,pixelV)
        #Mean
        mean_H = mean(new_h)
        mean_S = mean(new_s)
        mean_V = mean(new_v)
        
        #STD
        std_H = std(new_h)
        std_S = std(new_s)
        std_V = std(new_v)
        
        #Skew
        skew_H = skewness(new_h)
        skew_S = skewness(new_s)
        skew_V = skewness(new_v)
        
        kurt_H = kurtosis(new_h)
        kurt_S = kurtosis(new_s)
        kurt_V = kurtosis(new_v)
        
        jajanan = jajan + ' ke - ' + str(i)
        df_feature.loc[i] = [jajanan, mean_H, mean_S,mean_V, std_H, std_S,std_V,
                             skew_H,skew_S,skew_V, kurt_H,kurt_S,kurt_V]
        df_feature.to_excel('Feature Latih Warna ' +jajan+' New.xlsx')
        
        time_out = datetime.datetime.now()
        Real_time = time_out - time
        print('Finish => '+str(Real_time))
        
        new_h[:].clear()
        new_s[:].clear()
        new_v[:].clear()

def main(jajan):
    mulai = 1
    akhir = 181
    path = 'E:\jajanan pasar baru\FERRY\jajanan_sudah_crop/'
    get_feature(path, jajan, mulai, akhir)

if __name__ == '__main__':
    main('Ape')
    main('DadarGulung')
    main('Lumpur')
    main('PutuAyu')
    main('Soes')
    main('BikaAmbon')
    main('Bolu')
    main('CumCum')
    main('Getas')
    main('Lapis')
    main('Pukis')    
    