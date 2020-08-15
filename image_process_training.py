import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv
import csv
from scipy.stats import kurtosis, skew
import glob
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans
from collections import Counter
import math
import pandas as pd

scaler = MinMaxScaler()

df = pd.read_csv('image_feature.csv')
img_features = []
for i in range(0, 1499):
    temp = []
    for x in df:
        temp.append(df[x][i])
        
    img_features.append(temp)

train_feat_only = [[x for x in y[0:28]] for y in img_features]
scaled_train_feat = scaler.fit_transform(train_feat_only)
rounded_scale = [[round(x, 5) for x in y] for y in scaled_train_feat]

def get_color_feature(img):
    means = np.mean(img)
    stdev = np.std(img)
    skewness = [skew(x) for x in img]
    kurt = [kurtosis(x) for x in img]

    return [means, stdev, skew(skew(skewness)), kurtosis(kurtosis(kurt))]

def calc_glcm_all_agls(img, label, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    feature.append(label) 
    
    return feature

def glcm_features(img, label="no label"):
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    glcm_feat = calc_glcm_all_agls(gray, label, props=properties)
        
    return glcm_feat

# fungsi euclidean distance
def euclidean_distance(input_data, dataset):
    distances = []
    for data in dataset:
        dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(input_data, data)]))
        distances.append(dist)
    
    return distances

# fungsi klasifikasi knn
def knn(k, datatest, datatrain):
    data = datatrain
    
    data_without_label = [x[0:28] for x in data]
    label_only = [x[-1] for x in data]
    
    distance = euclidean_distance(datatest, data_without_label)
    zipped = list(zip(distance, label_only))
    temp = sorted(zipped, key=lambda x: x[0])
    
    nearest_neighbor = temp[:k]
    
    most_label = [x[1] for x in nearest_neighbor]
    
    return Counter(most_label).most_common(1)[0][0]

def sort_by_cluster(data):
    # inisialisasi variabel baru untuk menyimpan data berdasar cluster
    new_data = []

    # pemisahan data untuk setiap cluster
    for i in range(0, 150):
        temps = []
        for x in data:
            if i == x[1]:
                temps.append(x[0])
        new_data.append(temps)

    return new_data

def train(dataset):
    labels = [x[-1] for x in dataset]
    feat_only = [[x for x in y[0:28]] for y in dataset]
    scaled = scaler.fit_transform(feat_only)
    rounded_scale = [[round(x, 5) for x in y] for y in scaled]
    kmeans = KMeans(n_clusters=150, max_iter=200).fit(rounded_scale)
    y = kmeans.predict(rounded_scale)

    # menyatukan feature dengan label yang sesuai
    with_nama_makanan = []
    for i in range(len(scaled)):
        label = [labels[i]]
        with_nama_makanan.append(rounded_scale[i] + label)

    # menyatukan citra dengan cluster hasil clustering k-means
    with_label = list(zip(with_nama_makanan, y))

    # mengurutkan hasil clustering
    temp = sorted(with_label, key=lambda x: x[1])

    return [sort_by_cluster(temp), kmeans.cluster_centers_]

def normalize_test(data_train, data_test):
    data_train.append(data_test)
    scaled = scaler.fit_transform(data_train)
    
    return scaled[-1]

def test(data_test):
    global img_features, rounded_scale
    train_result = train(img_features)
    
    blurtest = cv.GaussianBlur(data_test, (5,5), 0)
    feature_test = get_color_feature(blurtest) + glcm_features(data_test)

    normalized_test = normalize_test(train_feat_only, feature_test[0:28])

    cluster_distance_test = euclidean_distance(normalized_test, train_result[1])
    nearest_cluster = cluster_distance_test.index(min(cluster_distance_test))
    classification_result = knn(3, normalized_test, train_result[0][nearest_cluster])

    return classification_result
# print(pd.DataFrame(res))
