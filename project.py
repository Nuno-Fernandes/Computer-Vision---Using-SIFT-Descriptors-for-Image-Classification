##################################################################################################################
#   University of Porto 
#   Faculty of Engineering
#   Computer Vision
#
# Project 2: Objects classification
#
# Authors:
#   * Katja Hader up201602072
#   * Nuno Granja Fernandes up201107699
#   * Samuel Arleo Rodriguez up201600802
##################################################################################################################

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join, splitext
import csv
import classification, representation, featurex
import time
from collections import Counter

class descriptor:
    def __init__(self, vector=None, label=None):
        self.vector = vector
        self.label = label

class image:
    def __init__(self, img=None, name=None, keyp=None, desc=None, hist=None):
        self.img = img
        self.id = None
        self.label = None           # Image class
        self.desc = desc            # Array of descriptor objects
        self.keyp = keyp            # Array of keypoints
        self.histogram = hist 
        self.name = name

    # Given a list of labels, assign them to their correspondent descriptor and computes img histogram
    def set_labels(self,labels,desc_size):
        self.histogram = np.zeros(desc_size, dtype=object)
        for i in range(0,len(labels)):
            self.desc[i].label = labels[i]
            self.histogram[label[i]] += 1
class images_set:
    def __init__(self, path_img=None, path_lab=None):
        self.images = None
        self.path_img = path_img
        self.path_lab = path_lab

    def load_images(self,inf,sup):
        try:
            # Creating a list with all pictures names
            onlyfiles = [ f for f in listdir(self.path_img) if isfile(join(self.path_img,f)) ]
            onlyfiles = sorted(onlyfiles)[inf:sup]
            self.images = np.empty(len(onlyfiles), dtype=object)
            # Reading each image and storing it into the images array
            for n in range(0, len(onlyfiles)):
                name = onlyfiles[n]
                self.images[n] = image(cv2.imread(join(self.path_img,name),0),int(splitext(name)[0]))
        except:
            print "Error opening the folder ",self.path_img,".Please check the file location."
            exit()

    def load_labels(self,inf,sup):
        try:
            # Loading labels into an matrix with columns |id|label|
            labels = np.genfromtxt(self.path_lab, delimiter=',',
                dtype=[('id','<i8'),('label','|S5')], skip_header=1)
        except:
            print "Error opening the file ",self.path_lab,".Please check the file location."
            exit()
        n = 0
        # Adding id and label to each image
        for (x,y) in labels[inf:sup]:
            self.images[n].id = x
            self.images[n].label = y
            if n == len(self.images)-1:
                break
            n += 1

    # Gets the list of descriptors (not descriptor objects yet) and returns a list of desc. objects
    # that have the actual descriptor in the vector attribute
    def build_desc(self, desc_list):
        return map(lambda x: descriptor(x), desc_list)                  # For each desc in desc_list map replaces it 
                                                                        # by a descriptor object
    # Assings descriptors and keypoints to their correspondent image
    # Args: features contains pairs of |Keypoints| Descriptors|
    def get_features(self, features):
        for i in range(0,len(features)):
            if features[i][1] is not None:                              # To discard pictures without keypoints
                self.images[i].desc = self.build_desc(features[i][1])   # Assign to each image a list of desc. objects
                self.images[i].keyp = features[i][0]
            #else: 
            #   self.images[i].desc = np.array([])

    # Computes the histogram of each image and assigns the label to each descriptor
    def set_descr_labels(self,labels,desc_size,bag_size):
        index = 0
        for img in self.images:                                         # For each image in the set
            img.histogram = np.zeros(desc_size, dtype=object)           # Initializes the histogram vector (250 words so far)
            if img.desc is not None:                                    # Discard images without keypoints
                for desc in img.desc:                                   # For each descriptor on each image
                    label = labels[index][0]                            # Stores the label of the current descriptor
                    desc.label = label                                  # Assigns to the descriptor its label
                    img.histogram[label] += 1                           # Adds 1 to the element in the position of the
                    index += 1                                          # label value. Ej label = 3, histogram[3] += 1
            else:
                img.histogram = np.zeros(bag_size,dtype=np.float32)

    # Classify descriptors of test images
    def classify_desc(self, centers, bag_size):
        desc_size = centers.shape[0]
        labels = np.linspace(0,desc_size-1,num=desc_size,
                                         dtype=np.int32).reshape(-1,1)  # Array with labels from (0 to 249)
        knn = cv2.ml.KNearest_create()
        knn.train(centers,cv2.ml.ROW_SAMPLE,labels)
        for img in self.images: 
            if img.desc is not None:
                desc = np.array(map(lambda x: x.vector, img.desc))          # Putting together descriptors of each image (x = descriptor)
                ret,result,neighbours,dist = knn.findNearest(desc,k=1)
                img.set_labels(desc, desc_size)                             # Set labels of descriptors of img
            else:
                img.histogram = np.zeros(bag_size,dtype=np.float32)

# Stores all descriptors of the set of images in a single variable to cluster them
def join_desc(res):
    # res has columns |Keypoint|Descriptors| and each row represent a keypoint
    # tmp stores just the descriptors
    tmp = [res[i][1] for i in range(0,len(res)) if res[i][1] is not None]
    # Getting descriptors size (all have the same given by SIFT: 128)
    desc_size = tmp[0][0].shape[0]
    # Counting number of descriptors
    num_desc = 0
    for img in tmp:
        for desc in img:
            num_desc += 1
    # Storing descriptors in des, but before we initialze it empty with the right dimensions: [num_desc,128]
    des = np.zeros((num_desc,desc_size))
    n = 0
    for img in tmp:
        for desc in img:
            des[n,:] = desc
            n += 1
    return des

#----------------------- LOADING DATA --------------------------

# Paths to the training and test data
path_train_imgs = "/home/samuel/CV2/train_data/"
path_test_imgs = "/home/samuel/CV2/train_data/"

# File with labels of training images
train_labels = "/home/samuel/CV2/labels_train.csv"
test_labels = "/home/samuel/CV2/labels_train.csv"

# Using a subset of the images set
inf_tr = 0
sup_tr = 9000
inf_ts = 9000
sup_ts = 10000

# Creating object images_set that encapsulates methods for loading images and labels,
# and also stores the loaded images and labels
train_set = images_set(path_train_imgs, train_labels)
test_set = images_set(path_test_imgs, test_labels)

# Loading sup-inf number of images 
train_set.load_images(inf_tr, sup_tr)
test_set.load_images(inf_ts, sup_ts)

# Loading labels of previously loaded pictures
train_set.load_labels(inf_tr,sup_tr)
test_set.load_labels(inf_ts,sup_ts)

#------------------- EXTRACTING FEATURES -----------------------

# Instantiating sift class
sift = cv2.xfeatures2d.SIFT_create()

# Applying SIFT to all training images. This returns the tuple (keypoints, descriptor)
# for each image, and it's transformed to a matrix with columns:
# |Keypoints| Descriptors|
res_train = map(lambda x: sift.detectAndCompute(x.img, None), train_set.images)
res_test = map(lambda x: sift.detectAndCompute(x.img, None), test_set.images)

# Storing each descriptor and keypoint with its image
train_set.get_features(res_train)
test_set.get_features(res_test)

# Storing all descriptors of training images in a single variable to cluster them
desc = join_desc(res_train)

# Changing type to float32 which is required by the kmeans function
desc = desc.astype('float32')

# Parameters of the k-means algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Number of cluster that will set the number of words of each bag
words_number = 150

# Measuring time of k-means
start = time.time()

# Applying k-means to all the descriptors
ret,label,centers=cv2.kmeans(desc,words_number,None,criteria,4,cv2.KMEANS_RANDOM_CENTERS)

# Computing and showing k-means run time
end = time.time()
print(end - start)

#------------------- REPRESENTATION STEP -----------------------

# Giving a label to each descriptor. Also passing size of descriptors: desc[0].shape[0]
train_set.set_descr_labels(label, words_number, words_number)

# Classifying descriptors of test images. Each descriptor can belong to 250 classes, i.e,
# each new word will be more similar to one of the 250 visual words computed in k-means
test_set.classify_desc(centers, words_number)

#------------------- CLASSIFYING STEP ------------------------

# Putting together all the bag of words
bw_train = np.array(map(lambda x: x.histogram, train_set.images),dtype=np.float32)
bw_test = np.array(map(lambda x: x.histogram, test_set.images),dtype=np.float32)


# Creating arrays with labels to pass them to the predictors
labels_tr = np.array(map(lambda x: x.label,train_set.images))
labels_ts = np.array(map(lambda x: x.label,test_set.images))

# Images IDs
ids_tr = np.array(map(lambda x: x.label,train_set.images))

# Changing format of labels to int
classes, numeric_tr = np.unique(labels_tr, return_inverse=True)
numeric_tr = (numeric_tr).astype(np.int32)

# Training kNN with bag of words of the training set
knn = cv2.ml.KNearest_create()
knn.train(bw_train,cv2.ml.ROW_SAMPLE,numeric_tr) 

# Predicting image class
ret,result,neighbours,dist = knn.findNearest(bw_test,k=1)

result = result.astype(np.int32)

pos = 0
neg = 0
for i in (classes[result]==labels_ts.reshape(-1,1)):
    if i:
        pos += 1
    else:
        neg += 1

print("POS: ",pos," NEG: ",neg)
print(float(pos)/float(neg))
#print(classes[result.astype(np.int32)] == labels_ts)

# - Si no funciona revisar los labels que retorna kmeans. Si retorna los centroides en orden: el primero de la lista
# es el que tiene label 1 (cluster 1) y asi, entonces esta bien. Si no, agarrar descriptores del train que ya tienen
# labels y buscar a que cluster pertenecen
# - Revisar simmilarity measure del knn cuando se comparen bag of words


# Entrenar usando todas las entradas con -1 y el mismo numero de entradas con 1
# Buscar en internet correlation of numeric var with nominal (test con p y chi, etc)
# Usar el decision tree en vez de logistic reg
