# preparing and saving the eye state dataset to a npy file format

#import necessary libraries
import numpy as np
import cv2 as cv
from random import shuffle
import glob

#get the dataset from a directory and create a list of labels i.e. 0 and 1 then shuffle them 
file_path = 'eyedataset/'
files = glob.glob( file_path + '/*.jpg')
labels = [0 if 'closed' in file else 1 for file in files]
shuffle_ = False #must be True
if shuffle_:
    zipd = list(zip(files,labels))#zip
    shuffle(zipd)#shuffle
    files,labels = zip(*zipd)#unzip



#fortraining_sets
train_files = files[0:int(0.83*len(files))]
train_labels = labels[0:int(0.83*len(labels))]
#for test
test_files = files[int(0.83*len(files)):]
test_labels = labels[int(0.83*len(labels)):]

#read the images and save them to a list
X = [] #x_train = []
for i in files:
    img = cv.imread(i)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    X.append(img)
X = np.array(X)

x_test = []
for i in test_files:
    img = cv.imread(i)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    x_test.append(img)
x_test = np.array(x_test)
x_train[1].shape


Y = np.array(labels)
#test_labels = np.array(test_labels)

#Y = Y.reshape(-1,1)
#test_labels = test_labels.reshape(-1,1)

X = X.reshape(X.shape[0],-1).T
#x_test = x_test.reshape(x_test.shape[0],-1).T
#Y = Y.T
#y_test = test_labels.T

print(X.shape)
print(Y.shape)
print('============')
#print(x_test.shape)
#print(y_test.shape)

np.save('labels.npy',Y)
np.save('dataset.npy',X)


np.save('x_train.npy',x_train)
np.save('x_test.npy',x_test)
np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)



x = np.load('x_train.npy')
x.shape

pwd

