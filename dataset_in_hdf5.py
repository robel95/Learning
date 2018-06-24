## saving a dataset using an hdf5 file (.hdf5)

import glob
from random import shuffle
import h5py
import numpy as np
import cv2 as cv

shuffle_data = True #to shuffle the cats and dogs

#create and save the hdf5 file and store the path 
hdf5_path = 'hdf5_files/dataset.hdf5'

#store the path of the images
cats_n_dogs_path = 'hdf5_files/dogscats/train/dogs/*.jpg'

#read adrs of the path of the images using glob(emptylist)
addrs = glob.glob(cats_n_dogs_path)
print(cv.imread(addrs[2]))

#add labels 0 for cats and 1 for dogs
labels = [ 0 if 'cat.' in adrs else 1 for adrs in addrs]



if shuffle_data:
    zipd = list(zip(addrs,labels))#zip
    shuffle(zipd)#shuffle
    addrs,labels = zip(*zipd)#unzip


#divide 60p for train, 20p each for validitation and test
#fortraining_sets
train_addrs = addrs[0:int(0.7*len(addrs))]
train_labels = labels[0:int(0.7*len(labels))]
#for test
test_addrs = addrs[int(0.7*len(addrs)):]
test_labels = labels[int(0.7*len(labels)):]



#size of the images shape
train_shape = (len(train_addrs),224,224,3)
test_shape = (len(test_addrs),224,224,3)


### create hdf5 file and creatte arrays




hdf5_file = h5py.File(hdf5_path,mode = 'w')

hdf5_file.create_dataset('train_img',train_shape,np.int8)# for the train set

hdf5_file.create_dataset('test_img',test_shape,np.int8)#for the test set

hdf5_file.create_dataset('train_mean',train_shape[1:],np.float32)

hdf5_file.create_dataset('test_labels',(len(test_addrs),),np.int8)

hdf5_file["test_labels"][...] = test_labels

hdf5_file.create_dataset('train_labels',(len(train_addrs),),np.int8)

hdf5_file["train_labels"][...] = train_labels

#the dataset is ready now its time to read images one by one and store it to the dataset

#for the mean
mean = np.zeros(train_shape[1:],np.float32)

#for the train addresses
import cv2 as cv
for i in range(len(train_addrs)):
    #get the images
    t_addrs = train_addrs[i]
    #read the image
    img = cv.imread(t_addrs)
    #resize the image
    img = cv.resize(img,(224,224),interpolation = cv.INTER_CUBIC)
    #convert bgr to rgb
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))
    
    

import cv2 as cv
#for the test addresses
for i in range (len(test_addrs)):
    #get the images
    test_ad = test_addrs[i]
    #read the images
    img = cv.imread(test_ad)
    #resize
    img = cv.resize(img,(224,224),interpolation = cv.INTER_CUBIC)
    #BGR2RGB
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    #save the img
    hdf5_file["test_img"][...] = img[None]
    

#save mean and close the hdf5 file
hdf5_file["train_mean"][i, ...] = mean
hdf5_file.close()

