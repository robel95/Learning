## deep neural network implementation using tensorflow framework

import numpy as np
import tensorflow as tf
import sklearn.model_selection as sk


#define parameters
batch_size = 100
learning_rate = 0.008

tf.reset_default_graph()
max_step = 35000

# get data
dataset = np.load('dataset.npy')
labels = np.load('labels.npy')

labels.shape
dataset = dataset.T
dataset.shape

images_placeholder = tf.placeholder(tf.float32,shape=[3634,576])
labels_placeholder = tf.placeholder(tf.int64, shape = None)

#weights and biases
w = tf.get_variable('w', shape=[576,2],initializer=tf.contrib.layers.xavier_initializer(seed = 1))
b = tf.Variable(tf.zeros([2]))                

#function z
z = tf.matmul(images_placeholder,w) + b
#activations #1 sigmoid
z.shape


#loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = z,labels = labels_placeholder))

#training model
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#prediction
# Operation comparing prediction with true label
correct_prediction = tf.equal(tf.argmax(z, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#spliting the dataset into test and train sets for x and y
x_train,x_test,y_train,y_test = sk.train_test_split(dataset,labels)

x_train.shape




with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.initialize_all_variables())
    a=[]
# Repeat max_steps times
    for i in range(max_step):
        images_batch = x_train
        labels_batch = y_train
        # Periodically print out the model's current accuracy
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
            sess.run(train_step, feed_dict={images_placeholder: images_batch,labels_placeholder: labels_batch})
            a.append(train_accuracy)
    
            #print('train_step',train_step)
            # After finishing the training, evaluate on the test set
            #test_accuracy = sess.run(accuracy, feed_dict={images_placeholder: x,labels_placeholder: y})
            #print('Test accuracy {:g}'.format(test_accuracy))
        print(sess.run(w))

max(a)









lr = [1,1.3,1.5,1.7,1.9,.1,.3,.5,.7,.9,.01,.03,.05,.07,0.09,.002,.004,.006,.008]
lr

for alpha in lr:
    a=[]
    train_step = tf.train.AdamOptimizer(learning_rate = alpha,beta1 = 0.9, beta2 = 0.99, epsilon=1e-08, use_locking=False, name='Adam').minimize(loss)
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.initialize_all_variables())

    # Repeat max_steps times
        #loop


        
        for i in range(max_step):
            images_batch = x_train
            labels_batch = y_train
            # Periodically print out the model's current accuracy
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={images_placeholder: images_batch, labels_placeholder: labels_batch})
                #print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))
                sess.run(train_step, feed_dict={images_placeholder: images_batch,labels_placeholder: labels_batch})
                a.append(train_accuracy)
        print('for alpha',alpha,'max accuracy = ',max(a))

