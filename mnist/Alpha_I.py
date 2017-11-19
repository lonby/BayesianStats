#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a script to replicate python codes for bayesian deep learning
, described by Alpha-I blog. This uses MNIST datasets to classify 0~9.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import edward as ed
from edward.models import Categorical, Normal

#%% Import datasets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
ed.set_seed(314159)
N = 100 # number of images in a minibatch
D = 784 # number of features (28 by 28)
K = 10  # number of classes

#%% Preparation for 1 layer neural network
x = tf.placeholder(tf.float32, [None,D])
w = Normal(loc=tf.zeros([D,K]), scale=tf.ones([D,K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
y = Categorical(tf.matmul(x,w)+b)

#%% Preparation for Variational Inference
# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [N])
# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({w: qw, b: qb}, data={y:y_ph})

# Initialse the infernce variables
inference.initialize(n_iter=5000, 
                     n_print=100, scale={y: float(mnist.train.num_examples) / N})


#%% Begin to train the model
# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()

# Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)



#%% Test the model
# Load the test images.
X_test = mnist.test.images
# TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
Y_test = np.argmax(mnist.test.labels,axis=1)

# Generate samples the posterior and store them.
n_samples = 100
prob_lst = []
samples = []
w_samples = []
b_samples = []
for _ in range(n_samples):  # sample 100개 추출을 위해 다음의 과정을 100번 반복
    w_samp = qw.sample()    # extract samples from posterior distribution of qw
    b_samp = qb.sample()    # extract samples from qb
    w_samples.append(w_samp)    # sample 100개 수집하기 위해 append
    b_samples.append(b_samp)    # sample 100개 수집하기 위해 append
    # Also compue the probabiliy of each class for each (w,b) sample.
    prob = tf.nn.softmax(tf.matmul( X_test,w_samp ) + b_samp)   # forward로 sample의 yhat 추출
    prob_lst.append(prob.eval())    # 추출한 w와 b의 forward 값을 수집
    sample = tf.concat([tf.reshape(w_samp,[-1]),b_samp],0) # tf.reshape : flatten! 벡터화!
                                 # tf.concat([~,~],0) 은 raw로 합치기, 1은 column으로 합치기
    samples.append(sample.eval())   # tensorflow 타입의 sample을 value로 변환



#%% Calculate the accuracy of the model
# For each sample we compute the predicted class and compare with the test labels.
# Predicted class is defined as the one which as maximum proability.
# We perform this test for each (w,b) in the posterior giving us a set of accuracies
# Finally we make a histogram of accuracies for the test data.
accy_test = []
for prob in prob_lst:
    y_trn_prd = np.argmax(prob,axis=1).astype(np.float32)   # np.argmax는 가장 큰 값의 idx출력
                            # 즉, 0~9 중의 idx중 가장 높은 확률값이 있는 idx를 출력하며 그것이 y_hat!
    acc = (y_trn_prd == Y_test).mean()*100
    accy_test.append(acc)

plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")

# Here we compute the mean of probabilties for each class for all the (w,b) samples.
# We then use the class with maximum of the mean proabilities as the prediction. 
# In other words, we have used (w,b) samples to construct a set of models and
# used their combined outputs to make the predcitions.
Y_pred = np.argmax(np.mean(prob_lst,axis=0),axis=1) # Traditional 예측의 정확도 측정방법
print("accuracy in predicting the test data = ", (Y_pred == Y_test).mean()*100)



#%% Check the posterior distribution infered by Variational Inference
# Create a Pandas DataFrame of posterior samples.
samples_df = pd.DataFrame(data = samples, index=range(n_samples))
# Now create a small subset by taking the first 5 weights, labelled as W_0, ... , W_4.
samples_5 = pd.DataFrame(data = samples_df[list(range(5))].values,columns=["W_0", "W_1", "W_2", "W_3", "W_4"])
# We use Seaborn PairGrid to make a triale plot to show auto and cross correlations.
g = sns.PairGrid(samples_5, diag_sharey=False)
g.map_lower(sns.kdeplot, n_levels = 4,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,legend=False)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Joint posterior distribution of the first 5 weights')


# Load the first image from the test data and its label.
test_image = X_test[0:1]
test_label = Y_test[0]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')


# Now the check what the model perdicts for each (w,b) sample from the posterior. This may take a few seconds...
sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul( X_test[0:1],w_samp ) + b_samp)
    sing_img_probs.append(prob.eval())



# Create a histogram of these predictions.
plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(10))
plt.xticks(np.arange(0,10))
plt.xlim(0,10)
plt.xlabel("Accuracy of the prediction of the test digit")
plt.ylabel("Frequency")



# As the nonMNIST data is in the same format as MNIST, we can use the TensorFlow functions.
# Please make sure that notMNIST_data directory is in the same directory as this notebook.
# Otherwise, please provide the full path.

### Note that if you haven't executed the bash commands above, TensorFlow method
### WILL download the MNIST data into notMNIST data below and you will get the 
### digit 7 as the first number!
not_mnist = input_data.read_data_sets("./notMNIST_data/", one_hot=True)



# Load the test images from the data and their lables. 
Xnm_test = not_mnist.test.images
# Once again convert one-hot-vector to the corresponding labels.
Ynm_test = np.argmax(not_mnist.test.labels,axis=1)




# Load the first image from the notMNIST.
test_image = Xnm_test[0]
test_label = Ynm_test[0]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')



# As before, compute the proabilities for each class for all (w,b) samples from the posterior.
nm_sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul( Xnm_test[0:1],w_samp ) + b_samp)
    nm_sing_img_probs.append(prob.eval())



# Now compute the histogram of perdictions from the (w,b) samples.
# In our previous test, all the weights from the posterior was able get the correct prediction.
# However, here we see that the model gives a wide range of possibilites.
# Hence we conclude that its cofidence is lower. 
plt.hist(np.argmax(nm_sing_img_probs,axis=2),bins=range(10))
plt.xticks(np.arange(0,10))
plt.xlim(0,10)
plt.xlabel("Accuracy of the prediction of the test letter")
plt.ylabel("Frequency")



%load_ext watermark
%watermark -v -m -p numpy,pandas,edward,tensorflow,seaborn,matplotlib



