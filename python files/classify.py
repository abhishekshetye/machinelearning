
# coding: utf-8

# In[47]:

import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[48]:

data = pd.read_csv('IRIS.csv', names=['f1','n1', 'f2','n2', 'f3','n3', 'f4','n4', 'f5'])
data


# In[49]:

sns.FacetGrid(data, hue="f5", size=5)     .map(plt.scatter, "f1", "f2")     .add_legend()


# In[50]:

#map data into arrays
s = np.asarray([1,0,0])
ve = np.asarray([0,1,0])
vi = np.asarray([0,0,1])
data['f5'] = data['f5'].map({'setosa':s, 'versicolor':ve, 'virginica': vi})
data


# In[51]:

data = data.iloc[np.random.permutation(len(data))]
data


# In[52]:

data = data.reset_index(drop=True)
data


# In[74]:

#training
X_input = data.ix[0:70, ['f1', 'f2', 'f3', 'f4']]
temp = data['f5']
Y_input = temp[0:71]
#test
xtest = data.ix[71:99, ['f1', 'f2', 'f3', 'f4']]
ytest = temp[71:100]


# In[54]:

#placeholders and variables
x = tf.placeholder(tf.float32, shape=[None, 4])
y_ = tf.placeholder(tf.float32, shape=[None, 3])
#weights and bias
w = tf.Variable(tf.zeros([4,3]))
b = tf.Variable(tf.zeros([3]))


# In[55]:

#model
#softmax function for multiclass classification
y = tf.nn.softmax(tf.matmul(x,w) + b)


# In[56]:

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# In[57]:

#optimizer
learning_rate = 0.01
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#calculating accuracy of our model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[58]:

#session parameters
sess = tf.InteractiveSession()
#initializing variables
init = tf.global_variables_initializer()
sess.run(init)
#number of interactions
epoch = 2000


# In[75]:

for step in range(epoch):
    c=sess.run([train_step,cross_entropy], feed_dict={x: X_input, y_:[t for t in Y_input.as_matrix()] })
    if step%50==0 :
       print (c)


# In[87]:

#random testing at Sn.130
a=data.ix[1,['f1','f2','f3','f4']]
b=a.reshape(1,4)
largest = sess.run(tf.arg_max(y,1), feed_dict={x: b})[0]
if largest==0:
    print ("flower is :Iris-setosa")
elif largest==1:
    print ("flower is :Iris-versicolor")
else :
    print ("flower is :Iris-virginica")


# In[88]:

print (sess.run(accuracy,feed_dict={x: xtest, y_:[t for t in ytest.as_matrix()]}))

