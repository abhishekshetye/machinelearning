import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
'''
input > weight > hidden layer 1 (activation func) > weights > hidden layer 2 >
activation function > weights > output layer

compare output cost function (cross entropy)
optimization function > minimize cost (adam optimizer... SGD, AdaGrad)

backpropogation


feed forward + backprop = epoch
'''


mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100
logs_path = "/tmp/mnist_demo/1"

# height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def neural_network_model(data):
	#raw data
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 
					'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
					'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
					'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 
					'biases': tf.Variable(tf.random_normal([n_classes]))}


	# (input * weights) + biases

	#layer 1
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	#activation function
	l1 = tf.nn.relu(l1)

	#layer 2
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	#activation function
	l2 = tf.nn.relu(l2)

	#layer 3
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	#activation function
	l3 = tf.nn.relu(l3)

	#output layer
	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
	
	#one hot array
	return output


def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
	# learning rate default 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)


	epochs = 10

	with tf.Session() as sess:
		#sess.run(tf.initialize_all_variables())
		train_writer = tf.summary.FileWriter(logs_path,  sess.graph)
		tf.global_variables_initializer().run()
		#writer.add_graph(sess.graph)
		
		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				ex, ey = mnist.train.next_batch(batch_size)
				print(ex)
				_, c = sess.run([optimizer, cost], feed_dict={x: ex, y: ey})
				epoch_loss += c
			#train_writer.add_summary(c, epoch)
			print("loss is %f" % epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
		train_writer.close()

train_neural_network(x)










