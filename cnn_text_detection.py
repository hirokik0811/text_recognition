'''
Created on Sep 25, 2017

@author: hiroki
'''

import tensorflow as tf

"""
Classifies if the 32x32 patch from an image has a character on its center.
0: Non character or off centered, 1: character on centered.
CNN with 2 convolutional layers, 2 average pooling layers, and 2 fully connected layers
"""
N_STEPS = 10000
BATCH_SIZE = 200
INPUT_SIZE = 32 # all input image batches should be 32 by 32
OUTPUT_SIZE = 2 # character or not
N1 = 96 # number of filters used for first convolutional layer and average pooling layer
N2 = 128 # number of filters used for second convolutional layer and average pooling layer
CONV1_FILTER_SIZE = 8
AVGPOOL1_FILTER_SIZE = 5
CONV2_FILTER_SIZE = 2
AVGPOOL2_FILTER_SIZE = 2
DROPOUT_RATE = 0.75
FC_SIZE = 1024

x = tf.placeholder(tf.float32, [None, INPUT_SIZE*INPUT_SIZE])
y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

# input layer
input = tf.reshape(x, [-1, INPUT_SIZE, INPUT_SIZE, 1])

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')
def avgpool(x, ksize):
    return tf.nn.avg_pool(x, ksize, strides=[1,1,1,1], padding='SAME')
    
# layer1: convolutional 96 filters of 5x5 with average pooling 96 filters of 5x5
W_conv1 = weight_variable([CONV1_FILTER_SIZE, CONV1_FILTER_SIZE, 1, N1])
b_conv1 = bias_variable([N1])
h_conv1 = tf.nn.relu(conv2d(input, W_conv1) + b_conv1)
h_pool1 = avgpool(h_conv1, ksize = [1, 5, 5, 1])

#layer2: convolution 256 filters of 2x2 with average pooling 256 filters of 2x2
W_conv2 = weight_variable([CONV2_FILTER_SIZE, CONV2_FILTER_SIZE, N1, N2])
b_conv2 = bias_variable([N2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = avgpool(h_conv2, ksize = [1, 2, 2, 1])

#layer3: fully-connected layer with 1024 neurons
AVGPOOL2_SIZE = ((INPUT_SIZE - CONV1_FILTER_SIZE + 1)/AVGPOOL1_FILTER_SIZE\
               - CONV2_FILTER_SIZE + 1)/AVGPOOL2_FILTER_SIZE
W_fc1 = weight_variable([AVGPOOL2_SIZE*AVGPOOL2_SIZE*N2, FC_SIZE])
b_fc1 = bias_variable([FC_SIZE])
h_fc1 = tf.nn.softmax(tf.matmul(h_pool2, W_fc1)+b_fc1)
keep_prob = tf.Variable(tf.float16)
h_dropped = tf.nn.dropout(h_fc1, keep_prob=DROPOUT_RATE, keep_prob=keep_prob)

#layer4: fully connected layer to readout
W_fc2 = weight_variable([FC_SIZE, OUTPUT_SIZE])
b_fc2 = bias_variable([OUTPUT_SIZE])
out = tf.nn.softmax(tf.matmul(h_dropped, W_fc2)+b_fc2)

# train variables
one_hot = tf.one_hot(indices=tf.cast(y_, tf.int16), depth = OUTPUT_SIZE)
loss = tf.losses.softmax_cross_entropy(one_hot, 
                                        logits = out,
                                        weights=1.0)
opt = tf.train.AdamOptimizer(learning_rate = 0.001)
train_op = opt.minimize(loss)

# evaluation
correct_prediction = tf.equal(tf.argmax(out), tf.argmax(y_))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float16))

def train_cnn(data):
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    
    for i in range(N_STEPS):
        batch = data.next_batch(BATCH_SIZE)
        if i % 500:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d accuracy %g" % (i, train_accuracy))
            train_op.run(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 0.5})
            saver.save(sess, 'checkpoints/detector.ckpt', global_step=i)
    print("test accuracy %g" % accuracy.eval(feed_dict={x:data.test.images,
                                                        y_:data.test.labels,
                                                        keep_prob: 1.0}))
    
def classify(image):
    """classifies to two classes: character or non-character."""
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, 'checkpoints/detector.ckpt')
    
    input = tf.reshape(image, [1, INPUT_SIZE*INPUT_SIZE])
    out.run(feed_dict = {x: input, keep_prob: 1.0})
    return tf.arg_max(out)
