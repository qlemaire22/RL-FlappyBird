import tensorflow as tf
from config import *

class Network():
    def __init__(self, img_size):
        self.a = tf.placeholder("float", [None, ACTIONS])
        self.y = tf.placeholder("float", [None])

        # network weights
        W_conv1 = weight_variable([8, 8, 4, 32])
        b_conv1 = bias_variable([32])

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])

        W_fc2 = weight_variable([512, ACTIONS])
        b_fc2 = bias_variable([ACTIONS])

        # input layer
        self.s = tf.placeholder("float", [None, img_size, img_size, 4])

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(self.s, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        _, a, b, c = h_conv3.shape

        size = int(a*b*c)

        W_fc1 = weight_variable([size, 512])
        b_fc1 = bias_variable([512])

        h_conv3_flat = tf.reshape(h_conv3, [-1, size])

        self.h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        self.readout = tf.matmul(self.h_fc1, W_fc2) + b_fc2


        self.readout_action = tf.reduce_sum(
            tf.multiply(self.readout, self.a), reduction_indices=1)

    def cost(self):
        return tf.reduce_mean(tf.square(self.y - self.readout_action))



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
