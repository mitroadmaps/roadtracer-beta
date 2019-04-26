import numpy
import tensorflow as tf
import os
import os.path
import random
import math
import time
from PIL import Image

BATCH_SIZE = 4
KERNEL_SIZE = 3

class Model:
	def _conv_layer(self, name, input_var, stride, in_channels, out_channels, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		padding = options.get('padding', 'SAME')
		batchnorm = options.get('batchnorm', False)
		transpose = options.get('transpose', False)

		with tf.variable_scope(name) as scope:
			if not transpose:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, in_channels, out_channels]
			else:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, out_channels, in_channels]
			kernel = tf.get_variable(
				'weights',
				shape=filter_shape,
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / KERNEL_SIZE / KERNEL_SIZE / in_channels)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[out_channels],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			if not transpose:
				output = tf.nn.bias_add(
					tf.nn.conv2d(
						input_var,
						kernel,
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			else:
				batch = tf.shape(input_var)[0]
				side = tf.shape(input_var)[1]
				output = tf.nn.bias_add(
					tf.nn.conv2d_transpose(
						input_var,
						kernel,
						[batch, side * stride, side * stride, out_channels],
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def _fc_layer(self, name, input_var, input_size, output_size, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		batchnorm = options.get('batchnorm', False)

		with tf.variable_scope(name) as scope:
			weights = tf.get_variable(
				'weights',
				shape=[input_size, output_size],
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / input_size)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[output_size],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			output = tf.matmul(input_var, weights) + biases
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def __init__(self, bn=False, angle_weight=10):
		tf.reset_default_graph()

		self.is_training = tf.placeholder(tf.bool)
		self.inputs = tf.placeholder(tf.float32, [None, 256, 256, 3])
		self.angle_targets = tf.placeholder(tf.float32, [None, 64])
		self.detect_targets = tf.placeholder(tf.float32, [None, 64, 64, 1])
		self.learning_rate = tf.placeholder(tf.float32)
		self.angle_onehot = tf.placeholder(tf.float32, [64, 64, 64])

		batch_size = tf.shape(self.inputs)[0]
		self.angle_onehot_tiled = tf.tile(tf.expand_dims(self.angle_onehot, axis=0), [batch_size, 1, 1, 1])

		# layers
		self.layer1 = self._conv_layer('layer1', self.inputs, 2, 3, 128, {'batchnorm': False}) # -> 128x128x128
		self.layer2 = self._conv_layer('layer2', self.layer1, 1, 128, 128, {'batchnorm': bn}) # -> 128x128x128
		self.layer3 = self._conv_layer('layer3', self.layer2, 2, 128, 256, {'batchnorm': bn}) # -> 64x64x256
		self.layer4 = self._conv_layer('layer4', self.layer3, 1, 256, 256, {'batchnorm': bn}) # -> 64x64x256
		self.layer5 = self._conv_layer('layer5', tf.concat([self.layer4, self.angle_onehot_tiled], axis=3), 1, 256+64, 256, {'batchnorm': bn}) # -> 64x64x256
		self.layer6 = self._conv_layer('layer6', self.layer5, 1, 256, 256, {'batchnorm': bn}) # -> 64x64x256
		self.layer7 = self._conv_layer('layer7', self.layer6, 2, 256, 512, {'batchnorm': bn}) # -> 32x32x512
		self.layer8 = self._conv_layer('layer8', self.layer7, 1, 512, 512, {'batchnorm': bn}) # -> 32x32x512
		self.layer9 = self._conv_layer('layer9', self.layer8, 2, 512, 512, {'batchnorm': bn}) # -> 16x16x512
		self.layer10 = self._conv_layer('layer10', self.layer9, 1, 512, 512, {'batchnorm': bn}) # -> 16x16x512
		self.layer11 = self._conv_layer('layer11', self.layer10, 2, 512, 512, {'batchnorm': bn}) # -> 8x8x512
		self.layer12 = self._conv_layer('layer12', self.layer11, 1, 512, 512, {'batchnorm': bn}) # -> 8x8x512
		self.layer13 = self._conv_layer('layer13', self.layer12, 1, 512, 512, {'batchnorm': bn}) # -> 8x8x512
		self.layer14 = self._conv_layer('layer14', self.layer13, 2, 512, 512, {'batchnorm': bn}) # -> 4x4x512
		self.layer15 = self._conv_layer('layer15', self.layer14, 1, 512, 512, {'batchnorm': bn}) # -> 4x4x512
		self.layer16 = self._conv_layer('layer16', self.layer15, 1, 512, 512, {'batchnorm': bn}) # -> 4x4x512
		self.layer17 = self._conv_layer('layer17', self.layer16, 2, 512, 512, {'batchnorm': bn}) # -> 2x2x512

		self.angle_outputs = self._conv_layer('angle_outputs', self.layer17, 2, 512, 64, {'activation': 'sigmoid', 'batchnorm': False})[:, 0, 0, :] # -> 64
		self.detect_pre_outputs = self._conv_layer('detect_pre_outputs', self.layer6, 1, 256, 2, {'batchnorm': False}) # -> 64x64x2
		self.detect_outputs = tf.nn.softmax(self.detect_pre_outputs)[:, :, :, 0:1]

		self.angle_loss = tf.reduce_mean(tf.square(self.angle_targets - self.angle_outputs))
		self.detect_labels = tf.concat([self.detect_targets, 1 - self.detect_targets], axis=3)
		self.detect_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.detect_labels, logits=self.detect_pre_outputs))
		#self.detect_loss = tf.reduce_mean(tf.square(self.detect_targets - self.detect_outputs))
		self.loss = self.angle_loss * angle_weight + self.detect_loss

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None)
