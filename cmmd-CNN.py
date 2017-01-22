#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: cmmd-CNN.py
# $Date: Fri Dec 04 16:5213 2015 +0800
# $Author: Yong Ren © <mails.tsinghua.edu.cn>
#

import os
import math
import sys

import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nlinalg as NL
import theano.tensor.nnet as Tnn

from util import datapy,color
from layer import nonlinearity
from layer import FullyConnected
from optimization import optimizer
from layer import ConvMaxPool

def kernel_gram_for_x(x,y,n,d):
	'''
	Use a mixture of Gaussian kernel
	'''
	mn = 5
	bw = [1,9,20,40,80]
	zx = T.tile(x,(n,1))
	zy = T.reshape(T.tile(y,(1,n)),(n*n,d),ndim=2)
	zz = T.reshape(T.sum(T.sqr((zx - zy)),1),(n,n),ndim=2)
	res = T.zeros_like(zz)
	for i in range(mn):
		res = res + T.exp(-zz/(2*bw[i]))
	return res


def kernel_gram(x,y,n,d):
	'''
	Use a mixture of Gaussian kernel
	'''
	mn = 5
	bw = [1,3,5,7,9]
	zx = T.tile(x,(n,1))
	zy = T.reshape(T.tile(y,(1,n)),(n*n,d),ndim=2)
	zz = T.reshape(T.sum(T.sqr((zx - zy)),1),(n,n),ndim=2)
	res = T.zeros_like(zz)
	for i in range(mn):
		res = res + T.exp(-zz/(2*bw[i]))
	return res

def gen_random_z(batch_size,hidden_dim):
	#samples = np.zeros((batch_size,hidden_dim),dtype=np.float32)
	samples = np.cast['float32'](np.random.randint(2,size= (batch_size,hidden_dim)))
	return samples

def cmmd(dataset='mnist.pkl.gz',batch_size=200, layer_num = 2, hidden_dim = 20,seed = 0,layer_size=[500,200,100]):

	validation_frequency = 1
	test_frequency = 1
	pre_train = 0
	pre_train_epoch = 30

	print "Loading data ......."
	datasets = datapy.load_data_gpu_60000(dataset, have_matrix = True)
	train_set_x, train_set_y, train_y_matrix = datasets[0]
	valid_set_x, valid_set_y, valid_y_matrix = datasets[1]
	test_set_x, test_set_y, test_y_matrix = datasets[2]

	n_train_batches = train_set_x.get_value().shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	rng = np.random.RandomState(seed)														  
	rng_share = theano.tensor.shared_randomstreams.RandomStreams(0)

	################################
	##		build model		 ##
	################################
	print "Building model ......."

	index = T.lscalar()
	x = T.matrix('x')  ##### batch_size * 28^2
	y = T.vector('y') 
	y_matrix = T.matrix('y_matrix') 
	random_z = T.matrix('random_z') ### batch_size * hidden_dim
	Inv_K_d = T.matrix('Inv_K_d')

	layers = []
	layer_output= []

	activation = nonlinearity.relu
	#activation = Tnn.sigmoid
	#### first layer
	'''
	layers.append(FullyConnected.FullyConnected(
			rng = rng,
			n_in = 28*28 + hidden_dim, 
			#n_in = 28*28, 
			n_out = layer_size[0],
			activation = activation
	))
	layer_output.append(layers[-1].output_mix(input=[x,random_z]))
	'''
	layers.append(ConvMaxPool.ConvMaxPool(
		rng = rng,
		image_shape=(batch_size,1,28,28),
		filter_shape = (32,1,5,5),
		poolsize = (2,2),
		border_mode = 'valid',
		activation = activation
	))
	layer_output.append(layers[-1].output(input=x.reshape((batch_size,1,28,28))))


	layers.append(ConvMaxPool.ConvMaxPool(
		rng,
		image_shape=(batch_size, 32, 12, 12),
		filter_shape=(32, 32, 3, 3),
		poolsize=(1, 1),
		border_mode='same',
		activation=activation
	))
	layer_output.append(layers[-1].output(input= layer_output[-1]))

	layers.append(ConvMaxPool.ConvMaxPool(
		rng,
		image_shape=(batch_size, 32, 12, 12),
		filter_shape=(64, 32, 3, 3),
		poolsize=(2, 2),
		border_mode='valid',
		activation=activation
	))
	layer_output.append(layers[-1].output(input= layer_output[-1]))

	layers.append(ConvMaxPool.ConvMaxPool(
		rng,
		image_shape=(batch_size, 64, 5, 5),
		filter_shape=(64, 64, 3, 3),
		poolsize=(1, 1),
		border_mode='same',
		activation=activation
	))
	layer_output.append(layers[-1].output(input= layer_output[-1]))

	layers.append(ConvMaxPool.ConvMaxPool(
		rng,
		image_shape=(batch_size, 64, 5, 5),
		filter_shape=(64, 64, 3, 3),
		poolsize=(1, 1),
		border_mode='same',
		activation=activation
	))
	layer_output.append(layers[-1].output(input= layer_output[-1]).flatten(2))

	'''
	#### middle layer
	for i in range(layer_num):
		layers.append(FullyConnected.FullyConnected(
			rng = rng,
			n_in = layer_size[i], 
			n_out = layer_size[i+1],
			activation = activation
		))
		layer_output.append(layers[-1].output(input= layer_output[-1]))
	'''

	#### last layer
	activation = Tnn.sigmoid
	layers.append(FullyConnected.FullyConnected(
		rng = rng,
		#n_in = layer_size[-1],
		n_in = 5 * 5 * 64,
		n_out = 10,
		activation = activation
	))
	y_gen = layers[-1].output(input = layer_output[-1])
	
	lambda1_ = 1e-3
	lambda_= theano.shared(np.asarray(lambda1_, dtype=np.float32))


	K_d = kernel_gram_for_x(x,x,batch_size,28*28)
	K_s = K_d 
	K_sd = K_d
	#Inv_K_d = NL.matrix_inverse(K_d +lambda_ * T.identity_like(K_d))
	Inv_K_s = Inv_K_d

	L_d = kernel_gram(y_matrix,y_matrix,batch_size,10)
	L_s = kernel_gram(y_gen,y_gen,batch_size,10)
	L_ds = kernel_gram(y_matrix,y_gen,batch_size,10)
	

	cost = -(NL.trace(T.dot(T.dot(T.dot(K_d, Inv_K_d), L_d), Inv_K_d)) +\
		 NL.trace(T.dot(T.dot(T.dot(K_s, Inv_K_s), L_s),Inv_K_s))- \
		 2 * NL.trace(T.dot(T.dot(T.dot(K_sd, Inv_K_d) ,L_ds ), Inv_K_s)))

	cost_pre = -T.sum(T.sqr(y_matrix - y_gen))


	cc = T.argmax(y_gen,axis=1)
	correct = T.sum(T.eq(T.cast(T.argmax(y_gen,axis=1),'int32'),T.cast(y,'int32')))

	################################
	##		updates			 ##
	################################
	params = []
	for aLayer in layers:
		params += aLayer.params
	gparams = [T.grad(cost,param) for param in params]
	gparams_pre = [T.grad(cost_pre,param) for param in params]

	learning_rate = 3e-4
	weight_decay=1.0/n_train_batches
	epsilon=1e-8

	l_r = theano.shared(np.asarray(learning_rate, dtype=np.float32))
	get_optimizer = optimizer.get_adam_optimizer_max(learning_rate=l_r,
		decay1=0.1, decay2=0.001, weight_decay=weight_decay, epsilon=epsilon)
	updates = get_optimizer(params,gparams)

	updates_pre = get_optimizer(params,gparams_pre)


	################################
	##		 pretrain model	 ##
	################################
	parameters = theano.function(
			inputs = [],
			outputs = params,
			)

	'''
	pre_train_model = theano.function(
		inputs = [index,random_z],
		outputs = [cost_pre, correct],
		updates=updates_pre,
		givens={
			x:train_set_x[index * batch_size:(index + 1) * batch_size],
			y:train_set_y[index * batch_size:(index + 1) * batch_size],
			y_matrix:train_y_matrix[index * batch_size:(index + 1) * batch_size],
		},
		on_unused_input='warn'
		)
	cur_epoch = 0
	if pre_train == 1:
		for cur_epoch in range(pre_train_epoch):
			print 'cur_epoch: ', cur_epoch,
			cor = 0 
			for minibatch_index in range(n_train_batches):
				cost_pre_mini,correct_pre_mini = pre_train_model(minibatch_index,gen_random_z(batch_size,hidden_dim))
				cor = cor + correct_pre_mini
			print 'correct number: ' , cor
		#np.savez(,model = model)
		'''

	if pre_train == 1:
		print "pre-training model....."
		pre_train = np.load('model.npz')['model']
		for (para, pre) in zip(params, pre_train):
			para.set_value(pre)

	################################
	##		 prepare data	   ##
	################################

	#### compute matrix inverse
	print "Preparing data ...."
	Invv = NL.matrix_inverse(K_d +lambda_ * T.identity_like(K_d))
	prepare_data = theano.function(
			inputs = [index],
			outputs = [Invv,K_d],
			givens = {
				x:train_set_x[index * batch_size:(index + 1) * batch_size],
				}
			)

	Inv_K_d_l, K_d_l =  prepare_data(0)

	for minibatch_index in range(1, n_train_batches):
		if minibatch_index % 10 == 0:
			print 'minibatch_index:', minibatch_index
		Inv_pre_mini, K_d_pre_mini = prepare_data(minibatch_index)
		Inv_K_d_l = np.vstack((Inv_K_d_l,Inv_pre_mini))
		K_d_l = np.vstack((K_d_l,K_d_pre_mini))

	Inv_K_d_g = theano.shared(Inv_K_d_l,borrow=True)
	K_d_g = theano.shared(K_d_l, borrow=True)


	################################
	##		 train model		##
	################################

	train_model = theano.function(
		inputs = [index,random_z],
		outputs = [correct,cost,y,cc,y_gen],
		updates=updates,
		givens={
			x:train_set_x[index * batch_size:(index + 1) * batch_size],
			y:train_set_y[index * batch_size:(index + 1) * batch_size],
			y_matrix:train_y_matrix[index * batch_size:(index + 1) * batch_size],
			#K_d:K_d_g[index * batch_size:(index + 1) * batch_size],
			Inv_K_d:Inv_K_d_g[index * batch_size:(index + 1) * batch_size],
		},
		on_unused_input='warn'
	)

	valid_model = theano.function(
		inputs = [index,random_z],
		outputs = correct,
		#updates=updates,
		givens={
			x:valid_set_x[index * batch_size:(index + 1) * batch_size],
			y:valid_set_y[index * batch_size:(index + 1) * batch_size],
			y_matrix:valid_y_matrix[index * batch_size:(index + 1) * batch_size],
		},
		on_unused_input='warn'
	)

	test_model = theano.function(
		inputs = [index,random_z],
		outputs = [correct,y_gen],
		#updates=updates,
		givens={
			x:test_set_x[index * batch_size:(index + 1) * batch_size],
			y:test_set_y[index * batch_size:(index + 1) * batch_size],
			y_matrix:test_y_matrix[index * batch_size:(index + 1) * batch_size],
		},
		on_unused_input='warn'
	)

	n_epochs = 500
	cur_epoch = 0



	print "Training model ......"

	while (cur_epoch < n_epochs) :
		cur_epoch = cur_epoch + 1
		cor = 0
		for minibatch_index in xrange(n_train_batches):
			print minibatch_index,
			print " : ",
			correct,cost,a,b,y_gen = train_model(minibatch_index,gen_random_z(batch_size,hidden_dim))
			cor = cor + correct
			print correct
			print b
			print y_gen
		with open('log.txt','a') as f:
			print >>f , "epoch: " , cur_epoch, "training_correct: " , cor

		if cur_epoch % validation_frequency == 0:
			cor2 = 0
			for minibatch_index in xrange(n_valid_batches):
				correct = valid_model(minibatch_index,gen_random_z(batch_size,hidden_dim))
				cor2 = cor2 + correct
			with open('log.txt','a') as f:
				print >>f , "	validation_correct: " , cor2

		if cur_epoch % test_frequency == 0:
			cor2 = 0
			for minibatch_index in xrange(n_test_batches):
				correct,y_gen = test_model(minibatch_index,gen_random_z(batch_size,hidden_dim))
				with open('log.txt','a') as f:
					for index in range(batch_size):
						if not np.argmax(y_gen[index]) == test_set_y[minibatch_index * batch_size + index]:
							print >>f , "index: " , minibatch_index * batch_size + index, 'true Y: ', test_set_y[minibatch_index * batch_size + index]
							print >>f , 'gen_y: ' , y_gen[index]

				cor2 = cor2 + correct
			with open('log.txt','a') as f:
				print >>f , "	test_correct: " , cor2
		
		if epoch %1 == 0:
			model = parameters()
			for i in range(len(model)):
				model[i] = np.asarray(model[i]).astype(np.float32)
			np.savez('model-'+str(epoch),model=model)


if __name__ == '__main__':
	cmmd()

# vim: foldmethod= marker
