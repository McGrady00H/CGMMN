#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# $File: cmmd-gen.py
# $Date: Mon Dec 07 18:4540 2015 +0800
# $Author: Yong Ren © <mails.tsinghua.edu.cn>
#

import os
import math
import sys
import random

import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nlinalg as NL
import theano.tensor.nnet as Tnn

from util import datapy,color,paramgraphics
from layer import nonlinearity
from layer import FullyConnected
from optimization import optimizer


def kernel_gram_for_x(x,y,n,d):
	'''
	Use a mixture of Gaussian kernel
	'''
	mn = 6
	bw = [1,10,20,40,80,160]
	zx = T.tile(x,(n,1))
	zy = T.reshape(T.tile(y,(1,n)),(n*n,d),ndim=2)
	zz = T.reshape(T.sum(T.sqr((zx - zy)),1),(n,n),ndim=2)
	res = T.zeros_like(zz)
	for i in range(mn):
		res = res + T.exp(-zz/(2*bw[i]))
	return res

def kernel_gram_for_y(x,y,n,d):
	'''
	Use a mixture of Gaussian kernel
	'''
	zx = T.tile(x,(n,1))
	zy = T.reshape(T.tile(y,(1,n)),(n*n,d),ndim=2)
	zz = T.reshape(T.sum(T.sqr((zx - zy)),1),(n,n),ndim=2)
	res = T.zeros_like(zz)
	res = res + T.gt(T.ones_like(zz),zz)
	return res

def gen_random_z(batch_size,hidden_dim):
	#samples = np.zeros((batch_size,hidden_dim),dtype=np.float32)
	#samples = np.cast['float32'](np.random.randint(2,size= (batch_size,hidden_dim)))
	samples = np.cast['float32']((2 * np.random.random((batch_size,hidden_dim))- 1)/2)
	return samples

def cmmd(dataset='mnist.pkl.gz',batch_size=100, layer_num = 3, hidden_dim = 5,seed = 0,layer_size=[64,256,256,512]):

	validation_frequency = 1
	test_frequency = 1
	pre_train = 1

	dim_input=(28,28)
	colorImg=False

	print "Loading data ......."
	#datasets = datapy.load_data_gpu_60000_with_noise(dataset, have_matrix = True)
	datasets = datapy.load_data_gpu_60000(dataset, have_matrix = True)
	train_set_x, train_set_y, train_y_matrix = datasets[0]
	valid_set_x, valid_set_y, valid_y_matrix = datasets[1]
	test_set_x, test_set_y, test_y_matrix = datasets[2]

	rng = np.random.RandomState(seed)                                                          
	rng_share = theano.tensor.shared_randomstreams.RandomStreams(0)

	n_train_batches = train_set_x.get_value().shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	aImage = paramgraphics.mat_to_img(train_set_x.get_value()[0:169].T,dim_input, colorImg=colorImg)
	aImage.save('mnist_sample','PNG')

	################################
	##        build model         ##
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
	layers.append(FullyConnected.FullyConnected(
			rng = rng,
			n_in = 10 + hidden_dim, 
			#n_in = 10,
			n_out = layer_size[0],
			activation = activation
	))
	layer_output.append(layers[-1].output_mix(input=[y_matrix,random_z]))
	#layer_output.append(layers[-1].output_mix2(input=[y_matrix,random_z]))
	#layer_output.append(layers[-1].output(input=x))
	#layer_output.append(layers[-1].output(input=random_z))

	#### middle layer
	for i in range(layer_num):
		layers.append(FullyConnected.FullyConnected(
			rng = rng,
			n_in = layer_size[i], 
			n_out = layer_size[i+1],
			activation = activation
		))
		layer_output.append(layers[-1].output(input= layer_output[-1]))

	#### last layer
	activation = Tnn.sigmoid
	#activation = nonlinearity.relu
	layers.append(FullyConnected.FullyConnected(
		rng = rng,
		n_in = layer_size[-1],
		n_out = 28*28,
		activation = activation
	))
	x_gen = layers[-1].output(input = layer_output[-1])
	
	lambda1_ = 100
	lambda_= theano.shared(np.asarray(lambda1_, dtype=np.float32))

	K_d = kernel_gram_for_y(y_matrix,y_matrix,batch_size,10)
	K_s = K_d 
	K_sd = K_d

	Invv_1 = T.sum(y_matrix,axis=0)/batch_size
	Invv = NL.alloc_diag(1/Invv_1)
	Inv_K_d = Invv
	#Inv_K_d = NL.matrix_inverse(K_d +lambda_ * T.identity_like(K_d))
	Inv_K_s = Inv_K_d

	L_d = kernel_gram_for_x(x,x,batch_size,28*28)
	L_s = kernel_gram_for_x(x_gen,x_gen,batch_size,28*28)
	L_ds = kernel_gram_for_x(x,x_gen,batch_size,28*28)
	

	'''
	cost = -(NL.trace(T.dot(T.dot(T.dot(K_d, Inv_K_d), L_d), Inv_K_d)) +\
			NL.trace(T.dot(T.dot(T.dot(K_s, Inv_K_s), L_s),Inv_K_s))- \
			2 * NL.trace(T.dot(T.dot(T.dot(K_sd, Inv_K_d) ,L_ds ), Inv_K_s)))
	'''

	'''
	cost = -(NL.trace(T.dot(L_d, T.ones_like(L_d) )) +\
			NL.trace(T.dot(L_s,T.ones_like(L_s)))- \
			2 * NL.trace(T.dot(L_ds,T.ones_like(L_ds) )))


	cost2 =  2 * T.sum(L_ds) - T.sum(L_s)  + NL.trace(T.dot(L_s, T.ones_like(L_s)))\
			- 2 * NL.trace( T.dot(L_ds , T.ones_like(L_ds)))
	cost2 = T.dot(T.dot(Inv_K_d, K_d),Inv_K_d)
	'''
	cost2 = K_d
	#cost2 = T.dot(T.dot(Inv_K_d,K_d),Inv_K_d)
	#cost =  - T.sum(L_d) +2 * T.sum(L_ds) - T.sum(L_s)
	cost2 = K_d
	cost2 = T.dot(T.dot(T.dot(y_matrix, Inv_K_d),Inv_K_d),y_matrix.T)

	cost = -(NL.trace(T.dot(T.dot(T.dot(T.dot(L_d, y_matrix),Inv_K_d), Inv_K_d),y_matrix.T)) +\
			NL.trace(T.dot(T.dot(T.dot(T.dot(L_s, y_matrix),Inv_K_s), Inv_K_s),y_matrix.T))- \
			2 * NL.trace(T.dot(T.dot(T.dot(T.dot(L_ds, y_matrix),Inv_K_d), Inv_K_s),y_matrix.T)))

	'''
	cost =  - T.sum(L_d) +2 * T.sum(L_ds) - T.sum(L_s)
	cost =  - NL.trace(K_s * Inv_K_s * L_s * Inv_K_s)+ \
			2 * NL.trace(K_sd * Inv_K_d * L_ds * Inv_K_s)
	'''

	################################
	##        updates             ##
	################################
	params = []
	for aLayer in layers:
		params += aLayer.params
	gparams = [T.grad(cost,param) for param in params]

	learning_rate = 3e-4
	weight_decay=1.0/n_train_batches
	epsilon=1e-8

	l_r = theano.shared(np.asarray(learning_rate, dtype=np.float32))
	get_optimizer = optimizer.get_adam_optimizer_max(learning_rate=l_r,
		decay1=0.1, decay2=0.001, weight_decay=weight_decay, epsilon=epsilon)
	updates = get_optimizer(params,gparams)


	################################
	##         pretrain model     ##
	################################
	parameters = theano.function(
			inputs = [],
			outputs = params,
			)

	gen_fig = theano.function(
			inputs = [y_matrix,random_z],
			outputs = x_gen,
			on_unused_input='warn',
	)

	if pre_train == 1:
		print "pre-training model....."
		pre_train = np.load('./result/MMD-100-5-64-256-256-512.npz')['model']
		for (para, pre) in zip(params, pre_train):
			para.set_value(pre)

		s = 8
		for jj in range(10):
			a = np.zeros((s,10),dtype=np.float32)
			for ii in range(s):
				kk = random.randint(0,9)
				a[ii,kk] = 1

			x_gen = gen_fig(a,gen_random_z(s,hidden_dim))

			ttt = train_set_x.get_value()
			for ll in range(s):
				minn = 1000000
				ss = 0
				for kk in range(ttt.shape[0]):
					tt =  np.linalg.norm(x_gen[ll] - ttt[kk])
					if tt < minn:
						minn = tt
						ss = kk
				#np.concatenate(x_gen,ttt[ss])
				x_gen = np.vstack((x_gen,ttt[ss]))
 
			aImage = paramgraphics.mat_to_img(x_gen.T,dim_input, colorImg=colorImg)
			aImage.save('samples_'+str(jj)+'_similar','PNG')

	################################
	##         prepare data       ##
	################################

	#### compute matrix inverse
	#print "Preparing data ...."
	#Invv = NL.matrix_inverse(K_d +lambda_ * T.identity_like(K_d))
	'''
	Invv_1 = T.sum(y_matrix,axis=0)/batch_size
	Invv = NL.alloc_diag(1/Invv_1)
	Inv_K_d = Invv

	prepare_data = theano.function(
			inputs = [index],
			outputs = [Invv,K_d],
			givens = {
				#x:train_set_x[index * batch_size:(index + 1) * batch_size],
				y_matrix:train_y_matrix[index * batch_size:(index + 1) * batch_size],
				}
			)

	Inv_K_d_l, K_d_l =  prepare_data(0)
	print Inv_K_d_l

	for minibatch_index in range(1, n_train_batches):
		if minibatch_index % 10 == 0:
			print 'minibatch_index:', minibatch_index
		Inv_pre_mini, K_d_pre_mini = prepare_data(minibatch_index)
		Inv_K_d_l = np.vstack((Inv_K_d_l,Inv_pre_mini))
		K_d_l = np.vstack((K_d_l,K_d_pre_mini))

	Inv_K_d_g = theano.shared(Inv_K_d_l,borrow=True)
	K_d_g = theano.shared(K_d_l, borrow=True)
	'''


	################################
	##         train model        ##
	################################



	train_model = theano.function(
		inputs = [index,random_z],
		outputs = [cost,x_gen,cost2],
		updates=updates,
		givens={
			x:train_set_x[index * batch_size:(index + 1) * batch_size],
			y:train_set_y[index * batch_size:(index + 1) * batch_size],
			y_matrix:train_y_matrix[index * batch_size:(index + 1) * batch_size],
			#K_d:K_d_g[index * batch_size:(index + 1) * batch_size],
			#Inv_K_d:Inv_K_d_g[index * batch_size:(index + 1) * batch_size],
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
			cost,x_gen,cost2 = train_model(minibatch_index,gen_random_z(batch_size,hidden_dim))
			print 'cost: ', cost
			print 'cost2: ', cost2
			if minibatch_index % 30 == 0:
				aImage = paramgraphics.mat_to_img(x_gen[0:1].T,dim_input, colorImg=colorImg)
				aImage.save('samples_epoch_'+str(cur_epoch)+'_mini_'+str(minibatch_index),'PNG')

		
		if cur_epoch %1 == 0:
			model = parameters()
			for i in range(len(model)):
				model[i] = np.asarray(model[i]).astype(np.float32)
			np.savez('model-'+str(cur_epoch),model=model)


if __name__ == '__main__':
	cmmd()

# vim: foldmethod= marker
