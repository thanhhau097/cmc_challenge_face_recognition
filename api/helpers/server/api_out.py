"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet_temp as facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import cv2
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
import mxnet as mx
import imageio
from keras.models import load_model


from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
ctx = mx.gpu()

def get_model_mxnet(path):
    sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)
    all_layers = sym.get_internals()
    sym = all_layers["fc1_output"]
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod

def get_model_facenet(model_path):
    model = load_model(model_path)
    return model

def get_feature_facenet(model, img):
    aligned_images = prewhiten(img)
    aligned_images = np.reshape(aligned_images, (1, 160, 160, 3))
    embs = l2_normalize(model.predict(aligned_images))
    # print(embs.shape)
    return embs

def get_feature_mxnet(model, img):
    img = mx.nd.array(img)
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)
    img = img.astype('float32')
    model.forward(Batch([img]))
    embedding = model.get_outputs()[0].asnumpy()
    embedding = preprocessing.normalize(embedding).flatten()
    return embedding


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def main(model_path_mxnet, bounding_boxes_list, result_path, threshold, model2=None):
	np.random.seed(seed=666)
	check_pp1 = 0
	check_pp2 = 0

	list_embs_arr = []
	# Load the model

	print('Loading feature extraction model')
	model = get_model_mxnet(model_path_mxnet)
	if (model2 != None):
		model2 = get_model_facenet(model2)

	print("Emb test image")
	# first element: image test
	image_160_path = "data/CONTAINER_2/QH_API/detected_160/quang_hai/"
	image_112_path = "data/CONTAINER_2/QH_API/detected_112/quang_hai/"
	image_160_test_path = "data/CONTAINER_2/QH_API/detected_160/"
	image_112_test_path = "data/CONTAINER_2/QH_API/detected_112/"
	test_img_name = bounding_boxes_list[0][1][0].split("/")
	test_img_name = test_img_name[len(test_img_name) - 1]
	test_img_mxnet = imageio.imread(image_112_test_path+test_img_name)
	test_img_facenet = imageio.imread(image_160_test_path+test_img_name)
	test_img_emb = get_feature_mxnet(model, test_img_mxnet)
	test_img_emb = test_img_emb.reshape((1, 512))
	if(model2):
		test_img_emb_2 = get_feature_facenet(model2, test_img_facenet)
		test_img_emb_2 = test_img_emb_2.reshape((1,128))
		test_img_emb = np.concatenate((test_img_emb, test_img_emb_2), axis=1)
		if(check_pp1 == 0):
			print("test_img_emb : {}".format(test_img_emb.shape))
			check_pp1 += 1
	# Embed detected image
	print("Emb db image")
	count_total = 0
	with open(result_path, 'w') as f_output:
		f_output.write('image,x1,y1,x2,y2,result\n')
		for i in range(1, len(bounding_boxes_list)):
			count_per_raw_image = 0
			top, left, bottom, right = 0, 0, 0, 0
			for j in range(1, len(bounding_boxes_list[i])):
				test_path = bounding_boxes_list[i][j][0]
				test_path = test_path.split("/")
				test_path = test_path[len(test_path) - 1]
				img_test_path_mxnet = imageio.imread(image_112_path + test_path)
				img_test_path_facenet = imageio.imread(image_160_path + test_path)
				db_emb = get_feature_mxnet(model, img_test_path_mxnet)
				db_emb = db_emb.reshape((1,512))
				if(model2):
					db_emb_2 = get_feature_facenet(model2, img_test_path_facenet)
					db_emb_2 = db_emb_2.reshape((1,128))
					db_emb = np.concatenate((db_emb, db_emb_2), axis=1)
					top1, left1, bottom1, right1 = bounding_boxes_list[i][j][1],\
												bounding_boxes_list[i][j][2],\
												bounding_boxes_list[i][j][3],\
												bounding_boxes_list[i][j][4]
					list_embs_arr.append([test_path, [top1, left1, bottom1, right1], db_emb])
				if(check_pp2 == 0):
					print("db_emb : {}".format(db_emb.shape))
					check_pp2 += 1 
				dist = euclidean_distances(test_img_emb, db_emb)

				thr = threshold
				if dist[0][0] < thr:
					count_per_raw_image += 1
					top, left, bottom, right = bounding_boxes_list[i][j][1],\
												bounding_boxes_list[i][j][2],\
												bounding_boxes_list[i][j][3],\
												bounding_boxes_list[i][j][4]

			if count_per_raw_image == 1:
				count_total += 1
				result_temp = str(bounding_boxes_list[i][0]) + "," + str(top) + "," + \
								str(left) + "," + str(bottom) + "," + str(right) + "," + str(1)
				f_output.write(result_temp)
				f_output.write("\n")
			else:
				result_temp = str(bounding_boxes_list[i][0]) + "," + str(0) + "," + \
								str(0) + "," + str(0) + "," + str(0) + "," + str(0)
				f_output.write(result_temp)
				f_output.write("\n")
	f_output.close()

	print("Total: %d" % count_total)
	return list_embs_arr