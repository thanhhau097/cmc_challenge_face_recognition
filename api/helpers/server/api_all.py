from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras import backend as K
from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import api.helpers.server.facenet as facenet
import api.helpers.server.detect_face as detect_face
import random
from time import sleep
import imageio
from skimage import transform, img_as_ubyte, io

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
# ctx = mx.cpu()
from api import model_mxnet, model_facenet, list_vectors

# def get_model_mxnet(path):
#     sym, arg_params, aux_params = mx.model.load_checkpoint(path, 0)
#     all_layers = sym.get_internals()
#     sym = all_layers["fc1_output"]
#     mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
#     mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
#     mod.set_params(arg_params, aux_params, allow_missing=True)
#     return mod
#
# def get_model_facenet(model_path):
#     model = load_model(model_path)
#     return model

def get_feature_facenet(img):
    aligned_images = prewhiten(img)
    aligned_images = np.reshape(aligned_images, (1, 160, 160, 3))
    embs = model_facenet.predict(aligned_images)
    embs = l2_normalize(embs)
    return embs

def get_feature_mxnet(img):
    img = mx.nd.array(img)
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)
    img = img.astype('float32')
    model_mxnet.forward(Batch([img]))
    embedding = model_mxnet.get_outputs()[0].asnumpy()
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

def crop_face(img, gpu_memory_fraction=0.7, random_order=False, detect_multiple_faces=True, margin=32):
    sleep(random.random())
    if img.ndim<2:
        print('Unable to align')
        raise Exception("ndim < 2")
        
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:,:,0:3]

    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    # random_key = np.random.randint(0, high=99999)
    random_key = 0
    
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces>0:
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces>1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
        else:
            det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            cropped_to_img = img_as_ubyte(cropped)
            scaled_112 = transform.resize(cropped_to_img, (112, 112), mode='symmetric', preserve_range=True)
            scaled_160 = transform.resize(cropped_to_img, (160, 160), mode='symmetric', preserve_range=True)
            bounding_boxes_list = [bb[0], bb[1], bb[2], bb[3]]
            imageio.imsave("./api/helpers/server/image_112.png", scaled_112)
            imageio.imsave("./api/helpers/server/image_160.png", scaled_160)
    return bounding_boxes_list

def vectorizer(image_112_path, image_160_path):
    img_test_path_mxnet = imageio.imread(image_112_path)
    img_test_path_facenet = imageio.imread(image_160_path)
    db_emb = get_feature_mxnet(img_test_path_mxnet)
    db_emb = db_emb.reshape((1,512))
    db_emb_2 = get_feature_facenet(img_test_path_facenet)
    db_emb_2 = db_emb_2.reshape((1,128))
    db_emb = np.concatenate((db_emb, db_emb_2), axis=1)
    return db_emb

def get_same_person(find_emb, threshold):
    list_result = []
    dict_check = []
    for i in range(0, len(list_vectors)):
        dist = euclidean_distances(find_emb, list_vectors[i][2])
        name = list_vectors[i][0]
        name_arr = name.split("_")
        name = ""
        for j in range(0, len(name_arr) - 1):
            if(i != len(name_arr) - 2):
                name += name_arr[j]+"_"
            else:
                name += name_arr[j]+".png"
        if (dist < threshold):
            if name not in dict_check:
                list_result.append([name, [dist, list_vectors[i][1]]])
                dict_check.append(name)
            else:
                for k in range(0, len(list_result)):
                    if(list_result[k][0] == name):
                        list_result.pop(k)
                        break
    return list_result