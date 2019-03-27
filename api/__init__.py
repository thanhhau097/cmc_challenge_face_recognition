from keras.models import load_model
import mxnet as mx
import pickle
ctx = mx.cpu()
import numpy as np

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

model_mxnet = get_model_mxnet("./api/helpers/models/model")
print("MODEL MXNET LOADED")

model_facenet = get_model_facenet("./api/helpers/keras-facenet/model/facenet_keras.h5")
model_facenet._make_predict_function()
print("MODEL FACENET LOADED")

list_vectors = pickle.load(open("./api/helpers/server/emb_vectors.pkl", 'rb'))
print("DATABASE LOADED")