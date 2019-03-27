import base64
import numpy as np
import cv2
import api.helpers.server.api_all as api_server
import imageio

def decode_base64(encoded_image):
    imgdata = base64.b64decode(encoded_image)
    nparr = np.fromstring(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_bbox(image):
    bb = api_server.crop_face(image)
    return bb

def get_result():
    vector = api_server.vectorizer("./api/helpers/server/image_112.png", "./api/helpers/server/image_160.png")
    list_res = api_server.get_same_person(vector, threshold=1.5)
    return list_res
