import os
import sys
# import detect_face
from  api.helpers.server.api_detect import main as d_main
from api.helpers.server.api_out import main as o_main
import timeit
import numpy as np
import os

CONTAINER_DIR = "data/CONTAINER_2/QH_API/"
RAW_DIR = "../data/RAW/QH/"
output_dir_112 = os.path.join(CONTAINER_DIR, "detected_112")
output_dir_160 = os.path.join(CONTAINER_DIR, "detected_160")
input_dir = RAW_DIR
gpu_memory_fraction = 0.7
detect_multiple_faces = True
margin = 32
random_order = False
test_image_path = os.path.join(RAW_DIR, "000_quang_hai.jpg")


def data_vector():
    print("Start detect 112 for mxnet")
    start_detect = timeit.default_timer()
    extracted_dir, bounding_boxes_list_mxnet = d_main(output_dir=output_dir_112,
                                    input_dir=input_dir,
                                    gpu_memory_fraction=gpu_memory_fraction,
                                    random_order=random_order,
                                    detect_multiple_faces=detect_multiple_faces,
                                    margin=margin,
                                    image_size=112,
                                    test_image_path=test_image_path)
    stop_detect = timeit.default_timer()
    detect_time = stop_detect - start_detect
    print("Time Detect 112 for mxnet: {}".format(detect_time))

    print("Start detect 160 for facenet")
    start_detect = timeit.default_timer()
    extracted_dir, bounding_boxes_list_facenet = d_main(output_dir=output_dir_160,
                                    input_dir=input_dir,
                                    gpu_memory_fraction=gpu_memory_fraction,
                                    random_order=random_order,
                                    detect_multiple_faces=detect_multiple_faces,
                                    margin=margin,
                                    image_size=160,
                                    test_image_path=test_image_path)
    stop_detect = timeit.default_timer()
    detect_time = stop_detect - start_detect
    print("Time Detect 160 for facenet: {}".format(detect_time))

    print("Start recog")
    model_facenet_path = "../keras-facenet/model/facenet_keras.h5"
    model_mxnet_path = "../models/model"
    base_data_dir = os.path.join(RAW_DIR, "quang_hai")
    result_path = "output.csv"
    start_recog = timeit.default_timer()
    list_arr = o_main(model_path_mxnet=model_mxnet_path,
            bounding_boxes_list=bounding_boxes_list_mxnet,
            result_path=result_path, 
        threshold=1.55,
        model2 = model_facenet_path)
    stop_recog = timeit.default_timer()
    recog_time = stop_recog - start_recog
    print("Time Recog: {}".format(recog_time))
