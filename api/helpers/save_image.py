import cv2
import os


def get_and_save_embedding(image_name, image):
    pass
    save_image(image_name, image)

def save_image(image_name, image):
    base_path = ""
    cv2.imwrite(os.path.join(base_path, image_name))