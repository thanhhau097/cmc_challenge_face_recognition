import cv2
import os


def get_and_save_embedding(image_name, image):
    save_image(image_name, image)

def save_image(image_name, image):
    base_path = "/Users/macos/Desktop" #/Hackathon/CMC/data/data/dataset/public_test/
    cv2.imwrite(os.path.join(base_path, image_name + '.png'))