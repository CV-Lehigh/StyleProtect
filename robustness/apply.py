import os
from tqdm import tqdm 
import cv2
from PIL import Image

def blur_image(image_path, save_path):
    img = cv2.imread(image_path)
    blurred_img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.05)
    cv2.imwrite(save_path, blurred_img)


def jpeg_compress(image_path, save_path):
    img = Image.open(image_path)
    img.save(save_path, 'JPEG', quality=75)

