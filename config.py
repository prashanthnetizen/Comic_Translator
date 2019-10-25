import numpy as np
import cv2
import configparser

config = configparser.ConfigParser()


def show_image(window_name, image):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)