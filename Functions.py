import numpy as np
import cv2
from config import *


def mark_text_bubbles(image):
    # Converting the image from BGR format to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # For white color
    max_white = np.array([0, 0, 255])
    min_white = np.array([0, 0, 230])
    final_mask = cv2.inRange(hsv_image, min_white, max_white)
    # Fetching the matching components in the original image
    result = cv2.bitwise_and(image, image, mask=final_mask)
    # Deriving Contours for the Image
    contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Copying the image and checking whether contours are matching.
    copy_img = image.copy()
    cv2.drawContours(copy_img, contours, -1, (0, 255, 0), 1)
    return copy_img


def