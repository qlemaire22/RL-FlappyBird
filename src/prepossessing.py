import cv2
import numpy as np


def transform_image_stacked(img, s_t):
    x_t1_colored = img[0:279, 0:410]
    x_t1 = cv2.cvtColor(cv2.resize(
        x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (80, 80, 1))
    s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
    return s_t1


def transform_image(img):
    x_t = img[0:279, 0:410]

    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    return s_t
