import cv2
import numpy as np

def apply_log_transform(r, a):
    img_log = (np.log(r+1))*255

    s = np.array(img_log * a ,dtype=np.uint8)

    return s


def apply_power_transform(r, gamma, c):
    s = np.array(c*(r/255)**gamma,dtype=np.uint8)
    #s = np.array(c*r**gamma,dtype=np.uint8)
    return s


def apply_negative_transform(r):
    s = 255-r
    return s


image_name = "tree-2.png"

img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)


log_trans_img = apply_log_transform(img, 2)
power_trans_img = apply_power_transform(img, 1.2, 127)
neg_trans_img = apply_negative_transform(img)

cv2.imshow('Normal Output Window', img)
cv2.waitKey()

cv2.imshow('Log Output Window', log_trans_img)
cv2.waitKey()

cv2.imshow('Power Output Window', power_trans_img)
cv2.waitKey()

cv2.imshow('Negative Output Window', neg_trans_img)
cv2.waitKey()

cv2.destroyAllWindows()