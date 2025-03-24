import cv2
import numpy as np
import matplotlib.pyplot as plt

image_name = "tree-2.png"

img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

filters = [
    {'name': 'Sharpening', 'kernel': np.array([[-1, -1, -1], 
                                                [-1, 8, -1], 
                                                [-1, -1, -1]])},
           
    {'name': 'Edge Detection 1', 'kernel': np.array([[-1, 0, 1], 
                                                    [0, 0, 0], 
                                                    [-1, 0, 1]])},

    {'name': 'Edge Detection 2', 'kernel': np.array([[0, 1, 0], 
                                                    [-1, 0, 1], 
                                                    [0, -1, 0]])},
]


#fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16), squeeze=False)

cv2.imshow('Original Image', img)
#axes[0, 0].set_title('Original Image')
#axes[0, 0].axis('off')

cv2.waitKey()

for i, f in enumerate(filters):
    filtered_img = cv2.filter2D(img, -1, f['kernel'])
    row = (i+1) // 2
    col = (i+1) % 2
    cv2.imshow(f['name'], filtered_img)
    cv2.waitKey()

cv2.destroyAllWindows()