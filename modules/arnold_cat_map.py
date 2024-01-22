import numpy as np
import matplotlib.pyplot as plt

def arnold_scramble (img, iterations):
    rows, cols = img.shape
    new_img = np.zeros_like(img)

    for ctr in range (0, iterations):
        for i in range (0, rows):
            for j in range (0, cols):

                new_i = (i + j) % rows
                new_j = (i + (2*j)) % cols
                new_img[new_i, new_j] = img[i, j]
        
        img = new_img.copy()
        
    plt.imshow(new_img, cmap='gray', interpolation='bicubic')
    plt.show()

    return new_img