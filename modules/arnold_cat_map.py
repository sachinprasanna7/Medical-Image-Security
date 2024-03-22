#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# In[2]:


img = cv2.imread('assets/input_image_arnold.png', 0)  #read the image

# In[3]:


resized_img = cv2.resize(img, (256, 256))   #resize the image to 256x256

# In[4]:


plt.imshow(resized_img, cmap='gray', interpolation='bicubic')

# In[5]:


#show the pixel values of the resized image
resized_img

# In[6]:


# Show the maximum and minimum values of the greyscale image
max_value = np.max(resized_img)
min_value = np.min(resized_img)

print("Maximum Value:", max_value)
print("Minimum Value:", min_value)

# In[7]:


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

        # This code is to visualise the arnold cat transform after every 10 iterations
        # if(ctr % 10 == 0):
        #     plt.imshow(new_img, cmap='gray', interpolation='bicubic')
        #     plt.show()

    
    # This code is to visualise the final arnold cat transform. The recovered image is the same as the original image
        
    # plt.imshow(new_img, cmap='gray', interpolation='bicubic')
    # plt.show()

    return new_img

# In[10]:


# Check if command-line arguments are provided
if len(sys.argv) > 1:
    # Get the number of iterations from the command-line argument
    iterations = int(sys.argv[2])
else:
    # Set a default value if no command-line argument is provided
    iterations = 1

# Now you can use 'iterations' as needed in your script
resized_img = arnold_scramble(resized_img, iterations)
2
max_value = np.max(resized_img)
min_value = np.min(resized_img)

print("Maximum Value:", max_value)
print("Minimum Value:", min_value)

cv2.imwrite('generated_assets/arnold.png', resized_img)

# In[ ]:


iterations= 192-iterations
def inverse_arnold_scramble(img, iterations):
    rows, cols = img.shape
    new_img = np.zeros_like(img)

    for ctr in range(0, iterations):
        for i in range(0, rows):
            for j in range(0, cols):
                new_i = (i - j) % rows
                new_j = ((-i) + (2 * j)) % cols
                new_img[new_i, new_j] = img[i, j]

        img = new_img.copy()

    return new_img

# In[ ]:


inverse_img=inverse_arnold_scramble(img, iterations)

cv2.imwrite('generated_assets/inverse_arnold.png', inverse_img)
