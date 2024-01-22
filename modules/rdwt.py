import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2


def rdwt(img):

    # Performing the Redundant Discrete Wavelet Transform (RDWT) with Haar wavelet
    coeffs_org = pywt.swt2(img, 'haar', level=1, axes=(0, 1))

    # Extracting the coefficients at each level
    A1, (B1, C1, D1) = coeffs_org[0]

    return A1, B1, C1, D1