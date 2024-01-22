import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from modules.rdwt import rdwt
from modules.rsvd import rsvd


def algorithm_1(path1, path2):

    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)

    A1, B1, C1, D1 = rdwt(img1)
    A2, B2, C2, D2 = rdwt(img2)


    Uh, Sh, Vh = rsvd(D1, 200)  # target rank is 200
    Uw, Sw, Vw = rsvd(D2, 200)  # target rank is 200

    covariance_matrix = np.cov(Sh, Sw)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sorting the eigenvalues in descending order and the eigenvectors will rearrange accordingly.
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    diag = np.diag(eigenvalues)

    if(diag[0][0] > diag[1][1]):
        PC1 = eigenvectors[0][0]/(eigenvectors[0][0] + eigenvectors[1][0])
        PC2 = eigenvectors[1][0]/(eigenvectors[0][0] + eigenvectors[1][0])

    else:
        PC1 = eigenvectors[0][1]/(eigenvectors[0][1] + eigenvectors[1][1])
        PC2 = eigenvectors[1][1]/(eigenvectors[0][1] + eigenvectors[1][1])

    return PC1, PC2