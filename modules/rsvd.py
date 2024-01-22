import numpy as np
import matplotlib.pyplot as plt
import cv2

def rsvd(A, r):

    # Finding the dimensions of A
    m, n = A.shape

    # Finding the random projection matrix
    P = np.random.randn(n, r)

    # Finding the projection of A onto the random subspace
    Z = A @ P

    # Finding the QR decomposition of Z
    Q, _ = np.linalg.qr(Z, mode='reduced')

    # Finding the B matrix, which is the projection of A onto Q
    B = Q.T @ A

    # Finding the SVD of B
    Uhat, S, V = np.linalg.svd(B, full_matrices=False)

    # Making the U matrix
    U = Q @ Uhat

    return U, S, V