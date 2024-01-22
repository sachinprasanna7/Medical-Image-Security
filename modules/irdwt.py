import numpy as np
import pywt


def irdwt(A1, B1, C1, D1, wavelet='haar'):
    # Combine components into a matrix
    combined_matrix = np.vstack([np.hstack([A1, B1]), np.hstack([C1, D1])])

    # Inverse wavelet transform along rows
    rows_inverse = pywt.idwt(combined_matrix[0, :], combined_matrix[1, :], wavelet)

    # Inverse wavelet transform along columns
    irdwt_result = pywt.idwt(rows_inverse[0, :], rows_inverse[1, :], wavelet)

    return irdwt_result