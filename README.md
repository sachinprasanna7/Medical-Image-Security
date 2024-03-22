# Medical-Image-Security

The following repository is an implementation of the paper titled 'SecDH: Security of COVID-19 Images Based on Data Hiding with PCA' by Singh, O. P., et al., published in Computer Communications (2022, Volume 191, Pages 368-377). It was implemented as a part to fulfill the project requirement of the Information Assurance and Security (IT352) course.

# Introduction
The paper introduces a data hiding scheme, guaranteeing the security of COVID-19 images. The scheme involves a seamless combination of RDWT-RSVD for imperceptible logo marking within the carrier media, effectively addressing the False Positive Problem (FPP). The repository showcases the implementation where the principal component of the mark image is concealed in the host image. Furthermore, Principal Component Analysis (PCA) is utilized to determine the normalized principal component for embedding purposes. The implementation aligns with the objectives outlined in the research paper and provides a practical demonstration of the proposed security measures.

# Algorithms Implemented

## Normalisation

Image normalization is a procedure to transform the image into standard image such that it contains detailed information of input image. It provides better resistance against the geometric attacks. The detailed steps of image normalization procedure are mentioned below:

1. Finding Moments
2. Translation
3. Shearing
4. Scaling

## Randomised Singular Value Decomposition (RSVD)

RSVD plays an important role to perform matrix factorization in image processing. This is because the computational cost of RSVD is less than SVD. The time complexity of RSVD is 𝑂(𝑖𝑗𝑟), where size of input matrix is denoted as 𝑖 × 𝑗 and rank of this matrix is termed as r.


## Redundant Discrete Wavelet Transform (RDWT)

The Redundant Discrete Wavelet Transform (RDWT) for images is a variant of the discrete wavelet transform (DWT) that incorporates redundancy by allowing overlapping wavelet coefficients, offering improved representation accuracy and flexibility for tasks like image denoising and compression. This redundancy enables more robust signal representation and better preservation of image features compared to traditional DWT.

The decomposition typically results in four subbands:

**Approximation (LL)**: Contains the low-frequency information and represents the coarse approximation of the original image.

**Horizontal (HL)**: Contains the high-frequency information in the horizontal direction, capturing details and edges that are primarily horizontal.

**Vertical (LH)**: Contains the high-frequency information in the vertical direction, capturing details and edges that are primarily vertical.

**Diagonal (HH)**: Contains the high-frequency information in both horizontal and vertical directions, capturing diagonal edges and texture details.

All the four subbands are plotted in the source file.

## Principal Component Analysis

Principal Component Analysis (PCA) is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional space while preserving the most important information. It identifies orthogonal axes, called principal components, along which the data exhibits the maximum variance, aiding in visualization, compression, and noise reduction.


## Arnold Cat Maps

Arnold cat map is one of the most popular scrambling methods, which was first introduced by V. I. Arnold. It contains several properties in terms of simplicity and periodicity. The scrambling operation is performed by altering the pixel values of mark image. Besides the recovery of mark image is possible only if the secret key is used.

## Algorithm 1: Determination of Principal Cofficients

The simplified concept of PCA based fusion is shown is implemented. It is utilized to compute the normalized principal component as a factor for embedding the mark information into the host media. First, covariance is calculated between host and mark image and then normalized principal coefficients are obtained with the help of Eigenvalues.

## Algorithm 2: Watermark Embedding

Image normalization procedure is utilized to transform the original image, ‘𝑂𝑟𝑔𝑖𝑚𝑔 ’ into normalized image, ‘𝑁𝑜𝑟𝑚𝑖𝑚𝑔’. Further, '𝑁𝑜𝑟𝑚𝑖𝑚𝑔' and mark image, ‘𝑊𝑎𝑡𝑖𝑚𝑔' are transformed using RDWT and RSVD respectively. Furthermore, PCA fusion is employed to compute the optimal embedding factor for embedding purpose. The principal component of mark image is concealed inside cover image. Furthermore, inverse operation of RSVD and RDWT is performed to compute the marked image, ‘𝑀𝑎𝑟𝑘𝑖𝑚𝑔 ’. Lastly, Arnold cat map is performed on 𝑀𝑎𝑟𝑘𝑖𝑚𝑔 enhance the additional security of proposed scheme.

## Algorithm 3: Recovery procedure of the mark

It is the just inverse procedure of embedding scheme. First, inverse Arnold cat map is performed to obtain decrypted marked image, ‘𝐷𝑒𝑐_𝑀𝑎𝑟𝑘𝑖𝑚𝑔 ’ and then it is transformed using RDWT and RSVD. Further, singular value is extracted using PCA fusion. Lastly, the mark image is recovered by applying the inverse of RSVD and IDWT respectively.

# Performance Metrics

## PSNR

PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher PSNR values indicate better image quality. However, PSNR does not always correlate well with human perception, especially for perceptually important features.

## SSIM 

SSIM assesses the similarity between two images in terms of luminance, contrast, and structure, mimicking human perception. Higher SSIM values indicate greater similarity between images. SSIM is considered more accurate than PSNR in assessing perceptual quality as it takes into account structural information.

## NC

NC measures the similarity between two signals or images by computing their cross-correlation and normalizing it. NC values close to 1 indicate high similarity, while values close to 0 indicate dissimilarity. NC is sensitive to changes in brightness and contrast but does not consider structural information as comprehensively as SSIM.

# Usage

Clone the repository onto your local machine.

Run the command: streamlit run Homepage.py

Use the web app on localhost.


# Code Contributors:
- Sachin Prasanna
- Abhayjit Singh Gulati
- Rounak Jain





