# Authors: Xiaoxuan Jia, Shailaja Akella

from scipy.stats import kurtosis
import cv2
import numpy as np
import math
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter


def img_mean(img):
    """Avearge pixel intensity (mean)"""
    tmp = img.flatten()
    return np.mean(tmp[tmp != 0])


def img_contrast(img):
    """s.d. of pixel intensity (contrast)"""
    tmp = img.flatten()
    return np.std(tmp[tmp != 0])


def img_kurtosis(img):
    """kurtosis of the pixel intensity (kurtosis)"""
    tmp = img.flatten()
    return kurtosis(tmp[tmp != 0])


def image_entropy(img):
    """calculate the entropy of an image"""
    # this could be made more efficient using numpy
    # histogram = cv2.calcHist([ttmp],[0],None,[256],[0,256]) # crash memory
    tmp = img.flatten()
    a = np.histogram(tmp[tmp != 0], bins=range(257))
    histogram = a[0]
    histogram_length = sum(histogram)
    samples_probability = [float(h) / histogram_length for h in histogram]
    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])


def energy(vid_mat):
    """calculate the energy of an image"""
    vid_rearr = rearrange(vid_mat, 'p b h -> p (b h)')
    diff_mat = np.abs(np.diff(vid_rearr, axis=0))
    std_sum = StandardScaler().fit_transform(np.mean(diff_mat, axis=1).reshape(-1, 1))
    std_sum = np.append(std_sum.reshape(vid_mat.shape[0] - 1), np.zeros(1))
    std_sum = std_sum.reshape(-1, 1)
    return savgol_filter(std_sum.T, 31, 3)


def edges(vid_mat):
    """calculate proportion of edges in an image"""
    prop_edges = np.zeros(vid_mat.shape[0])
    for i, frame in enumerate(vid_mat):
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(frame, (3, 3), 0)

        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
        prop_edges[i] = np.mean(edges)

    return prop_edges
