import numpy as np
import skimage
import cv2


def shift(arr, coordx=0, coordy=0):
    coords = (coordx, coordy)

    def get_slice(coord, size):
        if coord >= 0:
            return slice(0, size - coord, 1)
        else:
            return slice(-coord, size, 1)

    res = np.zeros(arr.shape, dtype=arr.dtype)
    if len(arr.shape) == 2:
        res[get_slice(coords[1], arr.shape[0]), get_slice(-coords[0], arr.shape[1])] = \
            arr[get_slice(-coords[1], arr.shape[0]), get_slice(coords[0], arr.shape[1])]
    elif len(arr.shape) == 3:
        res[:, get_slice(coords[1], arr.shape[1]), get_slice(-coords[0], arr.shape[2])] = \
            arr[:, get_slice(-coords[1], arr.shape[1]), get_slice(coords[0], arr.shape[2])]
    else:
        raise TypeError
    return res


def rotation(arr, angle):
    if len(arr.shape) == 2:
        return skimage.transform.rotate(arr, angle)
    elif len(arr.shape) == 3:
        return np.array([skimage.transform.rotate(obj, angle) for obj in arr])
    else:
        raise TypeError


def gaussian_blur(arr, ksize1=3, ksize2=3, sigmaX=1.0, sigmaY=None):
    if sigmaY is None:
        sigmaY = sigmaX
    ksize = (ksize1, ksize2)
    if len(arr.shape) == 2:
        return cv2.GaussianBlur(arr, ksize, sigmaX, sigmaY)
    elif len(arr.shape) == 3:
        return np.array([cv2.GaussianBlur(obj, ksize, sigmaX, sigmaY) for obj in arr])
    else:
        raise TypeError


def erosion(arr, kernel1=2, kernel2=2, iters=1):
    kernel = np.ones((kernel1, kernel2))
    if len(arr.shape) == 2:
        return cv2.erode(arr, kernel, iters)
    else:
        return np.array([cv2.erode(obj, kernel, iters) for obj in arr])


def dilation(arr, kernel1=2, kernel2=2, iters=1):
    kernel = np.ones((kernel1, kernel2))
    if len(arr.shape) == 2:
        return cv2.dilate(arr, kernel, iters)
    else:
        return np.array([cv2.dilate(obj, kernel, iters) for obj in arr])


def opening(arr, kernel1=2, kernel2=2):
    kernel = np.ones((kernel1, kernel2))
    if len(arr.shape) == 2:
        return cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    else:
        return np.array([cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel) for obj in arr])


def closing(arr, kernel1=2, kernel2=2):
    kernel = np.ones((kernel1, kernel2))
    if len(arr.shape) == 2:
        return cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    else:
        return np.array([cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel) for obj in arr])


def do_transforms(arr, what):
    if len(what) == 0:
        return arr.copy()
    funcs = {
        'rotate': rotation,
        'shift': shift,
        'erode': erosion,
        'dilate': dilation,
        'open': opening,
        'close': closing,
        'blur': gaussian_blur
    }
    for tr in what:
        func = funcs[tr[0]]
        if len(tr) == 1:
            arr = func(arr)
        else:
            arr = func(arr, *tr[1:])
    return arr
