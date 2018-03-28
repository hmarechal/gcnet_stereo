import numpy as np
from skimage.transform import resize


class Preprocessor():
    def center(self, img, axis=(1, 2)):
        sigma2 = np.var(img, axis=axis, keepdims=True)
        sigma2[np.where(sigma2 == 0.)] = 1.
        mean = np.mean(img, axis=axis, keepdims=True)
        result = (img - mean) / np.sqrt(sigma2)
        return result

    def resize_img(self, img, output_shape):
        img_resized = resize(img, output_shape, preserve_range=True)
        return img_resized

    def resize_label(self, img, output_shape):
        img_resized = resize(img, output_shape, preserve_range=True)
        return img_resized
