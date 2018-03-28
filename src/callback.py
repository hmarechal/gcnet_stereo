from PIL import Image
import numpy as np
from keras import  callbacks
from keras.preprocessing.image import img_to_array, load_img
from preprocessor import Preprocessor


class TestPredictionCallback(callbacks.Callback):
    def __init__(self, left_filename, right_filename, directory, config):
        super(TestPredictionCallback, self).__init__()

        self.directory = directory

        # Crop and expand dims to batch = 1
        crop_start_row = config['crop_start_row']
        crop_start_col = config['crop_start_col']
        crop_stop_row = crop_start_row + config['crop_height']
        crop_stop_col = crop_start_col + config['crop_width']

        preprocessor = Preprocessor()

        img = img_to_array(load_img(left_filename), data_format='channels_first')
        img = img[:, crop_start_row:crop_stop_row, crop_start_col:crop_stop_col]
        left_img = preprocessor.resize_img(img, [config['channels'], config['resized_height'], config['resized_width']])

        img = img_to_array(load_img(right_filename), data_format='channels_first')
        img = img[:, crop_start_row:crop_stop_row, crop_start_col:crop_stop_col]
        right_img = preprocessor.resize_img(img, [config['channels'], config['resized_height'], config['resized_width']])

        self.left_img = np.expand_dims(left_img, axis=0)
        self.right_img = np.expand_dims(right_img, axis=0)

    def on_epoch_end(self, epoch, logs=None):
        disparity = self.model.predict([self.left_img, self.right_img], 1, 1)

        disparity = np.squeeze(disparity)
        uint8_disp = np.asarray(disparity, dtype=np.uint8)

        disp_img = Image.fromarray(uint8_disp)
        disp_img.save(self.directory + '/' + 'disparity_' + str(epoch) + '.png')

class OptimizerStateCallback(callbacks.Callback):
    def __init__(self):
        super(OptimizerStateCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        config = self.model.optimizer.get_config()
        print(config)
