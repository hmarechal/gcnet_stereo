import numpy as np
from random import randint
from pfm_reader import PFMReader
from preprocessor import Preprocessor
from keras.preprocessing.image import img_to_array, load_img
import os
from scipy import misc


class DataGenerator():
    # def __init__(self, training_config, dataset_config):
    def __init__(self, config):
        self.full_width = config['width']
        self.full_height = config['height']
        self.channels = config['channels']
        # self.crop_width = config['crop_width']
        # self.crop_height = config['crop_height']
        # self.resized_width = config['resized_width']
        # self.resized_height = config['resized_height']
        self.resize = config['resize']
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        # self.random_crop = config['random_crop']
        self.crop_params = config['crop']
        self.pfm_reader = PFMReader()
        self.preprocessor= Preprocessor()

    def get_crop_window(self, full_width, full_height, crop_width, crop_height, random=False):
        if random:
            start_row = randint(0, full_height - crop_height)
            stop_row = start_row + crop_height

            start_col = randint(0, full_width - crop_width)
            stop_col = start_col + crop_width
        else:
            start_row = 200
            stop_row = 200 + 224
            start_col = 100
            stop_col = 100 + 224

        return start_row, stop_row, start_col, stop_col

    def _get_exploration_order(self, list_IDs):
        """Generates order of exploration"""
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def _generate_batch(self, labels, list_IDs_temp):
        """Generates data of batch_size samples"""
        left_batch = np.empty((self.batch_size, self.channels, self.resized_height, self.resized_width), dtype=np.float32)
        right_batch = np.empty((self.batch_size, self.channels, self.resized_height, self.resized_width), dtype=np.float32)
        y = np.empty((self.batch_size, self.resized_height, self.resized_width), dtype=np.float32)

        for i, ID in enumerate(list_IDs_temp):
            start_row, stop_row, start_col, stop_col = self.get_crop_window(self.full_width, self.full_height, self.crop_width, self.crop_height, self.random_crop)

            left = img_to_array(load_img(ID[0]), data_format='channels_first')
            left = left[:, start_row:stop_row, start_col:stop_col]
            left = self.preprocessor.resize_img(left, [self.channels, self.resized_height, self.resized_width])
            left_batch[i, :, :, :] = left

            right = img_to_array(load_img(ID[1]), data_format='channels_first')
            right = right[:, start_row:stop_row, start_col:stop_col]
            right = self.preprocessor.resize_img(right, [self.channels, self.resized_height, self.resized_width])
            right_batch[i, :, :, :] = right

            extension = os.path.splitext(labels[i][0])[1]
            if extension == '.pfm':
                data = self.pfm_reader.readPFM(labels[i][0])
                tmp = data[start_row:stop_row, start_col:stop_col]
            else:
                data = misc.imread(labels[i][0])
                tmp = (np.float32) (data[start_row:stop_row, start_col:stop_col] / 256.)
            tmp = self.preprocessor.resize_label(tmp, [self.resized_height, self.resized_width])
            y[i] = tmp

        return [left_batch, right_batch], y

    def iter_generate_batches(self, labels, list_ids):
        # Generates batches of samples
        while 1:
            # Generate order of exploration of dataset
            indexes = self._get_exploration_order(list_ids)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_ids_temp = [list_ids[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]
                labels_temp = [labels[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                # Generate data
                X, y = self._generate_batch(labels_temp, list_ids_temp)

                yield X, y


# def generators_from_data(x_train_id, y_train_id, x_val_id, y_val_id, x_test_id, y_test_id, training_config, dataset_config):
def generators_from_data(x_train_id, y_train_id, x_val_id, y_val_id, x_test_id, y_test_id, config):
    partition = dict()
    partition['train'] = x_train_id
    partition['valid'] = x_val_id
    partition['test'] = x_test_id

    train_labels = dict()
    for idx, element in enumerate(x_train_id):
        train_labels[idx] = y_train_id[idx]
    # training_generator = DataGenerator(training_config, dataset_config).iter_generate_batches(train_labels, partition['train'])
    training_generator = DataGenerator(config).iter_generate_batches(train_labels, partition['train'])

    val_labels = dict()
    for idx, element in enumerate(x_val_id):
        val_labels[idx] = y_val_id[idx]
    # validation_generator = DataGenerator(training_config, dataset_config).iter_generate_batches(val_labels, partition['valid'])
    validation_generator = DataGenerator(config).iter_generate_batches(val_labels, partition['valid'])

    test_labels = dict()
    for idx, element in enumerate(x_test_id):
        test_labels[idx] = y_test_id[idx]
    # testing_generator = DataGenerator(training_config, dataset_config).iter_generate_batches(test_labels, partition['test'])
    testing_generator = DataGenerator(config).iter_generate_batches(test_labels, partition['test'])

    return training_generator, validation_generator, testing_generator
