import os
from sklearn.utils import shuffle
import cPickle


class Dataset():
    def __init__(self, config_filename):
        self.name = None
        self.dataset_root = None
        self.dataset_path = None
        self.image_list_path = None
        self.label_list_path = None
        cache_dir = os.path.join(self.dataset_root, self.dataset_path, self.name)
        if not os.path.isdir(cache_dir):
            pass
        else:
            pass

    def num_files_in_path(self, path):
        # _, _, files = os.walk(path).__next__()
        _, _, files = os.walk(path).next()
        file_count = len(files)
        return file_count

    def id_pairs_from_single_dir(self, path, extension):
        d = []
        file_count = self.num_files_in_path(path + 'left/')
        for id_ in range(1, file_count+1):
            left = path + 'left/' + '{:0>8}'.format(str(id_) + extension)
            right = path + 'right/' + '{:0>8}'.format(str(id_) + extension)
            d.append([left, right])
        return d

    def id_pairs(self, frames_or_labels='frames'):
        if frames_or_labels is 'frames':
            base_path = self.dataset_root + self.frames_path
            extension = '.png'
        else:
            base_path = self.dataset_root + self.disparity_path
            extension = '.pfm'

        d = []
        for path in self.scene_path_list:
            full_path = base_path + path
            id_list = self.id_pairs_from_single_dir(full_path, extension)
            d = d + id_list

        return d

    def load_data_ids(self):
        x_ids = self.id_pairs('frames')
        y_ids = self.id_pairs('labels')
        return x_ids, y_ids

    def train_test_split(self, frames, labels, train_ratio=0.8):
        x, y = shuffle(frames, labels)

        total = len(x)
        train_size = int(train_ratio * total)

        x_train = x[0:train_size]
        y_train = x[0:train_size]
        x_test = x[train_size:]
        y_test = y[train_size:]

        return (x_train, y_train), (x_test, y_test)

    def train_val_test_split(self, frames, labels, ratios=[0.7, 0.2, 0.1]):
        x, y = shuffle(frames, labels)

        total = len(x)
        train_size = int(ratios[0] * total)
        val_size = int(ratios[1] * total)
        test_size = total - train_size - val_size

        x_train = x[0:train_size]
        y_train = x[0:train_size]
        x_val = x[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        x_test = x[train_size+val_size:]
        y_test = y[train_size + val_size:]

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def load(self):
        # train_file = open(self.dataset_root + '/datasets/driving/train_set.txt', 'r')
        # val_file = open(self.dataset_root + '/datasets/driving/val_set.txt', 'r')
        # test_file = open(self.dataset_root + '/datasets/driving/test_set.txt', 'r')
        train_file = open(self.dataset_root + '/datasets/veryfast/train_set.txt', 'r')
        val_file = open(self.dataset_root + '/datasets/veryfast/val_set.txt', 'r')
        test_file = open(self.dataset_root + '/datasets/veryfast/test_set.txt', 'r')
        train_set = cPickle.load(train_file)
        val_set= cPickle.load(val_file)
        test_set = cPickle.load(test_file)

        x_train_id, y_train_id = zip(*train_set)
        x_val_id, y_val_id = zip(*val_set)
        x_test_id, y_test_id = zip(*test_set)

        train_file.close()
        val_file.close()
        test_file.close()

        return (x_train_id, y_train_id), (x_val_id, y_val_id), (x_test_id, y_test_id)

# d = Dataset('~/projects/deep_learning/keras/gcnet_stereo/dataset/sceneflow_driving.ini')
