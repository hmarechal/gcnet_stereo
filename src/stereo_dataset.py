import os
import glob
from sklearn.utils import shuffle
import sys
if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle


class StereoDataset():
    def __init__(self, dataset_config):
        self.dataset_name = dataset_config.dataset['name']
        self.dataset_root = dataset_config.dataset['dataset_root']
        self.dataset_path = dataset_config.dataset['dataset_path']
        self.x_id = dataset_config.dataset['x_id']
        self.y_id = dataset_config.dataset['y_id']
        self.x_train_id = dataset_config.dataset['x_train_id']
        self.y_train_id = dataset_config.dataset['y_train_id']
        self.x_val_id = dataset_config.dataset['x_val_id']
        self.y_val_id = dataset_config.dataset['y_val_id']
        self.x_test_id = dataset_config.dataset['x_test_id']
        self.y_test_id = dataset_config.dataset['y_test_id']
        self.train_val_test_ratios = dataset_config.dataset['train_val_test_ratios']
        # self.data_list_path = dataset_config.dataset['data_list_path']
        # self.label_list_path = dataset_config.dataset['label_list_path']
        # self.left_data = dataset_config.dataset['left_data']
        # self.right_data = dataset_config.dataset['right_data']
        # self.left_label = dataset_config.dataset['left_label']
        # self.right_label = dataset_config.dataset['right_label']

    def num_files_in_path(self, path):
        if (sys.version_info > (3, 0)):
            _, _, files = os.walk(path).__next__()
        else:
             _, _, files = os.walk(path).next()
        file_count = len(files)
        return file_count

    def id_pairs_from_single_dir(self, path, extension, left, right):
        left_list = glob.glob(os.path.join(path, left))
        left_list.sort()

        right_list = glob.glob(os.path.join(path, right))
        right_list.sort()

        d = list(zip(left_list, right_list))

        return d

    def id_pairs(self, frames_or_labels='frames'):
        base_path = os.path.join(self.dataset_root, self.dataset_path)
        if frames_or_labels is 'frames':
            extension = '.png'
            list_path = self.data_set_path
            left = self.left_data
            right = self.right_data
        else:
            extension = '.pfm'
            list_path = self.label_list_path
            left = self.left_label
            right = self.right_label
            # dirty temporary hack to open .png images instead of trying to open .pfm with Kitti dataset
            if left == 'disp_noc_0':
                extension = '.png'

        d = []
        for path in list_path:
            full_path = os.path.join(base_path, path)
            id_list = self.id_pairs_from_single_dir(full_path, extension, left, right)
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
        y_train = y[0:train_size]

        x_val = x[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]

        x_test = x[train_size+val_size:]
        y_test = y[train_size + val_size:]

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def load(self):
        cur_script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(cur_script_dir, '../datasets')
        load_dir = os.path.join(cache_dir, self.dataset_name)

        train_file = open(os.path.join(load_dir, 'train_set.txt'), 'rb')
        val_file = open(os.path.join(load_dir, 'val_set.txt'), 'rb')
        test_file = open(os.path.join(load_dir, 'test_set.txt'), 'rb')

        train_set = pickle.load(train_file)
        val_set= pickle.load(val_file)
        test_set = pickle.load(test_file)

        x_train_id, y_train_id = zip(*train_set)
        x_val_id, y_val_id = zip(*val_set)
        x_test_id, y_test_id = zip(*test_set)

        train_file.close()
        val_file.close()
        test_file.close()

        return (x_train_id, y_train_id), (x_val_id, y_val_id), (x_test_id, y_test_id)