#!/usr/bin/env python

import os
import argparse
import configparser
import sys
from experiment import Experiment, ExperimentFileParser

if (sys.version_info > (3, 0)):
    import pickle
else:
    import cPickle as pickle
from configuration import DatasetConfigFileParser
from stereo_dataset import StereoDataset


def main(dataset_cfg):
    # Load dataset (IDs only, not the real data)
    # dataset = StereoDrivingDataset(cfg.dataset_name, cfg.dataset_root, cfg.dataset_path, cfg.data_list_path, cfg.label_list_path)
    dataset = StereoDataset(dataset_cfg)
    x_ids, y_ids = dataset.load_data_ids()
    (x_train_id, y_train_id), (x_val_id, y_val_id), (x_test_id, y_test_id) = dataset.train_val_test_split(x_ids, y_ids)

    cur_script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(cur_script_dir, '../datasets')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    save_dir = os.path.join(cache_dir, dataset_cfg.dataset_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_file = open(os.path.join(save_dir, 'train_set.txt'), 'wb')
    val_file = open(os.path.join(save_dir, 'val_set.txt'), 'wb')
    test_file = open(os.path.join(save_dir, 'test_set.txt'), 'wb')

    pickle.dump(zip(x_train_id, y_train_id), train_file)
    pickle.dump(zip(x_val_id, y_val_id), val_file)
    pickle.dump(zip(x_test_id, y_test_id), test_file)

    train_file.close()
    val_file.close()
    test_file.close()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', help='dataset configuration file to be parsed')

    args = parser.parse_args()
    # args.filename = '../datasets/sceneflow_driving_veryfast.ini'
    # args.filename = '../datasets/kitti.ini'
    args.filename = '/home/herve/projects/deep_learning/keras/gcnet_stereo/experiments/training_config_kitti.json'

    # Parse configuration file
    # config = configparser.ConfigParser()
    # config.read_file(open(args.filename))
    # dataset_config_parser = DatasetConfigFileParser()
    # dataset_configuration = dataset_config_parser.parse(config)

    config = ExperimentFileParser().read_config(args.filename)

    # main(dataset_configuration)
    main(config)
