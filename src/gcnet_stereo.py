#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow CUDA load statements
import shutil
import argparse
import configparser
import datetime as dt
import numpy as np
import keras
from keras.models import save_model, load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
from PIL import Image
from experiment import Experiment, ExperimentFileParser
from configuration import *
from stereo_dataset import StereoDataset
from data_generator import generators_from_data
from gcnet_builder import GCNetBuilder
from preprocessor import Preprocessor
from metrics import less_1px, less_3px, less_5px
from callback import TestPredictionCallback, OptimizerStateCallback
from losses import sparse_mean_absolute_error
from cost_volume import _concat_features
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

print('All imports: done')


def test(mode, config_file, verbose, directory, testing_config, dataset_config, epochs, start_epoch):
    print('Mode: Prediction')

    # Build model
    net_builder = GCNetBuilder(testing_configuration.max_disparity)
    model = net_builder.build(input_image_shape=
                              (dataset_config.channels, testing_config.model_input_height, testing_config.model_input_width))

    opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae', less_1px, less_3px])

    # Load weights (if any)
    load_path = os.path.join('../out/', directory, testing_configuration.model_name + '.hdf5')
    if os.path.isfile(load_path):
        model.load_weights(load_path)
        print('Weights loading: Success', load_path)
    else:
        warnings.warn('*** !!!!! WARNING !!!!! No model weights file loaded ***', RuntimeWarning, stacklevel=2)

    # Load data to predict
    dataset = StereoDataset(dataset_config)
    (x_train_id, y_train_id), (x_val_id, y_val_id), (x_test_id, y_test_id) = dataset.load()

    # # Crop and expand dims to batch = 1
    crop_start_row = testing_config.crop_start_row
    crop_start_col = testing_config.crop_start_col
    crop_stop_row = crop_start_row + testing_config.crop_height
    crop_stop_col = crop_start_col + testing_config.crop_width

    left = np.empty((testing_config.batch_size, dataset_config.channels, testing_config.resized_height, testing_config.resized_width), dtype=np.float32)
    right = np.empty((testing_config.batch_size, dataset_config.channels, testing_config.resized_height, testing_config.resized_width), dtype=np.float32)

    preprocessor = Preprocessor()

    for i in range(len(x_test_id)):
        test_id = x_test_id[i]
        filename = os.path.splitext(os.path.basename(test_id[0]))[0]
        better_name = test_id[0].split(os.path.join(dataset_config.dataset_root, dataset_config.dataset_path))[1]
        better_name= better_name.replace('/', '_')
        print ('better_name', better_name)
        print('Testing on ', test_id, filename)

        img = img_to_array(load_img(test_id[0]), data_format='channels_first')
        left_img = img[:, crop_start_row:crop_stop_row, crop_start_col:crop_stop_col]
        img = preprocessor.resize_img(left_img, [dataset_config.channels, testing_config.resized_height, testing_config.resized_width])
        left[0] = img

        img = img_to_array(load_img(test_id[1]), data_format='channels_first')
        right_img = img[:, crop_start_row:crop_stop_row, crop_start_col:crop_stop_col]
        img = preprocessor.resize_img(right_img, [dataset_config.channels, testing_config.resized_height, testing_config.resized_width])
        right[0] = img

        disparity = model.predict([left, right], 1, 1)
        disparity = np.squeeze(disparity)

        uint8_disp = np.asarray(disparity, dtype=np.uint8)

        disp_img = Image.fromarray(uint8_disp)
        disp_img.save('../out/' + directory + '/' 'test_disparity_' + better_name + '.png')

    # Data generator
    # train_gen, val_gen, testing_generator = generators_from_data(x_train_id, y_train_id, x_val_id, y_val_id, x_test_id, y_test_id,
    #                                                testing_config, dataset_config)
    #
    # n_tests = 1
    #
    # # test_metrics = model.evaluate_generator(train_gen, steps=1) #len(x_test_id) // testing_config.batch_size,)
    # # print(test_metrics)
    #
    # test_predictions = model.predict_generator(testing_generator, steps=n_tests)  # len(x_test_id) // testing_config.batch_size,)
    #
    # print(test_predictions.shape)
    #
    # N = test_predictions.shape[0]
    # # disparity = np.squeeze(test_predictions)
    #
    # for i in range(N):
    #     filename = os.path.splitext(os.path.basename(test_id[0]))[0]
    #     better_name = test_id[0].split(os.path.join(dataset_config.dataset_root, dataset_config.dataset_path))[1]
    #     better_name= better_name.replace('/', '_')
    #
    #     # uint8_disp = np.asarray(disparity[i], dtype=np.uint8)
    #     uint8_disp = np.asarray(test_predictions[i], dtype=np.uint8)
    #
    #     disp_img = Image.fromarray(uint8_disp)
    #     disp_img.save('../out/' + directory + '/' + 'test_disparity' + str(i) + '.png')
    #     disp_img.save('../out/' + directory + '/' 'test_disparity_' + better_name + '.png')


def build_modelcheckpoint_callback(where_dir):
    checkpoint_directory = os.path.join(where_dir, 'checkpoints')
    if not os.path.isdir(checkpoint_directory):
        os.makedirs(checkpoint_directory)

    checkpoint_filepath = os.path.join(checkpoint_directory, 'weights-{epoch:02d}-{val_loss:.2f}.hdf5')
    callback = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=False, save_weights_only=False)
    return callback

# def train(mode, config_file, verbose, out_directory, config, dataset_config):
def train(mode, config_file, verbose, out_directory, config):
    # Load dataset (IDs only, not the real data)
    dataset = StereoDataset(config)
    (x_train_id, y_train_id), (x_val_id, y_val_id), (x_test_id, y_test_id) = dataset.load()

    # Data generator
    training_generator, validation_generator, _ = \
        generators_from_data(x_train_id, y_train_id, x_val_id, y_val_id, x_test_id, y_test_id, config)

    # Build model
    # net_builder = GCNetBuilder(training_config.max_disparity)
    net_builder = GCNetBuilder(config['max_disparity'])
    model = net_builder.build(input_image_shape= \
                                  (config['channels'], config['model_input_height'], config['model_input_width']))
                              # (dataset_config.channels, training_config.model_input_height, training_config.model_input_width))

    # model_filename = os.path.join(out_directory, training_config.model_name + '.hdf5')
    model_filename = os.path.join(out_directory, config['model_name'] + '.hdf5')

    if verbose:
        layers = model.summary()
        print(layers)

    if mode == 1:
        # Build optimizer
        # opt = keras.optimizers.RMSprop(lr=training_config.learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = keras.optimizers.RMSprop(lr=config['learning_rate'], rho=0.9, epsilon=1e-08, decay=0.0)
        # model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mae', less_1px, less_3px, less_5px])
        model.compile(loss=sparse_mean_absolute_error, optimizer=opt, metrics=[sparse_mean_absolute_error, 'mae', less_1px, less_3px, less_5px])
    else:
        if os.path.isfile(model_filename):
            model = load_model(model_filename, custom_objects={'_concat_features': _concat_features,
                                                   'sparse_mean_absolute_error': sparse_mean_absolute_error,
                                                   'less_1px': less_1px,
                                                   'less_3px': less_3px,
                                                   'less_5px': less_5px})
            print('Loading model: Success')

            # Show optimizer for information
            opt = model.optimizer
            print(opt)
        else:
            warnings.warn('*** !!!!! WARNING !!!!! No model file loaded ***', RuntimeWarning, stacklevel=2)

    # Callbacks
    lr_reducer_cb = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper_cb = EarlyStopping(min_delta=0.001, patience=10)
    # csv_logger_cb = CSVLogger(os.path.join(out_directory, training_config.model_name + '.csv'))
    csv_logger_cb = CSVLogger(os.path.join(out_directory, config['model_name'] + '.csv'))
    tensorboard_cb = TensorBoard(log_dir=os.path.join(out_directory, 'graph'), histogram_freq=0, write_graph=True, write_images=True)
    checkpoint_cb = build_modelcheckpoint_callback(out_directory)
    # predict_cb = TestPredictionCallback(x_test_id[0][0], x_test_id[0][1], out_directory, dataset_config, training_config)
    predict_cb = TestPredictionCallback(x_test_id[0][0], x_test_id[0][1], out_directory, config)
    training_callbacks = [csv_logger_cb, tensorboard_cb, predict_cb, checkpoint_cb]

    # Training
    start = dt.datetime.now()
    try:
        model.fit_generator(training_generator,
                              # steps_per_epoch=len(x_train_id) // training_config.batch_size,
                              steps_per_epoch=len(x_train_id) // config['batch_size'],
                              # epochs=training_config.num_epochs,
                              epochs=config['.num_epochs'],
                              validation_data=validation_generator,
                              # validation_steps=len(x_val_id) // training_config.batch_size,
                              validation_steps=len(x_val_id) // config['batch_size'],
                              callbacks=training_callbacks
                              )
    except RuntimeWarning as e:
        print('RuntimeWarning, stopping !')
        exit()
    except KeyboardInterrupt:
        print('Got interrupted ! Saving model')
    stop = dt.datetime.now()

    elapsed = stop - start
    print('Time to fit model: ', elapsed)

    # Save model picture, weights and experiment configuration
    save_model(model, model_filename)
    shutil.copy(config_file, os.path.join(out_directory, os.path.basename(os.path.normpath(config_file))))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', help='0: predict, 1: train', type=int, default=1)
    parser.add_argument('-config', help='Training or testing configuration file')
    parser.add_argument('-verbose', help='0:silent, 1: verbose', type=int, default=0)
    parser.add_argument('-dir', help='source directory')

    args = parser.parse_args()

    mode = args.mode
    config_file = args.config
    verbose = args.verbose
    out_dir = args.dir

    # config_file = '/home/herve/projects/deep_learning/keras/gcnet_stereo/experiments/test_config_very_fast.ini'
    # config_file = '/home/herve/projects/deep_learning/keras/gcnet_stereo/experiments/training_config_veryfast.ini'
    # config_file = '/home/herve/projects/deep_learning/keras/gcnet_stereo/experiments/training_config_kitti.ini'
    # directory = 'out_2017-12-12-12:03:34'
    config_file = '/home/herve/projects/deep_learning/keras/gcnet_stereo/experiments/training_config_kitti.json'

    if mode == None:
        print('Error: undefined mode (training/tuning/testing). Please specifiy which mode to run in.')
        exit()

    if mode == 0:
        # Parse configuration file
        test_config_file_parser = configparser.ConfigParser()
        dataset_config_file_parser = configparser.ConfigParser()

        test_config_file_parser.read_file(open(config_file))
        test_parser = TestFileParser()
        testing_configuration = test_parser.parse(test_config_file_parser)

        dataset_filename = os.path.join('../datasets', testing_configuration.dataset_name)
        dataset_config_file_parser.read_file(open(dataset_filename + '.ini'))
        dataset_config_parser = DatasetConfigFileParser()
        dataset_configuration = dataset_config_parser.parse(dataset_config_file_parser)

        test(mode, config_file, verbose, out_dir, testing_configuration, dataset_configuration, epochs, start_epoch)
    else:
        # Parse configuration file
        # experiment = ExperimentFileParser().read_config(config_file)
        # training_config_file_parser = configparser.ConfigParser()
        # dataset_config_file_parser = configparser.ConfigParser()
        #
        # training_config_file_parser.read_file(open(config_file))
        # training_parser = TrainingFileParser()
        # training_configuration = training_parser.parse(training_config_file_parser)
        #
        # dataset_filename = os.path.join('../datasets', training_configuration.dataset_name)
        # dataset_config_file_parser.read_file(open(dataset_filename + '.ini'))
        # dataset_config_parser = DatasetConfigFileParser()
        # dataset_configuration = dataset_config_parser.parse(dataset_config_file_parser)

        config = ExperimentFileParser().read_config(config_file)

        if mode == 1:
            print('Mode: Training')

            # Create timestamped directory
            timestamp = dt.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            out_dir = os.path.join('../out', 'out_' + timestamp)

            if os.path.isdir(out_dir):
                print('Error, training required but directory already exists')
                exit()
            else:
                os.makedirs(out_dir)
        elif mode == 2:
            print('Mode: Tuning')

            if not os.path.isdir(out_dir):
                print('Error, tuning required but directory does not exist')
                exit()

        # train(mode, config_file, verbose, out_dir, training_configuration, dataset_configuration)
        train(mode, config_file, verbose, out_dir, config)