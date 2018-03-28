class Configuration(object):
    def __init__(self):
        self.name = None


class TrainingConfiguration(object):
    def __init__(self):
        # Constants
        self.model_name = None
        self.max_disparity = None
        self.model_input_width = None
        self.model_input_height = None

        self.dataset_name = None

        self.random_crop = None
        self.crop_start_row = None
        self.crop_start_col = None
        self.crop_width = None
        self.crop_height = None
        self.resized_width = None
        self.resized_height = None

        self.batch_size = None
        self.num_epochs = None
        self.shuffle = None
        self.learning_rate = None
        self.momentum = None


class TestConfiguration(object):
    def __init__(self):
        # Constants
        self.model_name = None
        self.max_disparity = None
        self.model_input_width = None
        self.model_input_height = None

        self.dataset_name = None

        self.random_crop = None
        self.crop_start_row = None
        self.crop_start_col = None
        self.crop_width = None
        self.crop_height = None
        self.resized_width = None
        self.resized_height = None

        self.batch_size = None
        self.shuffle = None


class DatasetConfiguration(object):
    def __init__(self):
        # Constants
        self.dataset_name = None

        self.dataset_root = None
        self.dataset_path = None
        self.data_list_path = None
        self.label_list_path = None

        self.full_width = None
        self.full_height = None
        self.channels = None
        
        self.train_val_test_ratios = None

        self.left_data = None
        self.right_data = None
        self.left_label = None
        self.right_label = None


class TrainingFileParser():
    def as_list(self, value, cast_to=float):
        val = [x.strip(', ') for x in value.splitlines()]
        return val

    def parse(self, config):
        model = config['MODEL']
        dataset = config['DATASET']
        preprocessing = config['PREPROCESSING']
        learning = config['LEARNING']

        configuration = TrainingConfiguration()

        configuration.model_name = model['model_name']
        configuration.max_disparity = int(model['max_disparity'])
        configuration.model_input_width = int(model['model_input_width'])
        configuration.model_input_height = int(model['model_input_height'])
        configuration.model_input_depth = int(model['model_input_depth'])

        configuration.dataset_name = dataset['dataset_name']

        configuration.random_crop = bool(preprocessing['random_crop'])
        configuration.crop_start_row = int(preprocessing['crop_start_row'])
        configuration.crop_start_col =int(preprocessing['crop_start_col'])
        configuration.crop_width = int(preprocessing['crop_width'])
        configuration.crop_height = int(preprocessing['crop_height'])
        configuration.resized_width = int(preprocessing['resized_width'])
        configuration.resized_height = int(preprocessing['resized_height'])

        configuration.batch_size = int(learning['batch_size'])
        configuration.num_epochs = int(learning['num_epochs'])
        configuration.shuffle = learning['shuffle']
        configuration.learning_rate = float(learning['learning_rate'])
        configuration.momentum = float(learning['momentum'])

        return configuration


class TestFileParser():
    def parse(self, config):
        model = config['MODEL']
        dataset = config['DATASET']
        preprocessing = config['PREPROCESSING']
        testing = config['TESTING']

        configuration = TestConfiguration()

        configuration.model_name = model['model_name']
        configuration.max_disparity = int(model['max_disparity'])
        configuration.model_input_width = int(model['model_input_width'])
        configuration.model_input_height = int(model['model_input_height'])
        configuration.model_input_depth = int(model['model_input_depth'])

        configuration.dataset_name = dataset['dataset_name']

        configuration.random_crop = bool(preprocessing['random_crop'])
        configuration.crop_start_row = int(preprocessing['crop_start_row'])
        configuration.crop_start_col =int(preprocessing['crop_start_col'])
        configuration.crop_width = int(preprocessing['crop_width'])
        configuration.crop_height = int(preprocessing['crop_height'])
        configuration.resized_width = int(preprocessing['resized_width'])
        configuration.resized_height = int(preprocessing['resized_height'])

        configuration.batch_size = int(testing['batch_size'])
        configuration.shuffle = testing['shuffle']

        return configuration


class DatasetConfigFileParser():
    def as_list(self, value, cast_to=float):
        val = [x.strip(', ') for x in value.splitlines()]
        return val

    def parse(self, config):
        dataset = config['DATASET']

        configuration = DatasetConfiguration()

        configuration.dataset_name = dataset['name']
        configuration.full_width = int(dataset['width'])
        configuration.full_height = int(dataset['height'])
        configuration.channels = int(dataset['channels'])
        configuration.train_val_test_ratios = self.as_list(dataset['train_val_test_ratios'])

        configuration.dataset_root = dataset['dataset_root']
        configuration.dataset_path = dataset['dataset_path']
        configuration.data_list_path = self.as_list(dataset['data_list_path'])
        configuration.label_list_path = self.as_list(dataset['label_list_path'])

        configuration.left_data = dataset['left_data']
        configuration.right_data = dataset['right_data']
        configuration.left_label = dataset['left_label']
        configuration.right_label = dataset['right_label']

        return configuration
