import json
from collections import OrderedDict


class Experiment():
    @staticmethod
    def from_config(config):
        exp = Experiment()
        exp.dataset = config['dataset']
        exp.preprocessing= config['preprocessing']
        exp.model = config['model']
        exp.learning = config['learning']

        print(exp)

        return exp

    def __getitem__(self, item):
        try:
            item = self.dataset[item]
        except:
            pass
        else:
            return item

        try:
            item = self.model[item]
        except:
            pass
        else:
            return item

        try:
            item = self.learning[item]
        except:
            pass
        else:
            return item

        try:
            first_item = self.preprocessing
            try:
                item = first_item['ZeroPadding2D'][item]
            except:
                pass
            else:
                return item

            try:
                item = first_item['crop'][item]
            except:
                pass
            else:
                return item

            try:
                item = first_item['resize'][item]
            except:
                pass
            else:
                return item

        except:
            print('Unfound item')
            raise
        else:
            return item



class ExperimentFileParser():
    def as_list(self, value, cast_to=float):
        val = [x.strip(', ') for x in value.splitlines()]
        return val

    def read_config(self, config_filename):
        with open(config_filename) as json_data_file:
            data = json.load(json_data_file, object_pairs_hook=OrderedDict)

        config = Experiment.from_config(data)

        return config

# config = ExperimentFileParser().read_config('../experiments/training_config_kitti.json')
# pass
