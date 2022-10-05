import argparse
from distutils.util import strtobool
import glob
import pathlib
import sys

import siml

sys.path.append('.')
sys.path.append('lib/siml/tests')
from lib.siml.tests import test_iso_gcn  # NOQA


RANK0_VARIABLE_PATTERN = 'nodal_p_step40'
RANK1_VARIABLE_PATTERN = 'nodal_U_step40'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        'original_data_directory',
        type=pathlib.Path,
        help='Original data directory')
    parser.add_argument(
        'transformed_data_directory',
        type=pathlib.Path,
        help='Transformed data directory')
    parser.add_argument(
        '-e', '--threshold_percent',
        type=float,
        default=.1,
        help='Threshold for relative comparison')
    parser.add_argument(
        '-d', '--decimal',
        type=int,
        default=3,
        help='Decimal for numpy testing.')
    parser.add_argument(
        '-o', '--output-directory',
        type=pathlib.Path,
        default=None,
        help='Output directory name. '
        'If not fed, will be determined automatically.')
    parser.add_argument(
        '-w', '--write-visualization',
        type=strtobool,
        default=0,
        help='If True, write visualization file of converted data [False]')
    parser.add_argument(
        '-r', '--required-file-name',
        type=str,
        default='nodal_U_step0.npy',
        help='Required file name to search for input data')
    parser.add_argument(
        '-p', '--preprocessors-pkl',
        type=pathlib.Path,
        default=None,
        help='Preprocessors.pkl file')
    args = parser.parse_args()

    validator = EquivarianceValidator(
        args.original_data_directory, args.transformed_data_directory,
        args)
    validator.validate()
    return


class EquivarianceValidator(test_iso_gcn.TestIsoGCN):

    def __init__(
            self,
            original_data_directory, transformed_data_directory, settings):
        self.original_data_directory = original_data_directory
        self.transformed_data_directory = transformed_data_directory
        self.settings = settings
        self.map_data_directories = self._generate_map_data_directories()

        if self.settings.preprocessors_pkl is None:
            self.settings.preprocessors_pkl = \
                self.original_data_directory / 'preprocessors.pkl'
        if not self.settings.preprocessors_pkl.is_file():
            raise ValueError(
                f"{self.settings.preprocessors_pkl} does not exist. "
                'Please feed --preprocessors-pkl option'
            )

        self._initialize_inferer()
        return

    def validate(self):
        for map_data_directory in self.map_data_directories:
            original_results = self.infer(map_data_directory['original'])
            transformed_results = self.infer(
                map_data_directory['list_transformed'])
            self.validate_results(
                original_results, transformed_results,
                rank0=RANK0_VARIABLE_PATTERN,
                rank1=RANK1_VARIABLE_PATTERN,
                decimal=self.settings.decimal,
                validate_x=False,
                threshold_percent=self.settings.threshold_percent)
        return

    def infer(self, data_directory):
        if isinstance(data_directory, list):
            data_directories = data_directory
        else:
            data_directories = [data_directory]
        print(f"Model: {self.model_directory}")
        return self.inferer.infer(
            model=self.model_directory,
            data_directories=data_directories)

    def _generate_map_data_directories(self):
        original_data_directories = siml.util.collect_data_directories(
            self.original_data_directory,
            required_file_names=self.settings.required_file_name)
        map_data_directories = []
        for original_data_directory in original_data_directories:
            list_transformed = list(self.transformed_data_directory.glob(
                '**/' + str(original_data_directory.relative_to(
                    self.original_data_directory))))
            if len(list_transformed) == 0:
                continue
            map_data_directories.append({
                'original': original_data_directory,
                'list_transformed': list_transformed,
            })
        return map_data_directories

    def _initialize_inferer(self):
        main_setting = siml.setting.MainSetting.read_settings_yaml(
            self.settings.settings_yaml)
        main_setting.data.train = [self.original_data_directory]
        main_setting.data.validation = [self.original_data_directory]
        main_setting.data.test = []
        main_setting.data.develop = []
        if self.settings.output_directory is None:
            main_setting.trainer.update_output_directory()
        else:
            main_setting.trainer.output_directory \
                = self.settings.output_directory
        main_setting.trainer.split_ratio = {}
        main_setting.trainer.n_epoch = 3
        main_setting.trainer.log_trigger_epoch = 1
        main_setting.trainer.stop_trigger_epoch = 1
        main_setting.trainer.batch_size = 1
        main_setting.trainer.validation_batch_size = 1
        main_setting.trainer.recursive = True

        trainer = siml.trainer.Trainer(main_setting)
        trainer.train()

        self.inferer = siml.inferer.Inferer(
            main_setting,
            converter_parameters_pkl=self.settings.preprocessors_pkl,
        )
        self.model_directory = str(main_setting.trainer.output_directory)
        return


if __name__ == '__main__':
    main()
