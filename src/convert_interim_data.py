import argparse
from distutils.util import strtobool
import pathlib

import siml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        '-f', '--force-renew',
        type=strtobool,
        default=0,
        help='If True, overwrite existing data [False]')
    parser.add_argument(
        '-i', '--input-directory',
        type=pathlib.Path,
        default=None,
        help='Input directory path')
    parser.add_argument(
        '-o', '--output-directory',
        type=pathlib.Path,
        default=None,
        help='Output directory path')
    parser.add_argument(
        '-p', '--preprocessors-pkl',
        type=pathlib.Path,
        default=None,
        help='preprocessors.pkl file')
    parser.add_argument(
        '-m', '--allow-missing',
        type=strtobool,
        default=0,
        help='If True, continue even if some of variables are missing [False]')
    args = parser.parse_args()

    setting = siml.setting.MainSetting.read_settings_yaml(args.settings_yaml)
    if args.input_directory is not None:
        setting.data.interim = [args.input_directory]
    if args.output_directory is not None:
        setting.data.preprocessed = [args.output_directory]
    preprocessor = siml.prepost.Preprocessor(
        setting, force_renew=args.force_renew,
        allow_missing=args.allow_missing)
    preprocessor.convert_interim_data(preprocessor_pkl=args.preprocessors_pkl)

    print('success')
    return


if __name__ == '__main__':
    main()
