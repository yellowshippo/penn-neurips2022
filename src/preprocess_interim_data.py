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
        '-r', '--recursive',
        type=strtobool,
        default=0,
        help='If True, serach data recursively [False]')
    args = parser.parse_args()

    preprocessor = siml.prepost.Preprocessor.read_settings(
        args.settings_yaml, force_renew=args.force_renew,
        recursive=args.recursive)
    preprocessor.preprocess_interim_data()

    print('success')


if __name__ == '__main__':
    main()
