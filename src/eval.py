import argparse
from distutils.util import strtobool
import pathlib

import siml

import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_path',
        type=pathlib.Path,
        help='Pretrained model path')
    parser.add_argument(
        'data_directories',
        type=pathlib.Path,
        nargs='+',
        help='Data directory')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output directory name')
    parser.add_argument(
        '-b', '--output-base',
        type=pathlib.Path,
        default=None,
        help='Output base directory name')
    parser.add_argument(
        '-p', '--preprocessors-pkl',
        type=pathlib.Path,
        default=None,
        help='Preprocessors.pkl file')
    parser.add_argument(
        '--perform-preprocess',
        type=strtobool,
        default=0,
        help='If true, perform preprocess')
    parser.add_argument(
        '-w', '--write-simulation-base',
        type=pathlib.Path,
        default=None,
        help='Simulation base directory to write inferred data')
    parser.add_argument(
        '-a', '--analyse-error-mode',
        type=str,
        default=None,
        help='If fed, analyse error stats [grad, ad, fluid]')

    args = parser.parse_args()

    if args.data_directories[0].is_file():
        with open(args.data_directories[0]) as f:
            lines = f.readlines()
        data_directories = [
            pathlib.Path(line.strip()) for line in lines if line.strip() != '']
    else:
        data_directories = args.data_directories

    inferer = siml.inferer.Inferer.from_model_directory(
        args.model_path, save=True,
        converter_parameters_pkl=args.preprocessors_pkl)
    inferer.setting.trainer.gpu_id = -1
    if args.output_base is not None:
        inferer.setting.inferer.output_directory_base = args.output_base
        print(inferer.setting.inferer.output_directory_base)
    else:
        inferer.setting.inferer.output_directory = args.out_dir
    inferer.setting.inferer.write_simulation = True
    if args.write_simulation_base:
        inferer.setting.inferer.write_simulation_base \
            = args.write_simulation_base
    inferer.setting.inferer.read_simulation_type = 'polyvtk'
    inferer.setting.inferer.write_simulation_type = 'polyvtk'
    inferer.setting.conversion.skip_femio = False
    inferer.setting.conversion.required_file_names = ['mesh.vtu']

    results = inferer.infer(
        data_directories=data_directories,
        perform_preprocess=args.perform_preprocess)

    if args.analyse_error_mode is None:
        pass
    else:
        output_base = stats.determine_output_base(results)
        if args.analyse_error_mode == 'grad':
            stats.grad_analyse_error(results, output_base)
        elif args.analyse_error_mode == 'ad':
            stats.ad_analyse_error(results, output_base)
        elif args.analyse_error_mode == 'fluid':
            stats.fluid_analyse_error(results, output_base)
        else:
            raise ValueError(
                f"Invalid --analyse-error-mode: {args.analyse_error_mode}")

    return


if __name__ == '__main__':
    main()
