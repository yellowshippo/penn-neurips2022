import argparse
from distutils.util import strtobool
from glob import glob
import pathlib
import shutil

import numpy as np
import pandas as pd


DICT_FLUID_MODEL_PATHS = {
    'penn_n16_rep4': 'pretrained/fluid/raw/penn_rep4',
    'wo_boundary_condition_in_nns': 'pretrained/fluid/raw/penn_wo_bc_wo_pinv_after_bc',  # NOQA
    'penn_n4_rep4': 'pretrained/fluid/raw/penn4_rep4',
    'penn_n4_rep8': 'pretrained/fluid/raw/penn4',
    'penn_n8_rep4': 'pretrained/fluid/raw/penn8_rep4',
    'penn_n8_rep8': 'pretrained/fluid/raw/penn8',
    'penn_n8_rep8': 'pretrained/fluid/raw/penn8',
    'mp-pde_tw2_n32': 'pretrained/fluid/raw/MPPDE32/MPPDE_ns_fluid_n1_tw2_unrolling1_time728491',  # NOQA
    'mp-pde_tw4_n32': 'pretrained/fluid/raw/MPPDE32/MPPDE_ns_fluid_n2_tw4_unrolling1_time728491',  # NOQA
    'mp-pde_tw10_n32': 'pretrained/fluid/raw/MPPDE32/MPPDE_ns_fluid_n6_tw10_unrolling1_time728486',  # NOQA
    'mp-pde_tw20_n32': 'pretrained/fluid/raw/MPPDE32/MPPDE_ns_fluid_n11_tw20_unrolling1_time7284657',  # NOQA
    'mp-pde_tw2_n64': 'pretrained/fluid/raw/MPPDE64/MPPDE_ns_fluid_n1_tw2_unrolling1_time728211625',  # NOQA
    'mp-pde_tw4_n64': 'pretrained/fluid/raw/MPPDE64/MPPDE_ns_fluid_n2_tw4_unrolling1_time728213229',  # NOQA
    'mp-pde_tw10_n64': 'pretrained/fluid/raw/MPPDE64/MPPDE_ns_fluid_n6_tw10_unrolling1_time728211610',  # NOQA
    'mp-pde_tw20_n64': 'pretrained/fluid/raw/MPPDE64/MPPDE_ns_fluid_n11_tw20_unrolling1_time72821161',  # NOQA
}

DICT_AD_MODEL_PATHS = {
    'wo_boundary_condition_in_nns': 'pretrained/advection_diffusion/wo_bc_wo_pinv_after_bc',  # NOQA
    'penn': 'pretrained/advection_diffusion/penn',  # NOQA
    'wo_boundary_condition_input': 'pretrained/advection_diffusion/wo_boundary_condition_input',  # NOQA
    'wo_pseudoinverse_decoder_w_dirichlet_layer_after_decoding': 'pretrained/advection_diffusion/wo_pseudoinverse_decoder_w_dirichlet_layer_after_decoding',  # NOQA
    'wo_neural_nonlinear_solver': 'pretrained/advection_diffusion/wo_neural_nonlinear_solver',  # NOQA
    'wo_dirichlet_layer': 'pretrained/advection_diffusion/wo_dirichlet_layer',  # NOQA
    'wo_pseudoinverse_decoder': 'pretrained/advection_diffusion/wo_pseudoinverse_decoder',  # NOQA
    'wo_encoded_boundary': 'pretrained/advection_diffusion/wo_encoded_boundary',  # NOQA
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--test',
        type=strtobool,
        default=1,
        help='If True, write in tests/data/pretrained directory [True]')
    parser.add_argument(
        '-k', '--key',
        type=str,
        default='fluid',
        help='key: [fluid], ad')
    args = parser.parse_args()

    if args.test:
        output_directory_base = pathlib.Path('tests/data/pretrained')
    else:
        output_directory_base = pathlib.Path('deploy')

    if args.key == 'fluid':
        dict_model_paths = DICT_FLUID_MODEL_PATHS
    elif args.key == 'ad':
        args.key = 'advection_diffusion'
        dict_model_paths = DICT_AD_MODEL_PATHS
    else:
        raise ValueError(f"Unexpected key: {args.key}")

    for model_name, model_root_directory in dict_model_paths.items():
        output_directory = output_directory_base / args.key / model_name
        output_directory.mkdir(parents=True)

        if 'mp-pde' in model_name:
            deploy_model_mppde(
                model_name, model_root_directory, output_directory)
        else:
            deploy_model_penn(
                model_name, model_root_directory, output_directory)

    return


def deploy_model_penn(model_name, model_root_directory, output_directory):
    model_root_path = pathlib.Path(model_root_directory)
    logs = [
        pathlib.Path(g) for g
        in glob(str(model_root_path / '**/log.csv'), recursive=True)]

    best_loss = np.inf
    for log in logs:
        directory = log.parent

        df = pd.read_csv(
            log, header=0, index_col=None, skipinitialspace=True)
        if len(df) == 0:
            print(f"No data: {log}")
            continue
        index_min = df['validation_loss'].idxmin()

        best_epoch = df['epoch'].iloc[index_min]
        validation_loss = df['validation_loss'].iloc[index_min]

        if validation_loss < best_loss:
            model_file = directory / f"snapshot_epoch_{best_epoch}.pth"
            best_loss = validation_loss

    shutil.copyfile(model_file, output_directory / 'model')
    shutil.copyfile(
        directory / 'settings.yml', output_directory / 'settings.yml')

    print(model_name)
    print(f"best_model: {model_file}")
    print(
        f"best_epoch: {best_epoch}, "
        f"best_loss: {validation_loss:3e}")
    print('--')
    return


def deploy_model_mppde(model_name, model_root_directory, output_directory):
    model_root_path = pathlib.Path(model_root_directory)
    model_file = model_root_path / 'model.pt'
    shutil.copyfile(model_file, output_directory / 'model.pt')

    print(f"best_model: {model_file}")
    print('--')
    return


if __name__ == '__main__':
    main()
