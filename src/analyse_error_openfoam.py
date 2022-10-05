import argparse
from distutils.util import strtobool
import pathlib
import re

import femio
import numpy as np
from scipy.spatial import KDTree

import stats


REQUIRED_FILE_NAME = 'predicted_nodal_U_step40.npy'
REFERENCE_NAME = 'fine'
REFERENCE_STEP = 4000
N_DATA = 25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'template_name',
        type=str,
        help='The name of the template')
    parser.add_argument(
        'benchmark_base_directory',
        type=pathlib.Path,
        help='Base directory containing benchmark data')
    parser.add_argument(
        'answer_base_directory',
        type=pathlib.Path,
        help='The corresponding root raw directory')
    parser.add_argument(
        '-f', '--force-renew',
        type=strtobool,
        default=0,
        help='If True, overwrite existing data [False]')
    args = parser.parse_args()

    output_base_directory = args.benchmark_base_directory / args.template_name
    output_base_directory.mkdir(exist_ok=True)
    stats_file = output_base_directory / 'stats.csv'

    vtk_directories = sorted([
        d for d in args.benchmark_base_directory.glob('**/VTK')
        if args.template_name in str(d)])
    list_dict = []
    with open(stats_file, 'w') as f:
        f.write(
            'directory,n_node,'
            'mse_u,stderror_u,'
            'mse_p,stderror_p,'
            'prediction_time\n'
        )

    for vtk_directory in vtk_directories:
        time_file = vtk_directory.parent / 'time.txt'
        if not time_file.is_file():
            raise ValueError(f"Time file not found: {time_file}")
        prediction_time = np.loadtxt(time_file)
        if prediction_time.size == 0:
            raise ValueError(f"Not finished: {vtk_directory}")

        fem_data = process_one_directory(vtk_directory, args)
        relative_directory = vtk_directory.relative_to(
            args.benchmark_base_directory).parent
        directory = f"data/fluid/preprocessed/test/{relative_directory}"

        loss_dict = stats.calculate_single_loss_fluid(
            fem_data, analyse_dirichlet=False)
        loss_dict.update({
            'prediction_time': prediction_time,
        })
        list_dict.append(loss_dict)

        with open(stats_file, 'a') as f:
            f.write(
                f"{directory},"
                f"{loss_dict['n_node']},"
                f"{loss_dict['mse_u']},"
                f"{loss_dict['stderror_u']},"
                f"{loss_dict['mse_p']},"
                f"{loss_dict['stderror_p']},"
                f"{prediction_time}\n"
            )
    print(f"Stats written in: {stats_file}")

    if len(list_dict) != N_DATA:
        raise ValueError(
            f"# of data differs: {len(list_dict)} vs {N_DATA}")

    global_stats_file = output_base_directory / 'global_stats.csv'
    global_mse_u, global_std_error_u = stats.calculate_global_stats(
        list_dict, 'u')
    global_mse_p, global_std_error_p = stats.calculate_global_stats(
        list_dict, 'p')
    mean_time, std_error_time, mean_time_per_node, std_error_time_per_node \
        = stats.calculate_time(list_dict)

    print('--')
    with open(global_stats_file, 'w') as f:
        print(
            f"mse_u: {global_mse_u:.5e} +/- {global_std_error_u:5e}")
        print(
            f"mse_p: {global_mse_p:.5e} +/- {global_std_error_p:5e}")
        print(
            f"prediction_time: {mean_time:.5e} +/- "
            f"{std_error_time:.5e}")
        print(
            f"prediction_time_per_node: {mean_time_per_node:.5e} +/- "
            f"{std_error_time_per_node:.5e}")

        f.write(f"global_mse_u,{global_mse_u}\n")
        f.write(f"global_std_error_u,{global_std_error_u}\n")
        f.write(f"global_mse_p,{global_mse_p}\n")
        f.write(f"global_std_error_p,{global_std_error_p}\n")

        f.write(f"global_mean_prediction_time,{mean_time}\n")
        f.write(f"global_std_error_prediction_time,{std_error_time}\n")
        f.write(
            'global_mean_prediction_time_per_node,'
            f"{mean_time_per_node}\n")
        f.write(
            'global_std_error_prediction_time_per_node,'
            f"{std_error_time_per_node}\n")

    print(f"Global stats written in: {global_stats_file}")

    return


def process_one_directory(vtk_directory, args):
    print(f"--\nProcessing: {vtk_directory}")

    subdirectories = [
        d.parent for d in vtk_directory.glob('**/internal.vtu')]
    time_indices = [
        int(re.search(r'_(\d+)$', str(d.name)).groups()[0])
        for d in subdirectories]
    selected_directory = subdirectories[np.argmax(time_indices)]
    fem_data = femio.read_files(
        'polyvtk', selected_directory / 'internal.vtu')

    stem_path = selected_directory.relative_to(
        args.benchmark_base_directory).parent
    answer_directory = (
        args.answer_base_directory
        / str(stem_path).replace(args.template_name, REFERENCE_NAME)) \
        / f"{REFERENCE_NAME}_{REFERENCE_STEP}"
    answer_fem_data = femio.read_files(
        'polyvtk', answer_directory / 'internal.vtu')

    answer_x = answer_fem_data.nodes.data[:, :2]
    benchmark_x = fem_data.nodes.data[:, :2]
    print('nearest neighbor search')
    tree = KDTree(answer_x)
    _, indices = tree.query(benchmark_x)
    searched_u = answer_fem_data.nodal_data.get_attribute_data('U')[indices]
    searched_p = answer_fem_data.nodal_data.get_attribute_data('p')[indices]

    fem_data.nodal_data.update_data(
        fem_data.nodes.ids, {
            'answer_nodal_U_step40': searched_u,
            'answer_nodal_p_step40': searched_p,
            'predicted_nodal_U_step40':
            fem_data.nodal_data.get_attribute_data('U'),
            'predicted_nodal_p_step40':
            fem_data.nodal_data.get_attribute_data('p'),
        })
    # fem_data.write(
    #     'polyvtk', vtk_directory / 'mesh.vtu', overwrite=args.force_renew)

    return fem_data


if __name__ == '__main__':
    main()
