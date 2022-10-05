import argparse
from distutils.util import strtobool
import pathlib
import re

import femio
import numpy as np
import pandas as pd
import siml

import stats


REQUIRED_FILE_NAME = 'predicted_nodal_U_step40.npy'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'predicted_base_directory',
        type=pathlib.Path,
        help='Directory containing predicted data')
    parser.add_argument(
        'interim_base_directory',
        type=pathlib.Path,
        help='The corresponding root interim directory')
    parser.add_argument(
        '-p', '--preprocessors-pkl',
        type=pathlib.Path,
        default=None,
        help='Pretrained directory name')
    parser.add_argument(
        '-f', '--force-renew',
        type=strtobool,
        default=0,
        help='If True, force renew data')
    parser.add_argument(
        '-s', '--write-steps',
        type=strtobool,
        default=0,
        help='If True, write files stepwise')
    args = parser.parse_args()

    if args.preprocessors_pkl is not None:
        converter = siml.prepost.Converter(args.preprocessors_pkl)
    df = pd.read_csv(
        args.predicted_base_directory / 'prediction.csv', header=0,
        index_col=0, skipinitialspace=True)

    data_directories = sorted([
        f.parent for f
        in args.predicted_base_directory.glob(f"**/{REQUIRED_FILE_NAME}")])

    list_dict = []
    stats_file = args.predicted_base_directory / 'stats.csv'
    with open(stats_file, 'w') as f:
        f.write(
            'directory,n_node,n_dirichlet_u_node,n_dirichlet_p_node,'
            'mse_u,stderror_u,'
            'mse_p,stderror_p,'
            'mse_dirichlet_u,stderror_dirichlet_u,'
            'mse_dirichlet_p,stderror_dirichlet_p,'
            'prediction_time\n'
        )
    for data_directory in data_directories:
        fem_data = process_one_directory(data_directory, args, converter)
        relative_directory = data_directory.relative_to(
            args.predicted_base_directory)
        if 'transform' in str(data_directory):
            directory = '../data/fluid/transformed/preprocessed/test/' \
                f"{relative_directory}"
        else:
            directory = f"../data/fluid/preprocessed/test/{relative_directory}"
        record = df.loc[directory]
        prediction_time = record['prediction_time']
        graph_creation_time = record['graph_creation_time']

        loss_dict = stats.calculate_single_loss_fluid(fem_data)
        loss_dict.update({
            'prediction_time': prediction_time,
            'graph_creation_time': graph_creation_time,
        })
        list_dict.append(loss_dict)

        with open(stats_file, 'a') as f:
            f.write(
                f"{directory},"
                f"{loss_dict['n_node']},"
                f"{loss_dict['n_dirichlet_u_node']},"
                f"{loss_dict['n_dirichlet_p_node']},"
                f"{loss_dict['mse_u']},"
                f"{loss_dict['stderror_u']},"
                f"{loss_dict['mse_p']},"
                f"{loss_dict['stderror_p']},"
                f"{loss_dict['mse_dirichlet_u']},"
                f"{loss_dict['stderror_dirichlet_u']},"
                f"{loss_dict['mse_dirichlet_p']},"
                f"{loss_dict['stderror_dirichlet_p']},"
                f"{prediction_time}\n"
            )
    print(f"Stats written in: {stats_file}")

    global_stats_file = args.predicted_base_directory / 'global_stats.csv'
    global_mse_u, global_std_error_u = stats.calculate_global_stats(
        list_dict, 'u')
    global_mse_p, global_std_error_p = stats.calculate_global_stats(
        list_dict, 'p')
    global_mse_dirichlet_u, global_std_error_dirichlet_u \
        = stats.calculate_global_stats(list_dict, 'dirichlet_u')
    global_mse_dirichlet_p, global_std_error_dirichlet_p \
        = stats.calculate_global_stats(list_dict, 'dirichlet_p')
    mean_time, std_error_time, mean_time_per_node, std_error_time_per_node \
        = stats.calculate_time(list_dict)
    graph_mean_time, graph_std_error_time, graph_mean_time_per_node, \
        graph_std_error_time_per_node = stats.calculate_time(
            list_dict, key='graph_creation_time')

    print('--')
    with open(global_stats_file, 'w') as f:
        print(
            f"mse_u: {global_mse_u:.5e} +/- {global_std_error_u:5e}")
        print(
            f"mse_p: {global_mse_p:.5e} +/- {global_std_error_p:5e}")
        print(
            f"mse_dirichlet_u: {global_mse_dirichlet_u:.5e} +/- "
            f"{global_std_error_dirichlet_u:5e}")
        print(
            f"mse_dirichlet_p: {global_mse_dirichlet_p:.5e} +/- "
            f"{global_std_error_dirichlet_p:5e}")
        print(
            f"prediction_time: {mean_time:.5e} +/- "
            f"{std_error_time:.5e}")
        print(
            f"prediction_time_per_node: {mean_time_per_node:.5e} +/- "
            f"{std_error_time_per_node:.5e}")
        print(
            f"graph_creation_time: {graph_mean_time:.5e} +/- "
            f"{graph_std_error_time:.5e}")
        print(
            'graph_creation_time_per_node: '
            f"{graph_mean_time_per_node:.5e} +/- "
            f"{std_error_time_per_node:.5e}")

        f.write(f"global_mse_u,{global_mse_u}\n")
        f.write(f"global_std_error_u,{global_std_error_u}\n")
        f.write(f"global_mse_p,{global_mse_p}\n")
        f.write(f"global_std_error_p,{global_std_error_p}\n")

        f.write(f"global_mse_dirichlet_u,{global_mse_dirichlet_u}\n")
        f.write(
            f"global_std_error_dirichlet_u,{global_std_error_dirichlet_u}\n")
        f.write(f"global_mse_dirichlet_p,{global_mse_dirichlet_p}\n")
        f.write(
            f"global_std_error_dirichlet_p,{global_std_error_dirichlet_p}\n")

        f.write(f"global_mean_prediction_time,{mean_time}\n")
        f.write(f"global_std_error_prediction_time,{std_error_time}\n")
        f.write(
            'global_mean_prediction_time_per_node,'
            f"{mean_time_per_node}\n")
        f.write(
            'global_std_error_prediction_time_per_node,'
            f"{std_error_time_per_node}\n")

        f.write(f"global_mean_graph_time,{graph_mean_time}\n")
        f.write(f"global_std_error_graph_time,{graph_std_error_time}\n")
        f.write(
            'global_mean_graph_time_per_node,'
            f"{graph_mean_time_per_node}\n")
        f.write(
            'global_std_error_graph_time_per_node,'
            f"{graph_std_error_time_per_node}\n")

    print(f"Global stats written in: {global_stats_file}")


def process_one_directory(data_directory, args, converter):
    print(f"--\nProcessing: {data_directory}")
    relative_path = data_directory.relative_to(args.predicted_base_directory)
    interim_path = args.interim_base_directory / relative_path
    if not interim_path.is_dir():
        raise ValueError(f"Interim path {interim_path} does not exist")

    fem_data = femio.read_directory(
        'polyvtk', interim_path)
    npy_files = list(data_directory.glob('*step*.npy'))
    for npy_file in npy_files:
        data = np.load(npy_file)
        if args.preprocessors_pkl is not None:
            data = converter.converters[
                npy_file.stem
                .replace('predicted_', '').replace('answer_', '')
                .replace('input_', '')
            ].inverse(data)
        fem_data.nodal_data.update_data(fem_data.nodes.ids, {
            f"{npy_file.stem}": data})

    fem_data.write(
        'polyvtk', data_directory / 'mesh.vtu', overwrite=args.force_renew)

    if args.write_steps:
        max_step = np.max([
            int(m.groups()[0]) for m
            in [re.search(r'_step(\d+).npy$', str(f)) for f in npy_files]
            if m is not None])
        for i in range(1, max_step + 1):
            step_fem_data = femio.FEMData(
                nodes=fem_data.nodes, elements=fem_data.elements,
                elemental_data=fem_data.elemental_data)
            keys = [
                f"answer_nodal_U_step{i}",
                f"answer_nodal_p_step{i}",
                f"predicted_nodal_U_step{i}",
                f"predicted_nodal_p_step{i}",
            ]
            for key in keys:
                if key in fem_data.nodal_data:
                    step_fem_data.nodal_data[key.replace(f"_step{i}", '')] \
                        = fem_data.nodal_data[key]
                else:
                    continue
            step_fem_data.write(
                'polyvtk', data_directory / f"{i}.vtu",
                overwrite=args.force_renew)

    return fem_data


if __name__ == '__main__':
    main()
