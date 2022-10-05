import argparse
from distutils.util import strtobool
import glob
import multiprocessing as multi
import pathlib
import re
import sys

import femio
import numpy as np
from scipy.stats import special_ortho_group
import siml


VTU_NAME = 'internal.vtu'
RANK0_VARIABLE_NAMES = ['p']
RANK1_VARIABLE_NAMES = ['U']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root_raw_directory',
        type=pathlib.Path,
        help='Root directory which contains raw data')
    parser.add_argument(
        'root_output_directory',
        type=pathlib.Path,
        help='Output directory name')
    parser.add_argument(
        '-n', '--n-transformation',
        type=int,
        default=3,
        help='The number of transformation performed per sample [3]')
    parser.add_argument(
        '-r', '--required-file-name',
        type=str,
        default='log.icoFoam',
        help='Required file name to search for input data [log.icoFoam]')
    parser.add_argument(
        '-t', '--max-time',
        type=int,
        default=51,
        help='Max time step to be considered [51]')
    args = parser.parse_args()

    raw_data_directories = [
        p.parent.parent for p
        in args.root_raw_directory.glob(f"**/{args.required_file_name}")]
    transformer = Transformer(
        args.root_raw_directory, args.root_output_directory,
        args.n_transformation, max_time=args.max_time)
    with multi.Pool() as pool:
        pool.map(transformer.transform, raw_data_directories)
    return


class Transformer:

    def __init__(
            self, root_raw_directory, root_output_directory,
            n_transformation, max_time):
        self.root_raw_directory = root_raw_directory
        self.root_output_directory = root_output_directory
        self.n_transformation = n_transformation
        self.max_time = max_time + 1
        return

    def transform(self, data_directory):
        print(f"Processing: {data_directory}")
        coarse_files = sorted(list(
            data_directory.glob(f"**/coarse/**/{VTU_NAME}")))
        if len(coarse_files) != 1:
            raise ValueError(f"Invalid coarse files: {len(coarse_files)}")
        candidate_fine_files = list(
            data_directory.glob(f"**/fine/**/{VTU_NAME}"))
        filtered_fine_files = self._sort_fine_files(
            candidate_fine_files)[:self.max_time]
        vtu_files = coarse_files + filtered_fine_files
        for i in range(self.n_transformation):
            rotation_matrix = np.eye(3)
            rotation_matrix_2d = special_ortho_group.rvs(2)
            rotation_matrix[:2, :2] = rotation_matrix_2d
            translation_vector = np.random.rand(1, 3)
            translation_vector[0, -1] = 0.
            output_directory_base = self.root_output_directory \
                / data_directory.relative_to(self.root_raw_directory) \
                / f"transform_{i}"
            for vtu_file in vtu_files:
                rela_path = vtu_file.parent.relative_to(data_directory)
                output_directory = \
                    output_directory_base / rela_path
                self._transform_single_file(
                    vtu_file, output_directory,
                    rotation_matrix, translation_vector)
        (output_directory_base / 'fine/log.icoFoam').touch()
        np.save(
            output_directory_base / 'fine/rotation_matrix.npy',
            rotation_matrix)
        np.save(
            output_directory_base / 'fine/translation_vector.npy',
            translation_vector)
        return

    def _sort_fine_files(self, fine_files):
        times = [
            int(re.search('fine_(\d+)/', str(f)).groups()[0])
            for f in fine_files]
        sorted_files = [fine_files[t] for t in np.argsort(times)]
        return sorted_files

    def _transform_single_file(
            self, vtu_file, output_directory,
            rotation_matrix, translation_vector):
        fem_data = femio.read_files('polyvtk', vtu_file)

        fem_data.nodes.data = np.einsum(
            'kl,il->ik', rotation_matrix, fem_data.nodes.data) \
            + translation_vector
        fem_data.nodal_data['NODE'].data = fem_data.nodes.data
        for key in fem_data.nodal_data.keys():
            if key in RANK1_VARIABLE_NAMES:
                fem_data.nodal_data[key].data = np.einsum(
                    'kl,il->ik',
                    rotation_matrix, fem_data.nodal_data[key].data)
        for key in fem_data.elemental_data.keys():
            if key in RANK1_VARIABLE_NAMES:
                fem_data.elemental_data[key].data = np.einsum(
                    'kl,il->ik',
                    rotation_matrix, fem_data.elemental_data[key].data)

        fem_data.write('polyvtk', output_directory / VTU_NAME)
        return


if __name__ == '__main__':
    main()
