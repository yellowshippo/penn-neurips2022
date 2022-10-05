
import argparse
import multiprocessing as multi
import pathlib
import random

import femio
import numpy as np
import siml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_directory',
        type=pathlib.Path,
        help='Output base direcoty')
    parser.add_argument(
        '-n',
        '--n-repetition',
        type=int,
        default=3,
        help='The number of repetition [3]')
    parser.add_argument(
        '-j',
        '--min-n_element',
        type=int,
        default=10,
        help='The minimum number of elements [10]')
    parser.add_argument(
        '-k',
        '--max-n_element',
        type=int,
        default=20,
        help='The maximum number of elements [20]')
    parser.add_argument(
        '-d',
        '--polynomial-degree',
        type=int,
        default=3,
        help='The number of polynomial degree to generate data [3]')
    parser.add_argument(
        '-p',
        '--max-process',
        type=int,
        default=None,
        help='If fed, set the maximum # of processes')
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=0,
        help='Random seed [0]')
    args = parser.parse_args()

    generator = GridDataGenerator(**vars(args))
    generator.generate()

    return


class GridDataGenerator:

    def __init__(
            self, output_directory, *,
            edge_length=1., n_repetition=3, seed=0,
            min_n_element=10, max_n_element=100,
            polynomial_degree=3, max_process=None):
        self.output_directory = pathlib.Path(output_directory)
        self.edge_length = edge_length
        self.n_repetition = n_repetition
        self.seed = seed
        self.min_n_element = min_n_element
        self.max_n_element = max_n_element
        self.polynomial_degree = polynomial_degree
        self.max_process = siml.util.determine_max_process(max_process)
        self.times = list(range(1, 11))

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        return

    def generate(self):
        """Create grid graph data.
        """

        with multi.Pool(self.max_process) as pool:
            pool.map(self._generate_one_data, list(range(self.n_repetition)))
        return

    def _generate_one_data(self, i_data):
        n_x_element = random.randint(
            self.min_n_element, self.max_n_element)
        n_y_element = random.randint(
            self.min_n_element, self.max_n_element)
        n_z_element = random.randint(
            self.min_n_element, self.max_n_element)
        dx = .1

        fem_data = femio.generate_brick(
            'hex', n_x_element, n_y_element, n_z_element,
            x_length=dx*n_x_element, y_length=dx*n_y_element,
            z_length=dx*n_z_element)

        target_dict_data, fem_data = self.add_data(fem_data)
        dict_data = self.extract_feature(fem_data, target_dict_data)

        output_directory = self.output_directory / str(i_data)
        self.save(output_directory, dict_data, fem_data)
        return

    def add_data(self, fem_data):
        center = np.mean(fem_data.nodes.data, axis=0)
        nodes = {
            'x': fem_data.nodes.data[:, 0] - center[0],
            'y': fem_data.nodes.data[:, 1] - center[1],
            'z': fem_data.nodes.data[:, 2] - center[2],
        }
        degrees = list(range(1, self.polynomial_degree + 1))
        polynomial_coefficients = {
            f"{component}_{degree}": 2 * np.random.rand() - 1
            for component in nodes.keys()
            for degree in degrees}
        raw_phi = np.sum([
            polynomial_coefficients[f"{component}_{degree}"]
            * nodes[component]**degree
            for component in nodes.keys()
            for degree in degrees], axis=0)
        scale = 1 / np.max(np.abs(raw_phi))
        phi = raw_phi * scale
        grad_phi = scale * np.sum([
            np.stack([
                degree * polynomial_coefficients[f"{component}_{degree}"]
                * nodes[component]**(degree - 1)
                for component in nodes.keys()], axis=-1)
            for degree in degrees], axis=0)

        dict_data = {
            'phi': phi[..., None],
            'grad_phi': grad_phi[..., None],
        }
        return dict_data, fem_data

    def extract_feature(self, fem_data, target_dict_data):

        nodal_adj = fem_data.calculate_adjacency_matrix_node()
        nodal_nadj = siml.prepost.normalize_adjacency_matrix(nodal_adj)

        node = fem_data.nodal_data.get_attribute_data('node')

        # Normal vector
        surface_fem_data = fem_data.to_surface()
        surface_incidence = surface_fem_data.calculate_incidence_matrix(
            order1_only=False)
        surface_normals = surface_fem_data.calculate_element_normals()
        surface_nodal_normals = femio.functions.normalize(
            surface_fem_data.convert_elemental2nodal(
                surface_normals, mode='effective',
                incidence=surface_incidence),
            keep_zeros=True)

        fem_normal = femio.FEMAttribute(
            'normal', fem_data.nodes.ids,
            np.zeros((len(fem_data.nodes.ids), 3)))
        fem_normal.loc[surface_fem_data.nodes.ids].data\
            = surface_nodal_normals
        fem_data.nodal_data.update({'normal': fem_normal})
        normal = fem_normal.data

        nodal_x_grad_hop1, nodal_y_grad_hop1, nodal_z_grad_hop1 \
            = fem_data.calculate_spatial_gradient_adjacency_matrices(
                'nodal', n_hop=1, moment_matrix=True,
                normals=normal, normal_weight=10.,
                consider_volume=False, adj=nodal_adj)
        inversed_moment_tensor = fem_data.nodal_data.get_attribute_data(
            'inversed_moment_tensors')[..., None]
        weighted_normal = fem_data.nodal_data.get_attribute_data(
            'weighted_surface_normals')[..., None]

        # Boundary condition
        neumann = np.einsum(
            'ij,ij->i',
            target_dict_data['grad_phi'][..., 0], normal)[..., None]
        directed_neumann = np.einsum(
            'ij,i->ij', normal, neumann[:, 0])[..., None]

        dict_data = {
            'neumann': neumann,
            'node': node,
            'nodal_adj': nodal_adj, 'nodal_nadj': nodal_nadj,
            'nodal_grad_x_1': nodal_x_grad_hop1,
            'nodal_grad_y_1': nodal_y_grad_hop1,
            'nodal_grad_z_1': nodal_z_grad_hop1,
            'inversed_moment_tensors_1': inversed_moment_tensor,
            'weighted_surface_normal_1': weighted_normal,
            'nodal_surface_normal': normal[..., None],
            'directed_neumann': directed_neumann,
        }
        dict_data.update(target_dict_data)

        return dict_data

    def save(self, output_directory, dict_data, fem_data):
        siml.prepost.save_dict_data(output_directory, dict_data)
        fem_data_to_save = siml.prepost.update_fem_data(
            fem_data, dict_data, allow_overwrite=True)
        fem_data_to_save.save(output_directory)
        fem_data_to_save.write('polyvtk', output_directory / 'mesh.vtu')
        (output_directory / 'converted').touch()
        return


if __name__ == '__main__':
    main()
