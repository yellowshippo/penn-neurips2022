import os
import pathlib

import numpy as np


def fluid_analyse_error(results, prediction_root_directory):
    list_dict = []
    stats_file = prediction_root_directory / 'stats.csv'
    with open(stats_file, 'w') as f:
        f.write(
            'directory,n_node,n_dirichlet_u_node,n_dirichlet_p_node,'
            'mse_u,stderror_u,'
            'mse_p,stderror_p,'
            'mse_dirichlet_u,stderror_dirichlet_u,'
            'mse_dirichlet_p,stderror_dirichlet_p,'
            'prediction_time\n'
        )

    for result in results:
        directory = result['data_directory']
        prediction_time = result['inference_time']

        fem_data = result['fem_data']
        loss_dict = calculate_single_loss_fluid(fem_data)
        loss_dict.update({'prediction_time': prediction_time})
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

    global_stats_file = prediction_root_directory / 'global_stats.csv'
    global_mse_u, global_std_error_u = calculate_global_stats(
        list_dict, 'u')
    global_mse_p, global_std_error_p = calculate_global_stats(
        list_dict, 'p')
    global_mse_dirichlet_u, global_std_error_dirichlet_u \
        = calculate_global_stats(list_dict, 'dirichlet_u')
    global_mse_dirichlet_p, global_std_error_dirichlet_p \
        = calculate_global_stats(list_dict, 'dirichlet_p')
    mean_time, std_error_time, mean_time_per_node, std_error_time_per_node \
        = calculate_time(list_dict)

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

    print(f"Global stats written in: {global_stats_file}")
    return


def calculate_single_loss_fluid(fem_data, analyse_dirichlet=True):
    if 'answer_nodal_U_step40_reshaped' in fem_data.nodal_data:
        answer_u = fem_data.nodal_data.get_attribute_data(
            'answer_nodal_U_step40_reshaped')
        predicted_u = fem_data.nodal_data.get_attribute_data(
            'predicted_nodal_U_step40_reshaped')
        answer_p = fem_data.nodal_data.get_attribute_data(
            'answer_nodal_p_step40')
        predicted_p = fem_data.nodal_data.get_attribute_data(
            'predicted_nodal_p_step40')
    elif 'predicted_nodal_U_step40' in fem_data.nodal_data:
        answer_u = fem_data.nodal_data.get_attribute_data(
            'answer_nodal_U_step40')
        predicted_u = fem_data.nodal_data.get_attribute_data(
            'predicted_nodal_U_step40')
        answer_p = fem_data.nodal_data.get_attribute_data(
            'answer_nodal_p_step40')
        predicted_p = fem_data.nodal_data.get_attribute_data(
            'predicted_nodal_p_step40')
    else:
        raise ValueError(f"Invalid data format: {fem_data.nodal_data.keys()}")

    return_dict = {
        'n_node': len(fem_data.nodes),
        'mse_u': mse(predicted_u, answer_u),
        'stderror_u': std_error(predicted_u, answer_u),
        'mse_p': mse(predicted_p, answer_p),
        'stderror_p': std_error(predicted_p, answer_p),
        'answer_u': answer_u,
        'predicted_u': predicted_u,
        'answer_p': answer_p,
        'predicted_p': predicted_p,
    }

    if analyse_dirichlet:
        # Analyse Dirichlet error
        if 'input_nodal_boundary_U_reshaped' in fem_data.nodal_data:
            dirichlet_u = fem_data.nodal_data.get_attribute_data(
                'input_nodal_boundary_U_reshaped')
        else:
            dirichlet_u = fem_data.nodal_data.get_attribute_data(
                'nodal_boundary_U_reshaped')
        filter_dirichlet_u = ~np.isnan(dirichlet_u)
        answer_dirichlet_u = dirichlet_u[filter_dirichlet_u]
        predicted_dirichlet_u = predicted_u[filter_dirichlet_u]

        if 'input_nodal_boundary_p' in fem_data.nodal_data:
            dirichlet_p = fem_data.nodal_data.get_attribute_data(
                'input_nodal_boundary_p')
        else:
            dirichlet_p = fem_data.nodal_data.get_attribute_data(
                'nodal_boundary_p')
        filter_dirichlet_p = ~np.isnan(dirichlet_p)
        answer_dirichlet_p = dirichlet_p[filter_dirichlet_p]
        predicted_dirichlet_p = predicted_p[filter_dirichlet_p]

        return_dict.update({
            'n_dirichlet_u_node': np.sum(filter_dirichlet_u),
            'n_dirichlet_p_node': np.sum(filter_dirichlet_p),
            'mse_dirichlet_u': mse(
                predicted_dirichlet_u, answer_dirichlet_u),
            'stderror_dirichlet_u': std_error(
                predicted_dirichlet_u, answer_dirichlet_u),
            'mse_dirichlet_p': mse(
                predicted_dirichlet_p, answer_dirichlet_p),
            'stderror_dirichlet_p': std_error(
                predicted_dirichlet_p, answer_dirichlet_p),
            'answer_dirichlet_u': answer_dirichlet_u,
            'predicted_dirichlet_u': predicted_dirichlet_u,
            'answer_dirichlet_p': answer_dirichlet_p,
            'predicted_dirichlet_p': predicted_dirichlet_p,
        })

    return return_dict


def grad_analyse_error(results, prediction_root_directory):
    list_dict = []
    stats_file = prediction_root_directory / 'stats.csv'
    with open(stats_file, 'w') as f:
        f.write(
            'directory,n_node,n_surface_node,mse_grad,stderror_grad,'
            'mse_neumann,stderror_neumann,prediction_time\n'
        )

    for result in results:
        directory = result['data_directory']
        prediction_time = result['inference_time']

        fem_data = result['fem_data']
        loss_dict = calculate_single_loss_grad(fem_data, 'grad_phi_reshaped')
        loss_dict.update({'prediction_time': prediction_time})
        list_dict.append(loss_dict)

        with open(stats_file, 'a') as f:
            f.write(
                f"{directory},"
                f"{loss_dict['n_node']},"
                f"{loss_dict['n_surface_node']},"
                f"{loss_dict['mse_grad']},"
                f"{loss_dict['stderror_grad']},"
                f"{loss_dict['mse_neumann']},"
                f"{loss_dict['stderror_neumann']},"
                f"{prediction_time}\n"
            )
    print(f"Stats written in: {stats_file}")

    global_stats_file = prediction_root_directory / 'global_stats.csv'
    global_mse_grad, global_std_error_grad = calculate_global_stats(
        list_dict, 'grad')
    global_mse_neumann, global_std_error_neumann = calculate_global_stats(
        list_dict, 'neumann')
    mean_time, std_error_time, mean_time_per_node, std_error_time_per_node \
        = calculate_time(list_dict)

    print('--')
    with open(global_stats_file, 'w') as f:
        print(
            f"mse_grad: {global_mse_grad:.5e} +/- {global_std_error_grad:5e}")
        print(
            f"mse_neumann: {global_mse_neumann:.5e} +/- "
            f"{global_std_error_neumann:.5e}")
        print(
            f"prediction_time: {mean_time:.5e} +/- "
            f"{std_error_time:.5e}")
        print(
            f"prediction_time_per_node: {mean_time_per_node:.5e} +/- "
            f"{std_error_time_per_node:.5e}")

        f.write(f"global_mse_grad,{global_mse_grad}\n")
        f.write(f"global_std_error_grad,{global_std_error_grad}\n")
        f.write(f"global_mse_neumann,{global_mse_neumann}\n")
        f.write(f"global_std_error_neumann,{global_std_error_neumann}\n")
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


def calculate_single_loss_grad(fem_data, variable_name):
    answer_grad = fem_data.nodal_data.get_attribute_data(
        f"answer_{variable_name}")
    predicted_grad = fem_data.nodal_data.get_attribute_data(
        f"predicted_{variable_name}")

    # Analyse Neumann error
    normal = fem_data.nodal_data.get_attribute_data('normal')
    filter_surface = np.linalg.norm(normal, axis=1) > 1e-5
    answer_neumann = fem_data.nodal_data.get_attribute_data('neumann')[
        filter_surface]
    predicted_neumann = np.einsum('ij,ij->i', normal, predicted_grad)[
        filter_surface, None]
    return {
        'n_node': len(fem_data.nodes),
        'n_surface_node': np.sum(filter_surface),
        'mse_grad': mse(predicted_grad, answer_grad),
        'stderror_grad': std_error(predicted_grad, answer_grad),
        'mse_neumann': mse(predicted_neumann, answer_neumann),
        'stderror_neumann': std_error(predicted_neumann, answer_neumann),
        'answer_grad': answer_grad,
        'predicted_grad': predicted_grad,
        'answer_neumann': answer_neumann,
        'predicted_neumann': predicted_neumann,
    }


def ad_analyse_error(results, prediction_root_directory):
    list_dict = []
    stats_file = prediction_root_directory / 'stats.csv'
    with open(stats_file, 'w') as f:
        f.write(
            'directory,n_node,n_dirichlet_node,mse_t,stderror_t,'
            'mse_dirichlet_t,stderror_dirichlet_t,prediction_time\n'
        )

    for result in results:
        directory = result['data_directory']
        prediction_time = result['inference_time']

        fem_data = result['fem_data']
        loss_dict = calculate_single_loss_ad(fem_data, [
            'nodal_T_step25',
            'nodal_T_step50',
            'nodal_T_step75',
            'nodal_T_step100',
        ])
        loss_dict.update({'prediction_time': prediction_time})
        list_dict.append(loss_dict)

        with open(stats_file, 'a') as f:
            f.write(
                f"{directory},"
                f"{loss_dict['n_node']},"
                f"{loss_dict['n_dirichlet_node']},"
                f"{loss_dict['mse_t']},"
                f"{loss_dict['stderror_t']},"
                f"{loss_dict['mse_dirichlet_t']},"
                f"{loss_dict['stderror_dirichlet_t']},"
                f"{prediction_time}\n"
            )
    print(f"Stats written in: {stats_file}")

    global_stats_file = prediction_root_directory / 'global_stats.csv'
    global_mse_t, global_std_error_t = calculate_global_stats(
        list_dict, 't')
    global_mse_dirichlet_t, global_std_error_dirichlet_t \
        = calculate_global_stats(
            list_dict, 'dirichlet_t', predicted_concat_axis=1)
    mean_time, std_error_time, mean_time_per_node, std_error_time_per_node \
        = calculate_time(list_dict)

    print('--')
    with open(global_stats_file, 'w') as f:
        print(
            f"mse_t: {global_mse_t:.5e} +/- {global_std_error_t:5e}")
        print(
            f"mse_dirichlet_t: {global_mse_dirichlet_t:.5e} +/- "
            f"{global_std_error_dirichlet_t:.5e}")
        print(
            f"prediction_time: {mean_time:.5e} +/- "
            f"{std_error_time:.5e}")
        print(
            f"prediction_time_per_node: {mean_time_per_node:.5e} +/- "
            f"{std_error_time_per_node:.5e}")

        f.write(f"global_mse_t,{global_mse_t}\n")
        f.write(f"global_std_error_t,{global_std_error_t}\n")
        f.write(f"global_mse_dirichlet_t,{global_mse_dirichlet_t}\n")
        f.write(
            f"global_std_error_dirichlet_t,{global_std_error_dirichlet_t}\n")
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


def calculate_single_loss_ad(fem_data, variable_names):
    answer_t = np.stack(
        [
            fem_data.nodal_data.get_attribute_data(
                f"answer_{variable_name}")
            for variable_name in variable_names], axis=0)
    predicted_t = np.stack(
        [
            fem_data.nodal_data.get_attribute_data(
                f"predicted_{variable_name}")
            for variable_name in variable_names], axis=0)

    # Analyse Dirichlet error
    if 'input_nodal_boundary_T' in fem_data.nodal_data:
        dirichlet_t = fem_data.nodal_data.get_attribute_data(
            'input_nodal_boundary_T')
    else:
        dirichlet_t = fem_data.nodal_data.get_attribute_data(
            'nodal_boundary_T')
    filter_dirichlet_t = ~np.isnan(dirichlet_t)
    answer_dirichlet_t = dirichlet_t[filter_dirichlet_t]
    predicted_dirichlet_t = predicted_t[:, filter_dirichlet_t]

    return {
        'n_node': len(fem_data.nodes),
        'n_dirichlet_node': np.sum(filter_dirichlet_t),
        'mse_t': mse(predicted_t, answer_t),
        'stderror_t': std_error(predicted_t, answer_t),
        'mse_dirichlet_t': mse(predicted_dirichlet_t, answer_dirichlet_t),
        'stderror_dirichlet_t':
        std_error(predicted_dirichlet_t, answer_dirichlet_t),
        'answer_t': answer_t,
        'predicted_t': predicted_t,
        'answer_dirichlet_t': answer_dirichlet_t,
        'predicted_dirichlet_t': predicted_dirichlet_t,
    }


def determine_output_base(results):
    candidate = results[0]['output_directory']
    for result in results[1:]:
        candidate = os.path.commonpath([candidate, result['output_directory']])
    return pathlib.Path(candidate)


def calculate_global_stats(list_dict, key, predicted_concat_axis=0):
    answer = np.concatenate([d[f"answer_{key}"] for d in list_dict])
    pred = np.concatenate(
        [d[f"predicted_{key}"] for d in list_dict], axis=predicted_concat_axis)
    square_error = (pred - answer)**2
    mse = np.mean(square_error)
    std_error = np.std(square_error) / np.sqrt(len(answer))
    return mse, std_error


def calculate_time(list_dict, key='prediction_time'):
    times = np.array([d[key] for d in list_dict])
    n_sample = len(list_dict)
    n_nodes = np.array([d['n_node'] for d in list_dict])
    time_per_nodes = times / n_nodes
    return np.mean(times), np.std(times) / np.sqrt(n_sample), \
        np.mean(time_per_nodes), np.std(time_per_nodes) / np.sqrt(n_sample)


def mse(a, b):
    return np.mean((a - b)**2)


def std_error(a, b):
    return np.std((a - b)**2) / np.sqrt(len(a))
