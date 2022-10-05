import argparse
from distutils.util import strtobool
import pathlib

import siml
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'settings_yaml',
        type=pathlib.Path,
        help='YAML file name of settings.')
    parser.add_argument(
        '-o', '--out-dir',
        type=pathlib.Path,
        default=None,
        help='Output directory name')
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=None,
        help='If fed, sed batch size')
    parser.add_argument(
        '-d', '--data-parallel',
        type=strtobool,
        default=0,
        help='If True, perform data parallelism [False]')
    parser.add_argument(
        '-g', '--gpu-id',
        type=int,
        default=-1,
        help='GPU ID [-1, meaning CPU]')
    parser.add_argument(
        '-r', '--restart-dir',
        type=pathlib.Path,
        default=None,
        help='Restart directory name')
    parser.add_argument(
        '-p', '--pretrained-directory',
        type=pathlib.Path,
        default=None,
        help='Pretrained directory name')
    parser.add_argument(
        '-l', '--lr',
        type=float,
        default=None,
        help='Learning rate')
    parser.add_argument(
        '-w', '--weight-decay',
        type=float,
        default=None,
        help='Weight decay')
    parser.add_argument(
        '-c', '--continue-training',
        type=strtobool,
        default=0,
        help='If True, continue training after stopping with decreasing lr')
    parser.add_argument(
        '-f', '--lr-factor',
        type=float,
        default=.5,
        help='Factor to be multiplied to lr when continuing')
    args = parser.parse_args()
    suffixes = []

    main_setting = siml.setting.MainSetting.read_settings_yaml(
        args.settings_yaml)
    main_setting.trainer.gpu_id = args.gpu_id
    if args.restart_dir is not None:
        main_setting.trainer.restart_directory = args.restart_dir
        suffixes.append(f"restart_{args.restart_dir}")
    if args.pretrained_directory is not None:
        main_setting.trainer.pretrain_directory = args.pretrained_directory
        suffixes.append(f"pretrain_{args.pretrained_directory}")
    if args.batch_size is not None:
        main_setting.trainer.batch_size = args.batch_size
        main_setting.trainer.validation_batch_size = args.batch_size
        suffixes.append(f"batchsize{args.batch_size}")
    if args.data_parallel:
        main_setting.trainer.data_parallel = args.data_parallel
        gpu_count = torch.cuda.device_count()
        original_batch_size = main_setting.trainer.batch_size
        main_setting.trainer.batch_size = original_batch_size * gpu_count
        main_setting.trainer.validation_batch_size \
            = main_setting.trainer.batch_size
        main_setting.trainer.num_workers = main_setting.trainer.num_workers \
            * gpu_count
        print(f"Batch size: {original_batch_size} x {gpu_count} GPUs")
    if args.lr is not None:
        main_setting.trainer.optimizer_setting['lr'] = args.lr
        suffixes.append(f"lr{args.lr:.3e}")
    if args.weight_decay is not None:
        main_setting.trainer.optimizer_setting[
            'weight_decay'] = args.weight_decay
        suffixes.append(f".decay{args.weight_decay}")

    if len(suffixes) > 0:
        main_setting.trainer.suffix = '_'.join(suffixes)
    if args.out_dir is None:
        main_setting.trainer.update_output_directory()
    else:
        main_setting.trainer.output_directory = args.out_dir

    trainer = siml.trainer.Trainer(main_setting)
    trainer.train()

    original_output_directory = trainer.setting.trainer.output_directory
    if args.continue_training:
        for continue_count in range(1, 10):
            print(f"--\nStart continuation: {continue_count}")
            main_setting.trainer.pretrain_directory = \
                trainer.setting.trainer.output_directory
            main_setting.trainer.optimizer_setting['lr'] = \
                trainer.setting.trainer.optimizer_setting['lr'] \
                * args.lr_factor
            main_setting.trainer.output_directory = pathlib.Path(
                f"{original_output_directory}"
                f"_cont{continue_count}"
                f"_lr{main_setting.trainer.optimizer_setting['lr']:.3e}")
            trainer = siml.trainer.Trainer(main_setting)
            trainer.train()
    return


if __name__ == '__main__':
    main()
