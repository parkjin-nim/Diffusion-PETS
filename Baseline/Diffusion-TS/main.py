import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='artifacts', 
                        help='directory to save the results')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Optional exact directory for run outputs.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Optional checkpoint base; Diffusion-TS appends _<seq_len>.')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--task', choices=['uncond', 'imputation', 'forecasting'], default=None,
                        help='Experiment task. Required during task-specific training.')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Optional run subdirectory, for example seed_42.')
    parser.add_argument('--milestone', type=int, default=10)
    parser.add_argument('--resume_milestone', type=int, default=0,
                        help='Resume training from an exact compatible checkpoint milestone.')

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  

    args = parser.parse_args()
    if not args.name:
        parser.error('--name is required')
    if not args.config_file:
        parser.error('--config_file is required')

    inferred_task = 'uncond' if args.sample == 0 else ('imputation' if args.mode == 'infill' else 'forecasting')
    task = args.task or inferred_task
    if not args.train and task != inferred_task:
        parser.error(f'--task {task} conflicts with the requested sampling mode ({inferred_task})')
    args.task = task

    run_parts = [args.output, task, args.name]
    if args.run_id:
        run_parts.append(args.run_id)
    default_save_dir = os.path.join(*run_parts)
    args.save_dir = os.path.abspath(args.save_dir or default_save_dir)
    args.checkpoint_dir = os.path.abspath(
        args.checkpoint_dir or os.path.join(args.save_dir, 'checkpoints')
    )

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    # ``configs`` belongs to Diffusion-PETS/TimesNet.  Removing it at the
    # adapter boundary keeps the copied experiment YAMLs usable without
    # changing the original Diffusion-TS model implementation.
    model_params = config['model']['params']
    model_params.pop('configs', None)
    if model_params['seq_length'] != 128:
        raise ValueError('Comparable experiments support seq_length=128 only.')
    if args.sample == 1 and args.mode == 'predict' and args.pred_len != 64:
        raise ValueError('Forecasting experiments require --pred_len 64.')
    if args.sample == 1 and args.mode == 'infill' and args.missing_ratio != 0.75:
        raise ValueError('Imputation experiments require --missing_ratio 0.75.')

    # Trainer remains untouched and retains its original ``_<seq_len>`` suffix.
    config['solver']['results_folder'] = args.checkpoint_dir

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        if args.resume_milestone:
            trainer.load(args.resume_milestone, verbose=True)
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        seq_length = config['model']['params']['seq_length']
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}_{seq_length}.npy'), samples)
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()
