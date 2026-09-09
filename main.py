import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config

from types import SimpleNamespace


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
                        help='Optional exact directory for model checkpoints.')
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
    parser.add_argument('--resume_step', type=int, default=None,
                        help='Resume training from an exact step checkpoint.')

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  
    
    # trace config
    parser.add_argument('--save_trace', type=int, default=0, choices=[0, 1])

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
    model_params = config['model']['params']
    configs = model_params.get('configs', {})
    configs.setdefault('top_k', 7)
    configs.setdefault('d_ff', model_params['d_model'] * 2)
    configs.setdefault('num_kernels', 3)
    model_params['configs'] = configs

    if model_params['seq_length'] != 128:
        # The final paper evaluates true-unconditional PETS-U at 64/128/256.
        # Anti-Anchor is installed only by the frozen PETS-U sampler, never in
        # this training entry point.
        experiment = config.get('experiment', {})
        cross_length_marker = (
            experiment.get('marker')
            in {'PETS_U_FINAL_R2', 'PETS_U_BF_AA_AA_CROSS_DATASET_LENGTH_V1'}
        )
        cross_length_protocol = (
            args.task == 'uncond'
            and model_params['seq_length'] in (64, 256)
            and configs.get('period_candidate_scope') == 'batch_shared'
            and configs.get('period_fft_source') == 'full'
            and int(configs.get('period_fft_history_length', -1))
                == int(model_params['seq_length'])
            and experiment.get('training_aggregation') == 'FULL'
            and bool(experiment.get('true_unconditional'))
        )
        if not (cross_length_marker and cross_length_protocol):
            raise ValueError(
                'Optimized experiments support seq_length=128 only unless '
                'the audited true-unconditional cross-length protocol is used.'
            )
    if args.sample == 1 and args.mode == 'predict' and args.pred_len != 64:
        raise ValueError('Forecasting experiments require --pred_len 64.')
    if args.sample == 1 and args.mode == 'infill' and args.missing_ratio != 0.75:
        raise ValueError('Imputation experiments require --missing_ratio 0.75.')

    logger = Logger(args)
    logger.save_config(config)
    model_params['configs'] = SimpleNamespace(**configs)
    # 모델 인스턴스화   
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-capable GPU is required for Diffusion-PETS training and sampling.')
    model = instantiate_from_config(config['model']).cuda()
    print(f"Model class: {type(model)}")

    # 모델에 TimesBlock이 포함되어 있는지 확인하는 코드 추가
    if hasattr(model, "model"):
        if hasattr(model.model, "check_timesblock_activation"):
            print("Checking TimesBlock activation in model.model:")
            model.model.check_timesblock_activation()
        else:
            print("The model.model does not have check_timesblock_activation method.")
    else:
        print("The model does not contain a sub-model named 'model'.")
        
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        # 샘플링: 조건부
        test_dataloader_info = build_dataloader_cond(config, args)
    
    # 샘플링: 무조건부
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)
    
    # --train이 True인 경우 훈련을 시작하고, 그렇지 않으면 모델을 로드하여 샘플링을 수행합니다.
    if args.train:
        if args.resume_step is not None:
            trainer.load(args.resume_step, verbose=True)
        trainer.train()
    # --train이 False이고 --sample이 1인 경우 조건부 샘플링을 수행합니다.
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        # 
        # Preserve the user-requested draw seed. Checkpoint RNG restoration is
        # only for an exact training resume, not for independent sampling.
        trainer.load(args.milestone, restore_rng=False)
        # 조건부 샘플링은 10%의 데이터만 가져온다 (proportion이 0.9로 설정되어 있기 때문). 따라서 dataloader에서 가져오는 데이터는 전체 데이터의 10%에 해당하는 샘플들입니다.
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        # 
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        

        # 조건부 샘플링: dataloader에서 배치 단위로 데이터를 가져와서 모델에 입력하여 샘플링을 수행합니다.
        #samples, *_ = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
        # 추가: --save_trace가 1인 경우 restore_with_trace를 사용하여 샘플링 과정에서 trace를 저장합니다. trace는 모델이 샘플링하는 동안의 내부 상태를 기록한 것으로, 나중에 분석에 사용할 수 있습니다.

        if args.save_trace == 1:
            samples, reals, masks, trace = trainer.restore_with_trace(
                dataloader,
                [dataset.window, dataset.var_num],
                coef,
                stepsize,
                sampling_steps,
                trace_steps=None,   # let sampler decide correctly
                max_batches=1
            )

            torch.save(
                trace,
                os.path.join(
                    args.save_dir,
                    f'conditional_trace_{args.name}_{config["model"]["params"]["seq_length"]}.pt'
                )
            )
        else:
            samples, *_ = trainer.restore(
                dataloader,
                [dataset.window, dataset.var_num],
                coef,
                stepsize,
                sampling_steps
            )
        ### 추가: --save_trace가 1인 경우 

        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)

        # 샘플링된 결과를 저장합니다. 파일 이름에 시퀀스 길이를 포함하도록 수정합니다.
        seq_length = config['model']['params']['seq_length'] 
        np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}_{seq_length}.npy'), samples)
    # --train이 False이고 --sample이 0인 경우 무조건부 샘플링을 수행합니다.
    else:
        trainer.load(args.milestone, restore_rng=False)
        # 무조건부 샘플링은 전체 데이터셋에서 샘플을 생성하기 때문에 dataloader에서 가져오는 데이터는 전체 데이터셋에 해당합니다. 
        # 따라서 dataloader에서 가져오는 데이터는 전체 데이터셋의 모든 샘플을 포함합니다.
        dataset = dataloader_info['dataset']

        # 추가: batch가 너무커서 cuda 메모리 에러가 난다. sequence length가 512 이상이면 size_every를 64로, 그렇지 않으면 2001로 설정한다. 
        # size_every는 batch크기다. 모델이 한 번에 처리할 수 있는 최대 시퀀스 길이에 따라 batch 크기를 조정하여 메모리 문제를 방지한다. 
        seq_length = config['model']['params']['seq_length'] 
        size_every_val = 64 if seq_length >= 512 else 2001
        # 무조건부 샘플링: dataloader에서 배치 단위로 데이터를 가져와서 모델에 입력하여 샘플링을 수행합니다
        #samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])  
        samples = trainer.sample(num=len(dataset), size_every=size_every_val, shape=[dataset.window, dataset.var_num])        
        
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()
