import os
import time
import copy
import csv
import json
import subprocess
import random
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.complete_config = copy.deepcopy(config)
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.dataloader = dataloader['dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger

        checkpoint_dir = getattr(args, 'checkpoint_dir', None)
        if checkpoint_dir is None:
            results_root = config['solver']['results_folder']
            checkpoint_dir = f'{results_root}_{model.seq_length}'
        self.results_folder = Path(checkpoint_dir)
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

        self.variant = config.get('fft_ablation', {}).get('variant', 'UNKNOWN')
        self.training_seed = int(config.get('fft_ablation', {}).get('seed', args.seed))
        self._telemetry_clock = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        telemetry_path = config['solver'].get(
            'telemetry_path',
            str(Path(args.save_dir) / 'training_telemetry.csv'),
        )
        self.telemetry_path = Path(telemetry_path)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry_columns = [
            'step', 'variant', 'seed', 'total_loss', 'diffusion_loss',
            'learning_rate',
            'encoder_k_logits', 'encoder_ema_k_logits', 'encoder_k_cont',
            'encoder_hard_k', 'encoder_raw_entropy_mean',
            'encoder_raw_shannon_effective_count_mean',
            'encoder_effective_entropy_mean',
            'encoder_effective_shannon_count_mean',
            'encoder_effective_inverse_hhi_count_mean',
            'encoder_maximum_effective_weight_mean',
            'encoder_removed_probability_mass_mean',
            'encoder_raw_probability_retained_mean',
            'encoder_raw_inverse_hhi_count_mean', 'encoder_raw_maximum_weight_mean',
            'encoder_raw_top1_top2_margin_mean', 'encoder_effective_top1_top2_margin_mean',
            'encoder_nearest_rounding_boundary_distance', 'encoder_lower_clamp_distance',
            'encoder_upper_clamp_distance', 'encoder_active_slot_count', 'encoder_inactive_slot_count',
            'encoder_effective_count_divided_by_hard_k', 'encoder_active_weight_ge_0.01_mean',
            'encoder_active_weight_ge_0.05_mean', 'encoder_active_weight_ge_0.10_mean',
            'encoder_k_logits_gradient_norm',
            'decoder_k_logits', 'decoder_ema_k_logits', 'decoder_k_cont',
            'decoder_hard_k', 'decoder_raw_entropy_mean',
            'decoder_raw_shannon_effective_count_mean',
            'decoder_effective_entropy_mean',
            'decoder_effective_shannon_count_mean',
            'decoder_effective_inverse_hhi_count_mean',
            'decoder_maximum_effective_weight_mean',
            'decoder_removed_probability_mass_mean',
            'decoder_raw_probability_retained_mean',
            'decoder_raw_inverse_hhi_count_mean', 'decoder_raw_maximum_weight_mean',
            'decoder_raw_top1_top2_margin_mean', 'decoder_effective_top1_top2_margin_mean',
            'decoder_nearest_rounding_boundary_distance', 'decoder_lower_clamp_distance',
            'decoder_upper_clamp_distance', 'decoder_active_slot_count', 'decoder_inactive_slot_count',
            'decoder_effective_count_divided_by_hard_k', 'decoder_active_weight_ge_0.01_mean',
            'decoder_active_weight_ge_0.05_mean', 'decoder_active_weight_ge_0.10_mean',
            'decoder_k_logits_gradient_norm',
            'candidate_fallback_count', 'step_seconds', 'peak_gpu_memory_mb',
        ]
        candidate_fields = [
            'unique_top1_periods_in_batch', 'top1_period_entropy',
            'top1_mode_period', 'top1_mode_fraction',
            'unique_candidate_vectors', 'candidate_vector_collision_rate',
            'mean_pairwise_candidate_jaccard',
        ]
        for role in ('encoder', 'decoder'):
            for field in candidate_fields:
                self.telemetry_columns.append(role + '_' + field)
        if not self.telemetry_path.exists():
            with self.telemetry_path.open('w', newline='') as handle:
                csv.DictWriter(handle, fieldnames=self.telemetry_columns).writeheader()
        metadata_path = self.telemetry_path.with_name('training_telemetry_metadata.json')
        metadata = {
            'interval_steps': self.log_frequency,
            'raw_entropy': '-sum(raw_probability * log(raw_probability + eps))',
            'effective_entropy': (
                '-sum(normalize(raw_probability * detached_hard_mask) * '
                'log(normalize(raw_probability * detached_hard_mask) + eps))'
            ),
            'effective_count': 'exp(effective_entropy)',
            'actual_latents': True,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')

    @staticmethod
    def _source_commit():
        try:
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=str(Path(__file__).resolve().parents[1]),
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return 'NOT_AVAILABLE'

    def _append_telemetry(
        self,
        total_loss,
        diffusion_loss,
        log_dict,
        k_gradient_norms,
    ):
        raw_blocks = {
            'encoder': self.model.model.encoder.times_block,
            'decoder': self.model.model.decoder.times_block,
        }
        ema_blocks = {
            'encoder': self.ema.ema_model.model.encoder.times_block,
            'decoder': self.ema.ema_model.model.decoder.times_block,
        }
        row = {
            'step': self.step,
            'variant': self.variant,
            'seed': self.training_seed,
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'learning_rate': self.opt.param_groups[0]['lr'],
            'step_seconds': (time.time() - self._telemetry_clock) / self.log_frequency,
            'peak_gpu_memory_mb': (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if torch.cuda.is_available() else 0.0
            ),
        }
        fallback_count = 0
        for role in ('encoder', 'decoder'):
            block = raw_blocks[role]
            ema_block = ema_blocks[role]
            metrics = block.last_concentration_metrics or {}
            candidate_metrics = block.last_candidate_diagnostics or {}
            fallback_count += int(candidate_metrics.get('fallback_count', 0))
            row.update({
                role + '_k_logits': float(block.k_logits.detach().item()),
                role + '_ema_k_logits': float(
                    ema_block.k_logits.detach().item()
                ),
                role + '_k_cont': block.last_k_cont,
                role + '_hard_k': block.last_k_value,
                role + '_k_logits_gradient_norm': k_gradient_norms[role],
            })
            for key, value in metrics.items():
                row[role + '_' + key] = value
            for key, value in candidate_metrics.items():
                if key not in ('fft_length', 'fallback_count'):
                    row[role + '_' + key] = value
        row['candidate_fallback_count'] = fallback_count
        with self.telemetry_path.open('a', newline='') as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.telemetry_columns,
                extrasaction='ignore',
            )
            writer.writerow(row)
        self._telemetry_clock = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'scheduler': self.sch.state_dict(),
            'config': self.complete_config,
            'source_commit': self._source_commit(),
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch_cpu': torch.get_rng_state(),
                'torch_cuda': (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
            },
        }
        if self.config['solver'].get('checkpoint_name_by_step', False):
            checkpoint_path = self.results_folder / f'checkpoint-step-{self.step}.pt'
        else:
            checkpoint_path = self.results_folder / f'checkpoint-{milestone}.pt'
        temporary_path = checkpoint_path.with_name(
            checkpoint_path.name + f'.tmp-{os.getpid()}'
        )
        torch.save(data, str(temporary_path))
        os.replace(str(temporary_path), str(checkpoint_path))
        print(f'checkpoint saved: {checkpoint_path}', flush=True)
    
    def save_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current classifer to {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        data = {
            'step': self.step_classifier,
            'classifier': self.classifier.state_dict()
        }
        torch.save(data, str(self.results_folder / f'ckpt_classfier-{milestone}.pt'))

    def load(self, milestone, verbose=False, restore_rng=True):
        if self.config['solver'].get('checkpoint_name_by_step', False):
            checkpoint_name = f'checkpoint-step-{milestone}.pt'
        else:
            checkpoint_name = f'checkpoint-{milestone}.pt'
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / checkpoint_name)))
        device = self.device
        data = torch.load(str(self.results_folder / checkpoint_name), map_location=device)
        print(str(self.results_folder / checkpoint_name))
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if 'scheduler' in data:
            self.sch.load_state_dict(data['scheduler'])
        self.ema.load_state_dict(data['ema'])
        rng_state = data.get('rng_state')
        if restore_rng and rng_state is not None:
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch_cpu'].cpu())
            if torch.cuda.is_available() and rng_state.get('torch_cuda') is not None:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in rng_state['torch_cuda']]
                )
        self.milestone = milestone

    def load_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'ckpt_classfier-{milestone}.pt'), map_location=device)
        self.classifier.load_state_dict(data['classifier'])
        self.step_classifier = data['step']
        self.milestone_classifier = milestone

    def train(self):
        device = self.device
        step = self.step
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        # train progress를 표시하려면 tqdm을 사용해서 with문을 만들어, 
        # While문 안에서 step이 train_num_steps보다 작은 동안 반복하도록 한다. 
        # tqdm의 초기값은 step으로 설정하고, 총 반복 횟수는 train_num_steps로 설정한다.
        # while문 안에서는 매 반복마다 total_loss를 계산하고, tqdm의 set_description 메서드를 사용하여 현재 loss를 표시한다.
        # pbar.update(1)을 호출하여 tqdm의 진행 상황을 업데이트한다.
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                total_diffusion_loss = 0.
                last_log_dict = None
                self.opt.zero_grad(set_to_none=True)
                for accumulation_index in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    diffusion_loss, log_dict = self.model(data, target=data)
                    loss = diffusion_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Non-finite training loss at step {step}: {loss.item()}')
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()
                    total_diffusion_loss += (
                        float(diffusion_loss.detach().item())
                        / self.gradient_accumulate_every
                    )
                    last_log_dict = log_dict

                pbar.set_description(f'loss: {total_loss:.6f}')

                k_gradient_norms = {}
                for role, block in (
                    ('encoder', self.model.model.encoder.times_block),
                    ('decoder', self.model.model.decoder.times_block),
                ):
                    gradient = block.k_logits.grad
                    k_gradient_norms[role] = (
                        0.0 if gradient is None
                        else float(gradient.detach().norm().item())
                    )

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.step += 1
                step += 1
                self.ema.update()

                if self.step % self.log_frequency == 0:
                    current_lr = self.opt.param_groups[0]['lr']
                    print(
                        f'train progress: step={self.step}/{self.train_num_steps} '
                        f'loss={total_loss:.6f} lr={current_lr:.3e}',
                        flush=True,
                    )
                    self._append_telemetry(
                        total_loss,
                        total_diffusion_loss,
                        last_log_dict,
                        k_gradient_norms,
                    )

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))
        

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]], dtype=np.float32)
        num_cycle = (num + size_every - 1) // size_every

        for cycle_idx in range(num_cycle):
            current_size = min(size_every, num - cycle_idx * size_every)
            sample = self.ema.ema_model.generate_mts(batch_size=current_size, model_kwargs=model_kwargs, cond_fn=cond_fn)

            # samples 배열에 sample을 추가한다. sample은 텐서이므로, detach()를 사용하여 그래프에서 분리하고, 
            # cpu()로 CPU로 이동한 다음, numpy()로 NumPy 배열로 변환한다. 그리고 np.row_stack을 사용하여 samples 배열에 sample을 추가한다.
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])

            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples[:num]

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize 
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            # sampling_steps가 self.model.num_timesteps와 같으면 sample_infill을 사용하고, 그렇지 않으면 fast_sample_infill을 사용한다.
            # self.ema.ema_model은 Diffusion_TS 모델의 EMA 버전입니다. sample_infill과 fast_sample_infill은 모델의 샘플링 메서드로, 조건부 샘플링을 수행합니다.
            # num_timesteps는 모델이 훈련된 시간 단계의 수를 나타냅니다. sampling_steps가 num_timesteps와 같으면, 모델은 전체 시간 단계에 대해 샘플링을 수행합니다.
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            # sampling_steps가 num_timesteps와 다르면, 모델은 빠른 샘플링을 수행합니다. fast_sample_infill은 샘플링 속도를 높이기 위해 일부 시간 단계를 건너뛰면서 샘플링을 수행합니다.
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks


    def restore_with_trace(
        self,
        raw_dataloader,
        shape=None,
        coef=1e-1,
        stepsize=1e-1,
        sampling_steps=50,
        trace_steps=None,
        max_batches=1
    ):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore with trace...')

        model_kwargs = {
            'coef': coef,
            'learning_rate': stepsize,
        }

        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        trace_out = None

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)

            if sampling_steps == self.model.num_timesteps:
                sample, trace = self.ema.ema_model.sample_infill_with_trace(
                    shape=x.shape,
                    target=x * t_m, 
                    partial_mask=t_m, 
                    model_kwargs=model_kwargs,
                    trace_steps=trace_steps   # usually None
                )
            else:
                sample, trace = self.ema.ema_model.fast_sample_infill_with_trace(
                    shape=x.shape,
                    target=x * t_m, # conditioning target
                    partial_mask=t_m, 
                    model_kwargs=model_kwargs,
                    sampling_timesteps=sampling_steps,
                    trace_steps=trace_steps   # usually None
                )

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

            if trace_out is None:
                trace_out = trace

                #  IMPORTANT FIX: store FULL ground truth, not masked target
                for tt in trace_out.keys():
                    trace_out[tt]["gt"] = x.detach().cpu()
                    trace_out[tt]["mask"] = t_m.detach().cpu()

            if idx + 1 >= max_batches:
                break

        if self.logger is not None:
            self.logger.log_info(
                'Restore-with-trace done, time: {:.2f}'.format(time.time() - tic)
            )

        return samples, reals, masks, trace_out

    def forward_sample(self, x_start):
       b, c, h = x_start.shape
       noise = torch.randn_like(x_start, device=self.device)
       t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()
       x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()
       return x_t, t

    def train_classfier(self, classifier):
        device = self.device
        step = 0
        self.milestone_classifier = 0
        self.step_classifier = 0
        dataloader = self.dataloader
        dataloader.dataset.shift_period('test')
        dataloader = cycle(dataloader)

        self.classifier = classifier
        self.opt_classifier = Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=5.0e-4)
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training classifier...'.format(self.args.name), check_primary=False)
        
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    x, y = next(dataloader)
                    x, y = x.to(device), y.to(device)
                    x_t, t = self.forward_sample(x)
                    logits = classifier(x_t, t)
                    loss = F.cross_entropy(logits, y)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                self.opt_classifier.step()
                self.opt_classifier.zero_grad()
                self.step_classifier += 1
                step += 1

                with torch.no_grad():
                    if self.step_classifier != 0 and self.step_classifier % self.save_cycle == 0:
                        self.milestone_classifier += 1
                        self.save_classifier(self.milestone_classifier)
                                            
                    if self.logger is not None and self.step_classifier % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

