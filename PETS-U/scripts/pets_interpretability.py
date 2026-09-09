import os
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Utils.io_utils import load_yaml_config, merge_opts_to_config, instantiate_from_config, seed_everything
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from pets_runtime import ema_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='OUTPUT/interpretability')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--trace_steps', type=str, default='auto',
                        help='comma-separated reverse steps, or "auto"')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def load_model_and_diffusion(config_file, checkpoint, gpu, opts):
    config = load_yaml_config(config_file)
    config['model']['params']['configs'] = SimpleNamespace(**config['model']['params']['configs'])
    config = merge_opts_to_config(config, opts)

    model = instantiate_from_config(config['model']).cuda(gpu)
    ckpt = torch.load(checkpoint, map_location=f'cuda:{gpu}')

    print(f"[PETS] checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    if isinstance(ckpt, dict):
        if 'ema' in ckpt:
            print("[PETS] Using EMA weights")
        elif 'model' in ckpt:
            print("[PETS] Using model weights")
        else:
            print("[PETS] Using raw checkpoint dict directly")

    total_norm = 0.0
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            total_norm += p.data.norm().item()
            count += 1
    print(f"[PETS] mean parameter norm: {total_norm / max(count,1):.6f}")

    if not isinstance(ckpt, dict):
        raise RuntimeError("expected a Diffusion-PETS checkpoint dictionary")
    incompatible = model.load_state_dict(ema_state(ckpt), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("checkpoint/model state mismatch")

    model.eval()
    return model, config


def compute_mean_spectrum(x_btc):
    """
    x_btc: torch.Tensor or np.ndarray [B, T, C]
    returns: np.ndarray [F]
    """
    if torch.is_tensor(x_btc):
        x_btc = x_btc.detach().cpu().numpy()
    spec = np.abs(np.fft.rfft(x_btc, axis=1))   # [B, F, C]
    spec = spec.mean(axis=(0, 2))               # [F]
    return spec


def plot_period_panel(ax, trace, key, title):
    xs, ys = [], []
    for t in sorted(trace.keys(), reverse=True):
        periods = trace[t].get(key, None)
        print(f"t={t}, key={key}, periods={periods}")
        if periods is None:
            continue
        for p in periods:
            xs.append(t)
            ys.append(p)
    if len(xs) > 0:
        ax.scatter(xs, ys, s=14, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('reverse step t')
    ax.set_ylabel('selected period')
    ax.invert_xaxis()

def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        x = int(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def plot_period_panel(ax, trace, key, title):
    xs, ys = [], []

    def unique_preserve_order(seq):
        seen = set()
        out = []
        for x in seq:
            x = int(x)
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    for t in sorted(trace.keys(), reverse=True):
        periods = trace[t].get(key, None)
        if periods is None:
            continue

        periods = unique_preserve_order(periods)
        print(f"t={t}, key={key},{title},unique periods={periods}")
        for p in periods:
            xs.append(t)
            ys.append(p)
        

    if len(xs) > 0:
        ax.scatter(xs, ys, s=14, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel('reverse step t')
    ax.set_ylabel('selected period')
    ax.invert_xaxis()

def plot_snapshots(fig, trace, row_offset, title_prefix, ncols=5,   dataset=None):
    selected_steps = sorted(trace.keys(), reverse=True)
    for i, t in enumerate(selected_steps):
        ax = fig.add_subplot(3, ncols, row_offset + i + 1)
        print(f"x_start shape: {trace[t]['x_start'].shape}")

        sample_raw = trace[t]["x_start"][0].numpy()  # [T, C]
        raw_sig = sample_raw[:, 0]

        print(f"[PETS][t={t}] RAW x_start stats: "
            f"min={raw_sig.min():.6f}, max={raw_sig.max():.6f}, "
            f"mean={raw_sig.mean():.6f}, std={raw_sig.std():.6f}")

        sample_vis = unnormalize_to_zero_to_one(sample_raw)
        sample_vis = dataset.scaler.inverse_transform(
            sample_vis.reshape(-1, sample_vis.shape[-1])
        ).reshape(sample_vis.shape)

        vis_sig = sample_vis[:, 0]
        print(f"[PETS][t={t}] INV x_start stats: "
            f"min={vis_sig.min():.6f}, max={vis_sig.max():.6f}, "
            f"mean={vis_sig.mean():.6f}, std={vis_sig.std():.6f}")
            
        ax.plot(vis_sig)
        # ax.set_ylim(-20, 40) # etth dataset에 맞춰 y축 범위를 조정합니다. 필요에 따라 다른 범위를 설정할 수 있습니다.
        #ax.set_ylim(-10, 100) # air quality dataset에 맞춰 y축 범위를 조정합니다. 필요에 따라 다른 범위를 설정할 수 있습니다.
        #ax.set_ylim(-20, 20) # traffic dataset에 맞춰 y축 범위를 조정합니다. 필요에 따라 다른 범위를 설정할 수 있습니다.
        ax.set_title(f'{title_prefix} t={t}')
        #ax.set_xticks([])
        ax.grid(alpha=0.25)


def plot_spectra(fig, trace, row_offset, title_prefix, ncols=5, dataset=None):
    selected_steps = sorted(trace.keys(), reverse=True)
    for i, t in enumerate(selected_steps):
        ax = fig.add_subplot(3, ncols, row_offset + i + 1)

        x = trace[t]["x_start"]  # [B, T, C]
        x = x.detach().cpu().numpy()

        # 🔥 IMPORTANT: same inverse transform as snapshots
        x = unnormalize_to_zero_to_one(x)
        x = dataset.scaler.inverse_transform(
            x.reshape(-1, x.shape[-1])
        ).reshape(x.shape)

        spec = compute_mean_spectrum(x)

        ax.plot(spec[1:])  # remove DC
        ax.set_title(f'{title_prefix} t={t}')
        ax.grid(alpha=0.25)


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    os.makedirs(args.save_dir, exist_ok=True)

    model, config = load_model_and_diffusion(args.config_file, args.checkpoint, args.gpu, args.opts)

    seq_len = config['model']['params']['seq_length']
    n_feat = config['model']['params']['feature_size']

    # interpretability 분석을 위해 dataloader와 dataset을 로드합니다. 
    # dataset은 정규화를 시켜놓습니다.
    dataloader_info = build_dataloader(config, args)
    dataset = dataloader_info['dataset']

    real_sample = dataset[0]
    if isinstance(real_sample, (list, tuple)):
        real_sample = real_sample[0]
    real_sample = real_sample.numpy() if torch.is_tensor(real_sample) else np.asarray(real_sample)

    if real_sample.ndim == 2:
        real_vis = unnormalize_to_zero_to_one(real_sample)
        real_vis = dataset.scaler.inverse_transform(
            real_vis.reshape(-1, real_vis.shape[-1])
        ).reshape(real_vis.shape)

        real_sig = real_vis[:, 0]
        print(f"[REAL] first window stats: "
            f"min={real_sig.min():.4f}, max={real_sig.max():.4f}, "
            f"mean={real_sig.mean():.4f}, std={real_sig.std():.4f}")



    diffusion = model  # your instantiated object appears to already be the diffusion wrapper
    Tdiff = diffusion.num_timesteps

    if args.trace_steps == 'auto':
        trace_steps = sorted(set([Tdiff - 1, int(0.75 * Tdiff), int(0.5 * Tdiff), int(0.25 * Tdiff), 0]))
    else:
        trace_steps = sorted(set([int(x) for x in args.trace_steps.split(',') if x.strip() != '']), reverse=True)

    
    print(f"seq_len={seq_len}, n_feat={n_feat}")
    print(f"trace_steps={trace_steps}")
    print(f"dataset scaler type={type(dataset.scaler)}")

    # 샘플링을 수행하면서 중간 단계의 x_start를 포함한 trace를 수집합니다. 
    # trace는 dict 형태로, key는 reverse step t, value는 해당 단계의 정보(예: x_start, selected periods 등)를 담고 있습니다.
    final_sample, trace, s_ts, n_ts, f_ts = diffusion.sample_with_trace(
        shape=(args.batch_size, seq_len, n_feat),
        trace_steps=trace_steps
    )
    # normalized-space comparison
    real = dataset[0]
    gen  = final_sample[0]
    print("REAL std:", real[:,0].std())
    print("PETS std:", gen[:,0].std())

    print(f"sampling timesteps: {s_ts}, num_timesteps: {n_ts}, fast_sample: {f_ts}")
    # save raw trace
    torch.save(trace, os.path.join(args.save_dir, 'interpretability_trace.pt'))

    for t in sorted(trace.keys(), reverse=True):
        if "sample" in trace[t]:
            s_raw = trace[t]["sample"][0].numpy()[:, 0]
            x0_raw = trace[t]["x_start"][0].numpy()[:, 0]
            print(f"[PETS][t={t}] sample std={s_raw.std():.4f}, x_start std={x0_raw.std():.4f}")

    # figure 1: selected periods
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    plot_period_panel(ax1, trace, 'periods_enc', 'Encoder selected periods')
    ax2 = fig.add_subplot(2, 2, 2)
    plot_period_panel(ax2, trace, 'periods_dec', 'Decoder selected periods')

    # figure 2: snapshots
    fig2 = plt.figure(figsize=(16, 8))
    selected_steps = sorted(trace.keys(), reverse=True)
    ncols = len(selected_steps)
    plot_snapshots(fig2, trace, 0, 'x_start', ncols=ncols, dataset=dataset) # added dataset argument
    plot_spectra(fig2, trace, ncols, 'spectrum', ncols=ncols, dataset=dataset) # added dataset argument

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_dir, 'period_trajectories.png'), dpi=200, bbox_inches='tight')
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.save_dir, 'snapshots_and_spectra.png'), dpi=200, bbox_inches='tight')

    print(f'Saved results to: {args.save_dir}')


if __name__ == '__main__':
    main()
