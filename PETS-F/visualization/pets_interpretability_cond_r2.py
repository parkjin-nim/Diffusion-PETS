import os
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Utils.io_utils import load_yaml_config, merge_opts_to_config, seed_everything
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Data.build_dataloader import build_dataloader_cond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--pets_trace', type=str, required=True)
    parser.add_argument('--ts_trace', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='OUTPUT/conditional_interpretability')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--trace_steps', type=str, default='auto',
                        help='comma-separated reverse steps, or "auto"')
    parser.add_argument('--channel_idx', type=int, default=0)
    parser.add_argument('--show_period_panel', type=int, default=1, choices=[0, 1])
    parser.add_argument('--max_periods_per_step', type=int, default=4)

    parser.add_argument('--mode', type=str, default='predict',
                        help='Infilling or Forecasting.')
    parser.add_argument('--pred_len', type=int, default=64, help='Length of Predictions.')

    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def load_config_and_dataset(config_file, args):
    config = load_yaml_config(config_file)
    config = merge_opts_to_config(config, args.opts)

    dataloader_info = build_dataloader_cond(config, args)
    dataset = dataloader_info['dataset']
    return config, dataset


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def inverse_transform_batch(x_btc, dataset):
    """
    x_btc: [B, T, C] in normalized model space
    """
    x_btc = to_numpy(x_btc).copy()
    x_btc = unnormalize_to_zero_to_one(x_btc)
    x_btc = dataset.scaler.inverse_transform(
        x_btc.reshape(-1, x_btc.shape[-1])
    ).reshape(x_btc.shape)
    return x_btc


def inverse_transform_single(x_tc, dataset):
    """
    x_tc: [T, C]
    """
    x_tc = to_numpy(x_tc).copy()
    x_tc = unnormalize_to_zero_to_one(x_tc)
    x_tc = dataset.scaler.inverse_transform(
        x_tc.reshape(-1, x_tc.shape[-1])
    ).reshape(x_tc.shape)
    return x_tc


def compute_mean_spectrum(x_btc, remove_dc=True, normalize=True, mode="max"):
    """
    x_btc: [B, T, C]
    """
    x_btc = to_numpy(x_btc)
    spec = np.abs(np.fft.rfft(x_btc, axis=1))   # [B, F, C]
    spec = spec.mean(axis=(0, 2))               # [F]

    if remove_dc and len(spec) > 0:
        spec = spec.copy()
        spec[0] = 0.0

    if normalize:
        if mode == "max":
            spec = spec / (spec.max() + 1e-8)
        elif mode == "sum":
            spec = spec / (spec.sum() + 1e-8)
        elif mode == "log":
            spec = np.log1p(spec)

    return spec


def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        x = int(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _to_1d_weights(weights):
    if weights is None:
        return None
    w = to_numpy(weights)
    w = np.asarray(w, dtype=np.float32)

    if w.ndim == 0:
        return np.array([float(w)], dtype=np.float32)
    if w.ndim == 1:
        return w
    while w.ndim > 1:
        w = w.mean(axis=0)
    return w.astype(np.float32)


def merge_duplicate_periods(periods, weights=None, reduce="sum"):
    periods = to_numpy(periods)
    if periods is None:
        return [], []

    periods = np.asarray(periods).astype(np.int64).tolist()
    weights = _to_1d_weights(weights)

    if weights is None:
        weights = np.ones(len(periods), dtype=np.float32)
    else:
        n = min(len(periods), len(weights))
        periods = periods[:n]
        weights = weights[:n]

    order = []
    acc = {}
    cnt = {}

    for p, w in zip(periods, weights):
        p = int(p)
        if p not in acc:
            acc[p] = 0.0
            cnt[p] = 0
            order.append(p)
        acc[p] += float(w)
        cnt[p] += 1

    if reduce == "mean":
        merged = [acc[p] / max(cnt[p], 1) for p in order]
    else:
        merged = [acc[p] for p in order]

    return order, merged


def plot_period_panel_weighted(ax, trace, steps, period_key="periods_dec", weight_key="weights_dec",
                               title="Decoder selected periods", max_periods_per_step=4, seq_len=128):
    xs, ys, cs, ss = [], [], [], []

    for t in steps:
        item = trace[t]
        periods = item.get(period_key, None)
        weights = item.get(weight_key, None)

        periods_u, weights_u = merge_duplicate_periods(periods, weights, reduce="sum")

        # remove full-window period for visualization only
        filtered = [(p, w) for p, w in zip(periods_u, weights_u) if p <seq_len]
        if len(filtered) > 0:
            periods_u, weights_u = zip(*filtered)
            periods_u, weights_u = list(periods_u), list(weights_u)
        else:
            periods_u, weights_u = [], []

        # limit number of periods per step for clearer visualization
        periods_u = periods_u[:max_periods_per_step]
        weights_u = weights_u[:max_periods_per_step]

        for p, w in zip(periods_u, weights_u):
            xs.append(t)
            ys.append(p)
            cs.append(w)
            ss.append(w)

    if len(xs) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No period trace", ha="center", va="center")
        return

    cs = np.asarray(cs, dtype=np.float32)
    ss = np.asarray(ss, dtype=np.float32)

    if np.allclose(ss.max(), ss.min()):
        sizes = np.full_like(ss, 40.0)
    else:
        sizes = 20.0 + 90.0 * (ss - ss.min()) / (ss.max() - ss.min() + 1e-8)

    sc = ax.scatter(xs, ys, c=cs, s=sizes, cmap="viridis", alpha=0.9, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("merged weight")

    ax.set_title(title)
    ax.set_xlabel("reverse step $t$")
    ax.set_ylabel("period")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)


def split_context_and_pred(full_seq, full_gt, full_mask):
    """
    full_seq: [T, C] current sample
    full_gt:  [T, C] ground truth
    full_mask:[T, C] boolean/0-1 observed mask

    Assumption for forecasting:
    observed region is the prefix, target region is the suffix.
    """
    mask_1d = full_mask[:, 0].astype(bool)
    pred_start = np.where(~mask_1d)[0][0]

    ctx = full_seq[:pred_start]
    pred = full_seq[pred_start:]
    gt_ctx = full_gt[:pred_start]
    gt_pred = full_gt[pred_start:]

    return pred_start, ctx, pred, gt_ctx, gt_pred


def compute_global_ylim(trace_pets, trace_ts, dataset, steps, channel_idx=0):
    vals = []

    for trace in [trace_pets, trace_ts]:
        for t in steps:
            sample = inverse_transform_single(trace[t]["sample"][0], dataset)
            gt = inverse_transform_single(trace[t]["gt"][0], dataset)
            mask = to_numpy(trace[t]["mask"][0]).astype(bool)

            _, ctx, pred, _, _ = split_context_and_pred(sample, gt, mask)
            full = np.concatenate([ctx[:, channel_idx], pred[:, channel_idx]], axis=0)
            vals.append(full)

    vals = np.concatenate(vals, axis=0)
    vmin, vmax = vals.min(), vals.max()
    pad = 0.05 * (vmax - vmin + 1e-8)
    return vmin - pad, vmax + pad


def plot_forecasting_row(fig, gs, row_idx, trace, dataset, steps, title_prefix,
                         channel_idx=0, y_lim=None, pred_len=64):
    for i, t in enumerate(steps):
        ax = fig.add_subplot(gs[row_idx, i])

        sample = inverse_transform_single(trace[t]["sample"][0], dataset)
        gt = inverse_transform_single(trace[t]["gt"][0], dataset)
        mask = to_numpy(trace[t]["mask"][0]).astype(bool)

        _, ctx, pred, _, gt_pred = split_context_and_pred(sample, gt, mask)

        x_pred = np.arange(pred_len)  # Assuming pred_len is the length of the prediction region

        # plot prediction only
        ax.plot(
            x_pred,
            pred[:, channel_idx],
            color='tab:blue',
            linewidth=1.2,
            label='generated'
        )

        # ground truth (only last step)
        if i == len(steps) - 1:
            ax.plot(
                x_pred,
                gt_pred[:, channel_idx],
                color='gray',
                linestyle='--',
                linewidth=1.0,
                label='ground truth'
            )
        ax.axvline(0, color='black', linestyle=':', linewidth=1)
        ax.text(0, ax.get_ylim()[1], "prediction start", fontsize=8)

        # show ground truth only at the final reverse step (rightmost panel)
        # if i == len(steps) - 1:
        #     ax.plot(
        #         x_pred, gt_pred[:, channel_idx],
        #         color='gray', 
        #         linestyle='--', 
        #         linewidth=1.0, 
        #         label='ground truth'
        #  )


        ax.set_title(f'{title_prefix} t={t}')
        ax.grid(alpha=0.25)

        if y_lim is not None:
            ax.set_ylim(y_lim)

        if i == 0:
            ax.set_ylabel('signal')

        if i == len(steps) - 1:
            ax.legend(frameon=False, fontsize=7)


def plot_forecasting_spectra_with_gt(fig, gs, row_idx, trace_pets, trace_ts, dataset, steps):
    for i, t in enumerate(steps):
        ax = fig.add_subplot(gs[row_idx, i])

        pets_sample = inverse_transform_batch(trace_pets[t]["sample"], dataset)
        pets_gt = inverse_transform_batch(trace_pets[t]["gt"], dataset)
        pets_mask = to_numpy(trace_pets[t]["mask"]).astype(bool)

        ts_sample = inverse_transform_batch(trace_ts[t]["sample"], dataset)
        ts_gt = inverse_transform_batch(trace_ts[t]["gt"], dataset)
        ts_mask = to_numpy(trace_ts[t]["mask"]).astype(bool)

        def get_pred_region(batch_sample, batch_gt, batch_mask):
            pred_region = []
            gt_region = []
            B = batch_sample.shape[0]
            for b in range(B):
                _, _, pred, _, gt_pred = split_context_and_pred(
                    batch_sample[b], batch_gt[b], batch_mask[b]
                )
                pred_region.append(pred)
                gt_region.append(gt_pred)
            return np.stack(pred_region, axis=0), np.stack(gt_region, axis=0)

        pets_pred, gt_pred = get_pred_region(pets_sample, pets_gt, pets_mask)
        ts_pred, _ = get_pred_region(ts_sample, ts_gt, ts_mask)

        spec_pets = compute_mean_spectrum(pets_pred, remove_dc=True, normalize=True, mode="max")
        spec_ts = compute_mean_spectrum(ts_pred, remove_dc=True, normalize=True, mode="max")
        spec_gt = compute_mean_spectrum(gt_pred, remove_dc=True, normalize=True, mode="max")

        # DC component is removed, so x-axis starts from the first non-DC frequency bin
        ax.plot(spec_pets[1:], linewidth=1.4, label='PETS')
        ax.plot(spec_ts[1:], linewidth=1.2, linestyle='--', label='Diffusion-TS')
        ax.plot(spec_gt[1:], linewidth=1.0, linestyle=':', color='black', label='GT')

        ax.set_title(f'spectrum t={t}')
        ax.grid(alpha=0.25)

        if i == 0:
            ax.set_ylabel('norm. mag.')

        if i == len(steps) - 1:
            ax.legend(frameon=False, fontsize=7)


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    os.makedirs(args.save_dir, exist_ok=True)

    config, dataset = load_config_and_dataset(args.config_file, args)

    trace_pets = torch.load(args.pets_trace, map_location='cpu')
    trace_ts = torch.load(args.ts_trace, map_location='cpu')

    if args.trace_steps == 'auto':
        trace_steps = sorted(set(trace_pets.keys()).intersection(set(trace_ts.keys())), reverse=True)
    else:
        trace_steps = sorted(set([int(x) for x in args.trace_steps.split(',') if x.strip() != '']), reverse=True)

    trace_steps = [t for t in trace_steps if t in trace_pets and t in trace_ts]

    print(f"trace_steps={trace_steps}")
    print(f"dataset scaler type={type(dataset.scaler)}")

    torch.save(trace_pets, os.path.join(args.save_dir, 'pets_conditional_trace_copy.pt'))
    torch.save(trace_ts, os.path.join(args.save_dir, 'ts_conditional_trace_copy.pt'))

    y_lim = compute_global_ylim(trace_pets, trace_ts, dataset, trace_steps, channel_idx=args.channel_idx)

    # separate period figure if requested
    if args.show_period_panel == 1:
        fig0 = plt.figure(figsize=(10, 4))
        gs0 = fig0.add_gridspec(1, 2)

        ax1 = fig0.add_subplot(gs0[0, 0])
        plot_period_panel_weighted(
            ax1,
            {t: trace_pets[t] for t in trace_steps},
            trace_steps,
            period_key='periods_enc',
            weight_key='weights_enc',
            title='Encoder selected periods',
            max_periods_per_step=args.max_periods_per_step,
            seq_len=config['model']['params']['seq_length']
        )

        ax2 = fig0.add_subplot(gs0[0, 1])
        plot_period_panel_weighted(
            ax2,
            {t: trace_pets[t] for t in trace_steps},
            trace_steps,
            period_key='periods_dec',
            weight_key='weights_dec',
            title='Decoder selected periods',
            max_periods_per_step=args.max_periods_per_step
        )

        fig0.tight_layout()
        fig0.savefig(
            os.path.join(args.save_dir, 'conditional_period_trajectories.png'),
            dpi=300, bbox_inches='tight'
        )

    # final composite figure
    ncols = len(trace_steps)
    if args.show_period_panel == 1:
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = fig.add_gridspec(4, ncols, height_ratios=[0.9, 1.0, 1.0, 0.95])

        ax_period = fig.add_subplot(gs[0, :])
        plot_period_panel_weighted(
            ax_period,
            {t: trace_pets[t] for t in trace_steps},
            trace_steps,
            period_key='periods_dec',
            weight_key='weights_dec',
            title='PETS decoder selected periods',
            max_periods_per_step=args.max_periods_per_step
        )

        row_pets = 1
        row_ts = 2
        row_spec = 3
    else:
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs = fig.add_gridspec(3, ncols, height_ratios=[1.0, 1.0, 0.95])

        row_pets = 0
        row_ts = 1
        row_spec = 2

    plot_forecasting_row(
        fig, gs, row_pets,
        {t: trace_pets[t] for t in trace_steps},
        dataset,
        trace_steps,
        title_prefix='Diffusion-PETS',
        channel_idx=args.channel_idx,
        y_lim=y_lim,
        pred_len=args.pred_len
    )

    plot_forecasting_row(
        fig, gs, row_ts,
        {t: trace_ts[t] for t in trace_steps},
        dataset,
        trace_steps,
        title_prefix='Diffusion-TS',
        channel_idx=args.channel_idx,
        y_lim=y_lim,
        pred_len=args.pred_len
    )

    plot_forecasting_spectra_with_gt(
        fig, gs, row_spec,
        {t: trace_pets[t] for t in trace_steps},
        {t: trace_ts[t] for t in trace_steps},
        dataset,
        trace_steps
    )

    fig.savefig(
        os.path.join(args.save_dir, 'conditional_snapshots_and_spectra.png'),
        dpi=300, bbox_inches='tight'
    )

    print(f'Saved upgraded conditional interpretability results to: {args.save_dir}')


if __name__ == '__main__':
    main()
