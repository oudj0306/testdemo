"""
XFeat match optimizations. Does NOT modify source code.
Tests: top_k variants, Lowe's ratio test, min_cossim filtering.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader
import tqdm, json

from modules.xfeat import XFeat
from modules.eval.megadepth1500 import (
    MegaDepth1500, tensor2bgr, compute_pose_error, error_auc
)


def match_xfeat_with_ratio_test(xfeat, img1, img2, top_k=4096, ratio_thresh=0.9):
    """XFeat matching + Lowe's ratio test on cosine similarities."""
    img1 = xfeat.parse_input(img1)
    img2 = xfeat.parse_input(img2)

    out1 = xfeat.detectAndCompute(img1, top_k=top_k)[0]
    out2 = xfeat.detectAndCompute(img2, top_k=top_k)[0]

    sim = out1['descriptors'] @ out2['descriptors'].T

    # Top-2 matches per descriptor
    vals, idxs = sim.topk(k=min(2, sim.shape[1]), dim=1)
    if vals.shape[1] < 2:
        # Only 1 match possible, accept all
        keep = torch.ones(sim.shape[0], dtype=torch.bool)
        mkpts0 = out1['keypoints'][keep].cpu().numpy()
        mkpts1 = out2['keypoints'][idxs[:, 0][keep]].cpu().numpy()
        return mkpts0, mkpts1

    best_vals, best_idxs = vals[:, 0], idxs[:, 0]
    second_vals = vals[:, 1]
    ratio = second_vals / (best_vals + 1e-8)
    keep = ratio < ratio_thresh

    mkpts0 = out1['keypoints'][keep].cpu().numpy()
    mkpts1 = out2['keypoints'][best_idxs[keep]].cpu().numpy()
    return mkpts0, mkpts1


def match_xfeat_star_with_ratio(xfeat, img1, img2, top_k=10000, ratio_thresh=0.9):
    """XFeat-star with ratio test on coarse matches."""
    img1 = xfeat.parse_input(img1)
    img2 = xfeat.parse_input(img2)

    out1 = xfeat.detectAndComputeDense(img1, top_k=top_k)
    out2 = xfeat.detectAndComputeDense(img2, top_k=top_k)

    desc1 = out1['descriptors'][0]
    desc2 = out2['descriptors'][0]
    sim = desc1 @ desc2.T

    vals, idxs = sim.topk(k=min(2, sim.shape[1]), dim=1)
    if vals.shape[1] < 2:
        keep = torch.ones(sim.shape[0], dtype=torch.bool)
    else:
        best_vals, best_idxs = vals[:, 0], idxs[:, 0]
        second_vals = vals[:, 1]
        ratio = second_vals / (best_vals + 1e-8)
        keep = ratio < ratio_thresh

    idx0 = torch.arange(len(keep), device=keep.device)[keep]
    idx1 = best_idxs[keep]
    match_list = [(idx0, idx1)]
    refined = xfeat.refine_matches(out1, out2, matches=match_list, batch_idx=0)
    return refined[:, :2].detach().cpu().numpy(), refined[:, 2:].detach().cpu().numpy()


def run_benchmark_custom(matcher_fn, dataset_dir, json_file, ransac_thr=2.5, max_pairs=None):
    dataset = MegaDepth1500(json_file=json_file, root_dir=dataset_dir + "/megadepth_test_1500")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    pairs = []
    for i, d in enumerate(tqdm.tqdm(loader, desc='Bench')):
        if max_pairs and i >= max_pairs:
            break
        src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']))
        del d['image0'], d['image1']
        src_pts = src_pts * d['scale0'].numpy()
        dst_pts = dst_pts * d['scale1'].numpy()
        d.update({"pts0": src_pts, "pts1": dst_pts, 'ransac_thr': ransac_thr})
        compute_pose_error(d)
        pairs.append(d)

    errors = [max(p['t_err'], p['R_err']) for p in pairs]
    aucs = error_auc(errors)
    errors_np = np.array(errors)
    accs = {t: float((errors_np <= t).sum() / len(errors_np) * 100) for t in [5, 10, 20]}
    return aucs, accs


if __name__ == '__main__':
    DATASET_DIR = 'data/Mega1500'
    JSON_FILE = 'assets/megadepth_1500.json'
    N_QUICK = 300  # Use 300 pairs for quick scan, set to None for full

    xfeat = XFeat()
    xfeat_star = XFeat(top_k=10000)

    configs = [
        # XFeat sparse variants
        ("XFeat baseline (4096)",       lambda i1, i2: xfeat.match_xfeat(i1, i2, top_k=4096)),
        ("XFeat top_k=8192",            lambda i1, i2: xfeat.match_xfeat(i1, i2, top_k=8192)),
        ("XFeat + ratio 0.9",           lambda i1, i2: match_xfeat_with_ratio_test(xfeat, i1, i2, top_k=4096, ratio_thresh=0.9)),
        ("XFeat + ratio 0.85",          lambda i1, i2: match_xfeat_with_ratio_test(xfeat, i1, i2, top_k=4096, ratio_thresh=0.85)),
        ("XFeat + ratio 0.8",           lambda i1, i2: match_xfeat_with_ratio_test(xfeat, i1, i2, top_k=4096, ratio_thresh=0.8)),
        ("XFeat 8k + ratio 0.9",        lambda i1, i2: match_xfeat_with_ratio_test(xfeat, i1, i2, top_k=8192, ratio_thresh=0.9)),
        # XFeat-star variants
        ("XFeat-star baseline (10k)",   lambda i1, i2: xfeat_star.match_xfeat_star(i1, i2, top_k=10000)),
        ("XFeat-star top_k=5000",       lambda i1, i2: xfeat_star.match_xfeat_star(i1, i2, top_k=5000)),
        ("XFeat-star + ratio 0.9",      lambda i1, i2: match_xfeat_star_with_ratio(xfeat_star, i1, i2, top_k=10000, ratio_thresh=0.9)),
    ]

    results = {}
    for name, fn in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        aucs, accs = run_benchmark_custom(fn, DATASET_DIR, JSON_FILE, max_pairs=N_QUICK)
        results[name] = {'auc': aucs, 'macc': accs}
        print(f"  AUC@5={aucs['auc@5']*100:.1f}  AUC@10={aucs['auc@10']*100:.1f}  AUC@20={aucs['auc@20']*100:.1f}")
        print(f"  mAcc@5={accs[5]:.1f}  mAcc@10={accs[10]:.1f}  mAcc@20={accs[20]:.1f}")

    os.makedirs('data/results', exist_ok=True)
    json.dump({k: {'auc': {kk: round(vv, 6) for kk, vv in v['auc'].items()}, 'macc': v['macc']}
               for k, v in results.items()},
              open('data/results/optimization_results.json', 'w'), indent=2)

    print(f"\n\n{'='*80}")
    print(f"Final Comparison ({N_QUICK}-pair quick scan)")
    print(f"{'='*80}")
    print(f"{'Method':<35} | {'AUC@5':>7} | {'AUC@10':>7} | {'AUC@20':>7} | {'mAcc@5':>7} | {'mAcc@10':>7} | {'mAcc@20':>7}")
    print("-" * 105)
    for name, r in results.items():
        a = r['auc']; m = r['macc']
        print(f"{name:<35} | {a['auc@5']*100:>6.1f} | {a['auc@10']*100:>6.1f} | {a['auc@20']*100:>6.1f} | {m[5]:>6.1f} | {m[10]:>6.1f} | {m[20]:>6.1f}")
