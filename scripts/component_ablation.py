"""
Component ablation study for XFeat.
Ablates: fine matcher (MLP), matching direction (MNN vs NN), descriptor dimensionality.
Does NOT modify any source code.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader
import tqdm, json, time

from modules.xfeat import XFeat
from modules.eval.megadepth1500 import (
    MegaDepth1500, tensor2bgr, compute_pose_error, error_auc
)

DATASET_DIR = 'data/Mega1500'
JSON_FILE = 'assets/megadepth_1500.json'
N_QUICK = 300


# ============================================================
# Ablation 1: Fine Matcher (MLP refinement) contribution
# ============================================================
def match_xfeat_star_no_refine(xfeat, img1, img2, top_k=10000):
    """XFeat-star coarse matching WITHOUT MLP refinement."""
    img1 = xfeat.parse_input(img1)
    img2 = xfeat.parse_input(img2)
    out1 = xfeat.detectAndComputeDense(img1, top_k=top_k)
    out2 = xfeat.detectAndComputeDense(img2, top_k=top_k)

    desc1 = out1['descriptors'][0]  # (N, 64) from batched (B, N, 64)
    desc2 = out2['descriptors'][0]
    idx0, idx1 = xfeat.match(desc1, desc2, min_cossim=-1)

    # keypoints are batched: (B, N, 2), so index with [0, idx0]
    mkpts0 = out1['keypoints'][0, idx0].cpu().numpy()
    mkpts1 = out2['keypoints'][0, idx1].cpu().numpy()
    return mkpts0, mkpts1


# ============================================================
# Ablation 2: MNN vs single-direction Nearest Neighbor
# ============================================================
def match_xfeat_single_nn(xfeat, img1, img2, top_k=4096):
    """XFeat with single-direction NN (no mutual check)."""
    img1 = xfeat.parse_input(img1)
    img2 = xfeat.parse_input(img2)
    out1 = xfeat.detectAndCompute(img1, top_k=top_k)[0]
    out2 = xfeat.detectAndCompute(img2, top_k=top_k)[0]

    sim = out1['descriptors'] @ out2['descriptors'].T
    idx1 = sim.argmax(dim=1)  # single-direction: query -> target best match
    idx0 = torch.arange(len(idx1), device=idx1.device)

    mkpts0 = out1['keypoints'][idx0].cpu().numpy()
    mkpts1 = out2['keypoints'][idx1].cpu().numpy()
    return mkpts0, mkpts1


# ============================================================
# Ablation 3: Descriptor dimensionality
# ============================================================
def match_xfeat_reduced_dim(xfeat, img1, img2, top_k=4096, dim=32):
    """XFeat with reduced descriptor dimension (truncates 64D -> dim)."""
    img1 = xfeat.parse_input(img1)
    img2 = xfeat.parse_input(img2)
    out1 = xfeat.detectAndCompute(img1, top_k=top_k)[0]
    out2 = xfeat.detectAndCompute(img2, top_k=top_k)[0]

    # Truncate descriptors to first `dim` channels
    desc1 = out1['descriptors'][:, :dim]
    desc2 = out2['descriptors'][:, :dim]

    # Re-normalize truncated descriptors for cosine similarity
    desc1 = desc1 / (desc1.norm(dim=1, keepdim=True) + 1e-8)
    desc2 = desc2 / (desc2.norm(dim=1, keepdim=True) + 1e-8)

    idx0, idx1 = xfeat.match(desc1, desc2, min_cossim=-1)

    mkpts0 = out1['keypoints'][idx0].cpu().numpy()
    mkpts1 = out2['keypoints'][idx1].cpu().numpy()
    return mkpts0, mkpts1


# ============================================================
# Benchmark runner
# ============================================================
def run_ablation(matcher_fn, name, ransac_thr=2.5):
    dataset = MegaDepth1500(json_file=JSON_FILE, root_dir=DATASET_DIR + "/megadepth_test_1500")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    pairs = []
    t0 = time.time()
    for i, d in enumerate(tqdm.tqdm(loader, desc=name)):
        if i >= N_QUICK:
            break
        src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']))
        del d['image0'], d['image1']
        src_pts = src_pts * d['scale0'].numpy()
        dst_pts = dst_pts * d['scale1'].numpy()
        d.update({"pts0": src_pts, "pts1": dst_pts, 'ransac_thr': ransac_thr})
        compute_pose_error(d)
        pairs.append(d)
    elapsed = time.time() - t0

    errors = [max(p['t_err'], p['R_err']) for p in pairs]
    avg_matches = np.mean([len(p['pts0']) for p in pairs])
    aucs = error_auc(errors)
    errors_np = np.array(errors)
    accs = {t: float((errors_np <= t).sum() / len(errors_np) * 100) for t in [5, 10, 20]}

    return aucs, accs, avg_matches, elapsed


if __name__ == '__main__':
    xfeat = XFeat()
    xfeat_star = XFeat(top_k=10000)

    ablations = [
        # --- Fine Matcher Ablation ---
        ("XFeat-star w/ refine (baseline)", lambda i1, i2: xfeat_star.match_xfeat_star(i1, i2, top_k=10000)),
        ("XFeat-star w/o refine (coarse only)", lambda i1, i2: match_xfeat_star_no_refine(xfeat_star, i1, i2)),

        # --- Matching Direction Ablation ---
        ("XFeat MNN (baseline)", lambda i1, i2: xfeat.match_xfeat(i1, i2, top_k=4096)),
        ("XFeat single-NN (no mutual)", lambda i1, i2: match_xfeat_single_nn(xfeat, i1, i2, top_k=4096)),

        # --- Descriptor Dimensionality Ablation ---
        ("XFeat 64D (baseline)", lambda i1, i2: xfeat.match_xfeat(i1, i2, top_k=4096)),
        ("XFeat 48D", lambda i1, i2: match_xfeat_reduced_dim(xfeat, i1, i2, top_k=4096, dim=48)),
        ("XFeat 32D", lambda i1, i2: match_xfeat_reduced_dim(xfeat, i1, i2, top_k=4096, dim=32)),
        ("XFeat 16D", lambda i1, i2: match_xfeat_reduced_dim(xfeat, i1, i2, top_k=4096, dim=16)),
    ]

    results = {}
    print(f"{'='*90}")
    print(f"XFeat Component Ablation Study ({N_QUICK}-pair quick scan, RANSAC thr=2.5)")
    print(f"{'='*90}")

    for name, fn in ablations:
        print(f"\n{'-'*70}")
        print(f">> {name}")
        print(f"{'-'*70}")
        aucs, accs, avg_m, elapsed = run_ablation(fn, name)
        results[name] = {'auc': aucs, 'macc': accs, 'avg_matches': avg_m, 'time_s': elapsed}
        print(f"  AUC@5={aucs['auc@5']*100:.1f}  AUC@10={aucs['auc@10']*100:.1f}  AUC@20={aucs['auc@20']*100:.1f}")
        print(f"  mAcc@5={accs[5]:.1f}  mAcc@10={accs[10]:.1f}  mAcc@20={accs[20]:.1f}")
        print(f"  Avg matches={avg_m:.0f}  Time={elapsed:.1f}s")

    # ---- Summary Tables ----
    print(f"\n\n{'='*90}")
    print(f"SUMMARY: Fine Matcher Ablation (XFeat-star)")
    print(f"{'='*90}")
    print(f"{'Config':<40} | {'AUC@5':>7} | {'AUC@10':>7} | {'AUC@20':>7} | {'#Matches':>8}")
    print("-" * 80)
    for name in ["XFeat-star w/ refine (baseline)", "XFeat-star w/o refine (coarse only)"]:
        r = results[name]
        print(f"{name:<40} | {r['auc']['auc@5']*100:>6.1f} | {r['auc']['auc@10']*100:>6.1f} | {r['auc']['auc@20']*100:>6.1f} | {r['avg_matches']:>7.0f}")

    print(f"\n{'='*90}")
    print(f"SUMMARY: Matching Direction Ablation (XFeat)")
    print(f"{'='*90}")
    print(f"{'Config':<40} | {'AUC@5':>7} | {'AUC@10':>7} | {'AUC@20':>7} | {'#Matches':>8}")
    print("-" * 80)
    for name in ["XFeat MNN (baseline)", "XFeat single-NN (no mutual)"]:
        r = results[name]
        print(f"{name:<40} | {r['auc']['auc@5']*100:>6.1f} | {r['auc']['auc@10']*100:>6.1f} | {r['auc']['auc@20']*100:>6.1f} | {r['avg_matches']:>7.0f}")

    print(f"\n{'='*90}")
    print(f"SUMMARY: Descriptor Dimensionality Ablation (XFeat)")
    print(f"{'='*90}")
    print(f"{'Config':<40} | {'AUC@5':>7} | {'AUC@10':>7} | {'AUC@20':>7} | {'#Matches':>8}")
    print("-" * 80)
    for name in ["XFeat 64D (baseline)", "XFeat 48D", "XFeat 32D", "XFeat 16D"]:
        r = results[name]
        print(f"{name:<40} | {r['auc']['auc@5']*100:>6.1f} | {r['auc']['auc@10']*100:>6.1f} | {r['auc']['auc@20']*100:>6.1f} | {r['avg_matches']:>7.0f}")

    os.makedirs('data/results', exist_ok=True)
    json.dump({k: {'auc': {kk: round(vv, 6) for kk, vv in v['auc'].items()},
                    'macc': v['macc'], 'avg_matches': v['avg_matches'], 'time_s': v['time_s']}
               for k, v in results.items()},
              open('data/results/ablation_results.json', 'w'), indent=2)
    print(f"\nResults saved to data/results/ablation_results.json")
