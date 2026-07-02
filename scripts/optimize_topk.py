"""
top_k ablation study for XFeat on MegaDepth-1500.
Does NOT modify any source code — wraps matcher functions with custom top_k.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np

from modules.xfeat import XFeat
from modules.eval.megadepth1500 import MegaDepth1500, run_pose_benchmark, compute_pose_error, compute_maa
from modules.eval.megadepth1500 import tensor2bgr, estimate_pose_poselib


def make_matcher(method, top_k_val):
    """Create a matcher function with a specific top_k value."""
    xfeat = XFeat(top_k=top_k_val)

    if method == 'xfeat':
        def matcher(img1, img2):
            return xfeat.match_xfeat(img1, img2, top_k=top_k_val)
    else:
        def matcher(img1, img2):
            return xfeat.match_xfeat_star(img1, img2, top_k=top_k_val)

    return matcher


def run_topk_ablation(method, dataset_dir, json_file, topk_values, ransac_thr=2.5):
    """Run ablation over multiple top_k values."""
    dataset = MegaDepth1500(json_file=json_file, root_dir=dataset_dir + "/megadepth_test_1500")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = {}
    for topk in topk_values:
        print(f"\n{'='*60}")
        print(f"Testing {method} with top_k = {topk}")
        print(f"{'='*60}")

        matcher_fn = make_matcher(method, topk)

        pairs = []
        for d in tqdm.tqdm(loader, desc=f'top_k={topk}'):
            src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']))
            del d['image0'], d['image1']
            src_pts = src_pts * d['scale0'].numpy()
            dst_pts = dst_pts * d['scale1'].numpy()
            d.update({"pts0": src_pts, "pts1": dst_pts, 'ransac_thr': ransac_thr})
            compute_pose_error(d)
            pairs.append(d)

        errors = []
        for p in pairs:
            errors.append(max(p['t_err'], p['R_err']))

        from modules.eval.megadepth1500 import error_auc
        aucs = error_auc(errors)
        errors_np = np.array(errors)
        accs = {t: (errors_np <= t).sum() / len(errors_np) * 100 for t in [5, 10, 20]}

        results[topk] = {'auc': aucs, 'macc': accs, 'total_matches': sum(len(p['pts0']) for p in pairs)}

        print(f"  AUC@5={aucs['auc@5']*100:.1f} AUC@10={aucs['auc@10']*100:.1f} AUC@20={aucs['auc@20']*100:.1f}")
        print(f"  mAcc@5={accs[5]:.1f} mAcc@10={accs[10]:.1f} mAcc@20={accs[20]:.1f}")

    return results


if __name__ == '__main__':
    DATASET_DIR = 'data/Mega1500'
    JSON_FILE = 'assets/megadepth_1500.json'
    TOPK_VALUES = [512, 1024, 2048, 4096, 8192]

    print("=" * 60)
    print("XFeat top_k Ablation Study")
    print("=" * 60)

    for method in ['xfeat', 'xfeat_star']:
        print(f"\n### {method} ###")
        results = run_topk_ablation(method, DATASET_DIR, JSON_FILE, TOPK_VALUES)

        print(f"\n{'top_k':>8} | {'AUC@5':>8} | {'AUC@10':>8} | {'AUC@20':>8} | {'mAcc@5':>8} | {'mAcc@10':>8} | {'mAcc@20':>8}")
        print("-" * 80)
        for topk in TOPK_VALUES:
            r = results[topk]
            print(f"{topk:>8} | {r['auc']['auc@5']*100:>7.1f} | {r['auc']['auc@10']*100:>7.1f} | {r['auc']['auc@20']*100:>7.1f} | {r['macc'][5]:>7.1f} | {r['macc'][10]:>7.1f} | {r['macc'][20]:>7.1f}")
