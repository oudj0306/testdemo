import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.xfeat import XFeat


def draw_matches(img1, img2, mkpts0, mkpts1, title='Matches', max_pts=500):
    """Draw matching lines between two images side by side."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1+w2] = img2 if img2.ndim == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    if len(mkpts0) > max_pts:
        idxs = np.random.choice(len(mkpts0), max_pts, replace=False)
        mkpts0 = mkpts0[idxs]
        mkpts1 = mkpts1[idxs]

    colors = plt.cm.hsv(np.linspace(0, 1, len(mkpts0)))[:, :3] * 255
    for i, (p0, p1) in enumerate(zip(mkpts0, mkpts1)):
        c = tuple(int(x) for x in colors[i])
        pt0 = (int(p0[0]), int(p0[1]))
        pt1 = (int(p1[0]) + w1, int(p1[1]))
        cv2.line(canvas, pt0, pt1, c, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0, 3, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, c, -1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = cv2.putText(canvas, f'{title} ({len(mkpts0)} matches)', (10, 25), font, 0.7, (0, 255, 0), 2)
    return canvas


def main():
    out_dir = 'data/results'
    os.makedirs(out_dir, exist_ok=True)

    ref_path = 'assets/ref.png'
    tgt_path = 'assets/tgt.png'

    img1 = cv2.imread(ref_path)
    img2 = cv2.imread(tgt_path)

    print(f'Reference: {img1.shape}, Target: {img2.shape}')

    xfeat = XFeat()

    # XFeat sparse matching
    print('Running XFeat sparse matching...')
    mkpts0, mkpts1 = xfeat.match_xfeat(img1, img2)
    print(f'  XFeat: {len(mkpts0)} matches')
    canvas_sparse = draw_matches(img1, img2, mkpts0, mkpts1, 'XFeat Sparse')
    cv2.imwrite(os.path.join(out_dir, 'xfeat_sparse_matches.png'), canvas_sparse)

    # XFeat-star semi-dense matching
    print('Running XFeat-star semi-dense matching...')
    result = xfeat.match_xfeat_star(img1, img2)
    mkpts0_sd, mkpts1_sd = result
    print(f'  XFeat-star: {len(mkpts0_sd)} matches')
    canvas_dense = draw_matches(img1, img2, mkpts0_sd, mkpts1_sd, 'XFeat-star Semi-Dense')
    cv2.imwrite(os.path.join(out_dir, 'xfeat_star_matches.png'), canvas_dense)

    # Side-by-side comparison with fewer matches visible
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, canvas, name in zip(
        axes,
        [canvas_sparse, canvas_dense],
        ['XFeat (Sparse)', 'XFeat-star (Semi-Dense)']
    ):
        ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to {out_dir}/')


if __name__ == '__main__':
    main()
