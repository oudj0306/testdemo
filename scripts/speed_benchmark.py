"""XFeat speed benchmark across resolutions (GPU)."""
import torch
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.xfeat import XFeat
import numpy as np

resolutions = [(480, 320), (640, 480), (800, 600), (1024, 768), (1280, 960)]
n_warmup, n_iter = 5, 50

xfeat = XFeat()

print(f"{'Resolution':>12} | {'Infer(ms)':>10} | {'Detect+Comp(ms)':>14} | {'FPS':>8}")
print("-" * 55)

for W, H in resolutions:
    img = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    # Warmup
    for _ in range(n_warmup):
        xfeat.detectAndCompute(img)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        xfeat.detectAndCompute(img)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_iter * 1000

    fps = 1000 / elapsed
    print(f"{W}x{H:>5}  | {elapsed:>8.1f}  | {elapsed:>12.1f}  | {fps:>6.1f}")

# Also measure match time for a typical pair
img1 = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
img2 = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

for _ in range(n_warmup):
    xfeat.match_xfeat(img1, img2)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iter):
    xfeat.match_xfeat(img1, img2)
torch.cuda.synchronize()
match_ms = (time.time() - t0) / n_iter * 1000

for _ in range(n_warmup):
    xfeat.match_xfeat_star(img1, img2)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_iter):
    xfeat.match_xfeat_star(img1, img2)
torch.cuda.synchronize()
star_ms = (time.time() - t0) / n_iter * 1000

print(f"\n{'Method':>12} | {'Total(ms)':>10} | {'FPS':>8}")
print("-" * 38)
print(f"{'XFeat match':>12} | {match_ms:>8.1f}  | {1000/match_ms:>6.1f}")
print(f"{'XFeat* match':>12} | {star_ms:>8.1f}  | {1000/star_ms:>6.1f}")

# Model info
p = sum(x.numel() for x in xfeat.net.parameters())
print(f"\nModel params: {p:,} (~{p*4/1e6:.1f} MB)")
