import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def rand_region(size, patch_size):
    H, W = size[2], size[3]
    pH, pW = patch_size[2], patch_size[3]
    maxH = H - pH
    maxW = W - pW
    x1 = np.random.randint(0, maxH)
    y1 = np.random.randint(0, maxW)
    x2 = x1 + pH
    y2 = y1 + pW
    return x1, y1, x2, y2
