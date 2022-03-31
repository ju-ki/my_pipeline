from torchvision.transforms import Resize
import torch.nn.functional as F
from .aug_util import rand_bbox, rand_region
import torch
import numpy as np
from .fmix import sample_mask
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0
    assert x.size(0) > 1

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam


def cutmix(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, bbx1:bbx2, bby1:bby2] = x[index, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    target_a, target_b = y, y[index]
    return x, target_a, target_b, lam


def fmix(data, targets, alpha, decay_power, shape, max_soft=0.0, reformulate=False, device=None):
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    x1 = torch.from_numpy(mask).to(device)*data
    x2 = torch.from_numpy(1-mask).to(device)*shuffled_data
    return (x1+x2), targets, shuffled_targets,


def apply_resizemix(X, alphabeta, y=None):
    alpha, beta = alphabeta
    SEG = not isinstance(y, type(None))
    assert alpha > 0, 'alpha should be larger than 0'
    assert beta < 1, 'beta should be smaller than 1'
    batch_size = X.size(0)
    index = torch.randperm(batch_size)
    tau = np.random.uniform(alpha, beta, batch_size)
    lam = tau ** 2
    H, W = X.size()[2:]
    for b in range(batch_size):
        _tau = tau[b]
        patch_size = (int(H*_tau), int(W*_tau))
        resized_X = F.interpolate(X[index[b]].unsqueeze(
            0), size=patch_size, mode='bilinear', align_corners=False).squeeze(0)
        x1, y1, x2, y2 = rand_region((H, W), patch_size)
        X[b, ..., x1:x2, y1:y2] = resized_X
        if SEG:
            resized_y = F.interpolate(y[index[b]].unsqueeze(
                0), size=patch_size, mode='nearest').squeeze(0)
            y[b, ..., x1:x2, y1:y2] = resized_y
    lam = torch.Tensor(lam).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    if SEG:
        return X, y, index, lam
    return X, index, lam


def resizemix(x, y, alpha=0.1, beta=0.8):
    assert alpha > 0, 'alpha should be larger than 0'
    assert beta < 1, 'beta should be smaller than 1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rand_index = torch.randperm(x.size()[0]).to(device)
    tau = np.random.uniform(alpha, beta)
    lam = tau ** 2

    H, W = x.size()[2:]
    resize_transform = Resize((int(H*tau), int(W*tau)))
    resized_x = resize_transform(x[rand_index])

    target_a = y[rand_index]
    target_b = y
    x1, y1, x2, y2 = rand_region(x.size(), resized_x.size())
    x[:, :, y1:y2, x1:x2] = resized_x
    return x, target_a, target_b, lam
