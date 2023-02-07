import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

def manhattan(px, px_old):
    return abs(np.array(px) - np.array(px_old)).sum()

def get_uncalc_neighbours(px, mask, dists):
    ns_ = []
    px_h, px_w = px

    for neighbour in [
            (px_h+1,px_w-1),
            (px_h  ,px_w-1),
            (px_h-1,px_w-1),
            (px_h+1,px_w  ),
            (px_h-1,px_w  ),
            (px_h-1,px_w+1),
            (px_h  ,px_w+1),
            (px_h+1,px_w+1)
            ]:
        if mask[neighbour] and dists[neighbour] == -1:
            ns_.append(neighbour)

    return ns_

def calc_dist(px, px_old, mask, dists):
    r"""
    """
    dists[tuple(px)] = dists[tuple(px_old)] + manhattan(px, px_old)

    pxs = get_uncalc_neighbours(px, mask, dists)
    if pxs == []:
        return

    for px_nxt in pxs:
        calc_dist(px_nxt, px, mask, dists)

def get_fourier_from_mask(img, n_subsamples=50, size_record=None):

    img_dist = distance_transform_edt(img)
    img_contour = img_dist == 1
    dists = np.zeros_like(img_contour) - 1

    if size_record is not None:
        size_record.append(img_dist.max())

    init_px = np.argwhere(img_contour)[0]

    dists[tuple(init_px)] = 0
    calc_dist(init_px, init_px, img_contour, dists)

    positions = np.argwhere(dists != -1)
    sorted_positions = positions[np.argsort(dists[positions[:,0], positions[:,1]])]
    dsts_ = np.sqrt(np.square((sorted_positions - sorted_positions.mean(0))).sum(1))

    ## Subsampling here ##
    step_ = len(dsts_) / n_subsamples
    idxs_ = np.floor(np.arange(n_subsamples) * step_).astype(int)
    dsts_subsampled = dsts_[idxs_] # out is [n_subsamples,]

    sp = np.fft.fft(dsts_subsampled)
    freq = np.fft.fftfreq(sp.shape[-1])

    return sp, freq
