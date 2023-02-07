def rescale_to(img, to=(0., 255.), eps_=1e-6):
    r"""
    :img: [B,C,*]
    """
    outp_min = to[0]
    outp_max = to[1]

    outp_ = img - img.min()
    outp_ = outp_ / (outp_.max() + eps_)
    outp_ = outp_ * (outp_max - outp_min)
    outp_ = outp_ + outp_min

    return outp_

