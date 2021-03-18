import cv2
import numpy as np
from numpy import linalg as npla
import scipy as sp


def color_transfer_sot(src, trg, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0):
    """
    Color Transform via Sliced Optimal Transfer
    ported by @iperov from https://github.com/dcoeurjo/OTColorTransfer

    src         - any float range any channel image
    dst         - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter

    return value - clip it manually
    """
    if not np.issubdtype(src.dtype, np.floating):
        raise ValueError("src value must be float")
    if not np.issubdtype(trg.dtype, np.floating):
        raise ValueError("trg value must be float")

    if len(src.shape) != 3:
        raise ValueError("src shape must have rank 3 (h,w,c)")

    if src.shape != trg.shape:
        raise ValueError("src and trg shapes must be equal")

    src_dtype = src.dtype
    h, w, c = src.shape
    new_src = src.copy()

    advect = np.empty((h * w, c), dtype=src_dtype)
    for step in range(steps):
        advect.fill(0)
        for batch in range(batch_size):
            dir = np.random.normal(size=c).astype(src_dtype)
            dir /= npla.norm(dir)

            projsource = np.sum(new_src * dir, axis=-1).reshape((h * w))
            projtarget = np.sum(trg * dir, axis=-1).reshape((h * w))

            idSource = np.argsort(projsource)
            idTarget = np.argsort(projtarget)

            a = projtarget[idTarget] - projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += a * dir[i_c]
        new_src += advect.reshape((h, w, c)) / batch_size

    if reg_sigmaXY != 0.0:
        src_diff = new_src - src
        src_diff_filt = cv2.bilateralFilter(src_diff, 0, reg_sigmaV, reg_sigmaXY)
        if len(src_diff_filt.shape) == 2:
            src_diff_filt = src_diff_filt[..., None]
        new_src = src + src_diff_filt
    return new_src


def color_transfer_mkl(x0, x1):
    eps = np.finfo(float).eps

    h, w, c = x0.shape
    h1, w1, c1 = x1.shape

    x0 = x0.reshape((h * w, c))
    x1 = x1.reshape((h1 * w1, c1))

    a = np.cov(x0.T)
    b = np.cov(x1.T)

    Da2, Ua = np.linalg.eig(a)
    Da = np.diag(np.sqrt(Da2.clip(eps, None)))

    C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = np.linalg.eig(C)
    Dc = np.diag(np.sqrt(Dc2.clip(eps, None)))

    Da_inv = np.diag(1. / (np.diag(Da)))

    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T)

    mx0 = np.mean(x0, axis=0)
    mx1 = np.mean(x1, axis=0)

    result = np.dot(x0 - mx0, t) + mx1
    return np.clip(result.reshape((h, w, c)).astype(x0.dtype), 0, 1)


def color_transfer_idt(i0, i1, bins=256, n_rot=20):
    import scipy.stats

    relaxation = 1 / n_rot
    h, w, c = i0.shape
    h1, w1, c1 = i1.shape

    i0 = i0.reshape((h * w, c))
    i1 = i1.reshape((h1 * w1, c1))

    n_dims = c

    d0 = i0.T
    d1 = i1.T

    for i in range(n_rot):

        r = sp.stats.special_ortho_group.rvs(n_dims).astype(np.float32)

        d0r = np.dot(r, d0)
        d1r = np.dot(r, d1)
        d_r = np.empty_like(d0)

        for j in range(n_dims):
            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())

            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _ = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(np.float32)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1]

            f = np.interp(cp0r, cp1r, edges[1:])

            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

        d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0

    return np.clip(d0.T.reshape((h, w, c)).astype(i0.dtype), 0, 1)


def reinhard_color_transfer(source, target, clip=False, preserve_paper=False, source_mask=None, target_mask=None):
    """
	Transfers the color distribution from the source to the target
	image using the mean and standard deviations of the L*a*b*
	color space.

	This implementation is (loosely) based on to the "Color Transfer
	between Images" paper by Reinhard et al., 2001.
	Url: https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf

	Title: "Super fast color transfer between images"
    Author: Adrian Rosebrock
    Date: June 30. 2014
    Url: https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/

	Parameters:
	-------
	source: NumPy array
		OpenCV image (w, h, 3) in BGR color space (the source image) (float32)
		The image to be modified
	target: NumPy array
		OpenCV image (w, h, 3) in BGR color space (the target image) (float32)
		The image containing the colorspace we wish to mimic
	clip: Should components of L*a*b* image be scaled by np.clip before
		converting back to BGR color space?
		If False then components will be min-max scaled appropriately.
		Clipping will keep target image brightness truer to the input.
		Scaling will adjust image brightness to avoid washed out portions
		in the resulting color transfer that can be caused by clipping.
	preserve_paper: Should color transfer strictly follow methodology
		layed out in original paper? The method does not always produce
		aesthetically pleasing results.
		If False then L*a*b* components will scaled using the reciprocal of
		the scaling factor proposed in the paper.  This method seems to produce
		more consistently aesthetically pleasing results
	source_mask: The mask for the source image
	target_mask: The mask for the target image

	Returns:
	-------
	transfer: NumPy array
		OpenCV image (w, h, 3) NumPy array (float32)
	"""

    # FIXME: debug
    print('source.shape: ', source.shape)
    print('source_mask.shape: ', source_mask.shape)
    print('target.shape: ', target.shape)
    print('target_mask.shape: ', target_mask.shape)

    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = lab_image_stats(source, mask=source_mask)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = lab_image_stats(target, mask=target_mask)

    # subtract the means from the source image
    (l, a, b) = cv2.split(source)
    l -= lMeanSrc
    a -= aMeanSrc
    b -= bMeanSrc

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        l = (lStdTar / lStdSrc) * l if lStdSrc != 0 else l
        a = (aStdTar / aStdSrc) * a if aStdSrc != 0 else a
        b = (bStdTar / bStdSrc) * b if bStdSrc != 0 else b
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l = (lStdSrc / lStdTar) * l if lStdTar != 0 else l
        a = (aStdSrc / aStdTar) * a if aStdTar != 0 else a
        b = (bStdSrc / bStdTar) * b if bStdTar != 0 else b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities if they fall
    # outside the ranges for LAB
    # For 32-bit images, OpenCV uses L=[0, 100], a=[-127, 127], b=[-127, 127]
    l = _scale_array(l, 0, 100, clip=clip, mask=source_mask)
    a = _scale_array(a, -127, 127, clip=clip, mask=source_mask)
    b = _scale_array(b, -127, 127, clip=clip, mask=source_mask)

    # merge the channels together and convert back to the BGR colorspace
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype(np.uint8), cv2.COLOR_LAB2BGR)
    np.clip(transfer, 0, 1, out=transfer)

    # return the color transferred image
    return transfer


def linear_color_transfer(target_img, source_img, mode='pca', eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c) and float in [0,1].
    Modes are chol, pca or sym for different choices of basis.
    '''
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2, 0, 1).reshape(t.shape[-1], -1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2, 0, 1).reshape(s.shape[-1], -1)
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
    matched_img = ts.reshape(*target_img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    matched_img += mu_s
    matched_img[matched_img > 1] = 1
    matched_img[matched_img < 0] = 0
    return np.clip(matched_img.astype(source_img.dtype), 0, 1)


def lab_image_stats(image, mask=None):
    # compute the mean and standard deviation of each channel
    l, a, b = cv2.split(image)

    if mask is not None:
        # If mask has shape (w,h,c), remove the channel axis, and convert to (w,h)
        im_mask = np.squeeze(mask) if len(np.shape(mask)) == 3 else mask
        # Filter the LAB channels on only the masked areas,
        # so our statistics are calculated on the masked region
        l, a, b = l[im_mask == 1], a[im_mask == 1], b[im_mask == 1]

    l_mean, l_std = np.mean(l), np.std(l)
    a_mean, a_std = np.mean(a), np.std(a)
    b_mean, b_std = np.mean(b), np.std(b)

    # return the color statistics
    return l_mean, l_std, a_mean, a_std, b_mean, b_std


def _scale_array(arr, min_val, max_val, clip=True, mask=None):
    """
    Limit values in an array, with option of clipping or scaling.

    With clip enabled, values outside the given interval are clipped to the interval edges.
    With clip disabled, the array is appropriately scaled so that all values are within the interval.
    Optionally, a mask may be provided, and the array will be scaled so all values within the masked
    area are within the interval (values outside of the mask are scaled as well, but are not
    guaranteed to be within the interval range.

    Parameters:
    -------
    arr: array to be trimmed to range
    min_val: minimum value
    max_val: maximum value
    clip: should array be limited by np.clip? if False then input
        array will be min-max scaled to range
        [max(arr.min(), min_val), min(arr.max(), max_val)]
    Returns:
    -------
    NumPy array that has been limited to [min_val, max_val] range for masked region
    if a mask is provided, otherwise for the entire array.
    """
    if clip:
        return np.clip(arr, min_val, max_val)

    if mask is not None:
        source_min = np.min(mask * arr)
        source_max = np.max(mask * arr)
    else:
        source_min = np.min(arr)
        source_max = np.max(arr)

    if min_val <= source_min and source_max <= max_val:
        # Return the original array if all values are within bounds
        return arr

    target_min = max(source_min, min_val)
    target_max = min(source_max, max_val)

    return (arr - source_min) * (target_max - target_min) / (source_max - source_min) + target_min


def channel_hist_match(source, template, hist_match_threshold=255, mask=None):
    # Code borrowed from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    masked_source = source
    masked_template = template

    if mask is not None:
        masked_source = source * mask
        masked_template = template * mask

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    masked_source = masked_source.ravel()
    masked_template = masked_template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles = hist_match_threshold * s_quantiles / s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles = 255 * t_quantiles / t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def color_hist_match(src_im, tar_im, hist_match_threshold=255):
    h, w, c = src_im.shape
    matched_R = channel_hist_match(src_im[:, :, 0], tar_im[:, :, 0], hist_match_threshold, None)
    matched_G = channel_hist_match(src_im[:, :, 1], tar_im[:, :, 1], hist_match_threshold, None)
    matched_B = channel_hist_match(src_im[:, :, 2], tar_im[:, :, 2], hist_match_threshold, None)

    to_stack = (matched_R, matched_G, matched_B)
    for i in range(3, c):
        to_stack += (src_im[:, :, i],)

    matched = np.stack(to_stack, axis=-1).astype(src_im.dtype)
    return matched


def color_transfer_mix(img_src, img_trg):
    img_src = np.clip(img_src * 255.0, 0, 255).astype(np.uint8)
    img_trg = np.clip(img_trg * 255.0, 0, 255).astype(np.uint8)

    img_src_lab = cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
    img_trg_lab = cv2.cvtColor(img_trg, cv2.COLOR_BGR2LAB)

    rct_light = np.clip(linear_color_transfer(img_src_lab[..., 0:1].astype(np.float32) / 255.0,
                                              img_trg_lab[..., 0:1].astype(np.float32) / 255.0)[..., 0] * 255.0,
                        0, 255).astype(np.uint8)

    img_src_lab[..., 0] = (np.ones_like(rct_light) * 100).astype(np.uint8)
    img_src_lab = cv2.cvtColor(img_src_lab, cv2.COLOR_LAB2BGR)

    img_trg_lab[..., 0] = (np.ones_like(rct_light) * 100).astype(np.uint8)
    img_trg_lab = cv2.cvtColor(img_trg_lab, cv2.COLOR_LAB2BGR)

    img_rct = color_transfer_sot(img_src_lab.astype(np.float32), img_trg_lab.astype(np.float32))
    img_rct = np.clip(img_rct, 0, 255).astype(np.uint8)

    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_BGR2LAB)
    img_rct[..., 0] = rct_light
    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_LAB2BGR)

    return (img_rct / 255.0).astype(np.float32)


def color_transfer(ct_mode, img_src, img_trg, img_src_mask=None, img_trg_mask=None):
    """
    color transfer for [0,1] float32 inputs
    """
    if ct_mode == 'lct':
        out = linear_color_transfer(img_src, img_trg)
    elif ct_mode in ['rct', 'rct-c', 'rct-p', 'rct-pc', 'rct-m', 'rct-mc', 'rct-mp', 'rct-mpc']:
        rct_options = list(*ct_mode.split('-')[1:])
        clip = 'c' in rct_options
        preserve_paper = 'p' in rct_options
        source_mask = img_src_mask if 'm' in rct_options else None
        target_mask = img_trg_mask if 'm' in rct_options else None
        out = reinhard_color_transfer(img_src, img_trg, clip=clip, preserve_paper=preserve_paper,
                                      source_mask=source_mask, target_mask=target_mask)
    elif ct_mode == 'mkl':
        out = color_transfer_mkl(img_src, img_trg)
    elif ct_mode == 'idt':
        out = color_transfer_idt(img_src, img_trg)
    elif ct_mode == 'sot':
        out = color_transfer_sot(img_src, img_trg)
        out = np.clip(out, 0.0, 1.0)
    else:
        raise ValueError(f"unknown ct_mode {ct_mode}")
    return out
