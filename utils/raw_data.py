import numpy as np
import rawpy
import torch

from utils.img_util import denormalize


def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    N, C, _, _ = bayer_images.shape
    outs = bayer_images * wbs.view(N, C, 1, 1)
    return outs


def binning(bayer_images, camera='SonyA7S2', save_image_L118=False):
    """RGBG -> RGB"""
    if camera == 'L118':
        if save_image_L118:
            lin_rgb = torch.stack([
                bayer_images[:, 0, ...],
                bayer_images[:, 1, ...] * 0.7,
                bayer_images[:, 2, ...]], dim=1)
            return lin_rgb
        else:
            lin_rgb = torch.stack([
                bayer_images[:, 0, ...],
                bayer_images[:, 1, ...],
                bayer_images[:, 2, ...]], dim=1)
            return lin_rgb
    else:
        lin_rgb = torch.stack([
            bayer_images[:, 0, ...],
            torch.mean(bayer_images[:, [1, 3], ...], dim=1),
            bayer_images[:, 2, ...]], dim=1)
        return lin_rgb



def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(
        0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    outs = torch.clamp((outs * 255).int(), min=0, max=255).float() / 255
    return outs


def pack_raw_bayer(raw):  # 将bayer模式转换成四通道
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)

    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2, R[1][0]:W:2],  # RGBG
                    im[G1[0][0]:H:2, G1[1][0]:W:2],
                    im[B[0][0]:H:2, B[1][0]:W:2],
                    im[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0).astype(np.float32)
    return out



def depack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    _, H, W = raw.shape
    raw = raw.astype(np.uint16)

    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)

    raw_flatten = np.zeros((H * 2, W * 2))
    raw_flatten[R[0][0]::2, R[1][0]::2] = raw[0]  # 将 4 通道的 RAW 图像数据按照 Bayer 图案排列成一个 2D 数组
    raw_flatten[G1[0][0]::2, G1[1][0]::2] = raw[1]
    raw_flatten[B[0][0]::2, B[1][0]::2] = raw[2]
    raw_flatten[G2[0][0]::2, G2[1][0]::2] = raw[3]

    raw_flatten = raw_flatten.astype(np.uint16)
    return raw_flatten

# 转换成rgb（可选择是否归一化，不归一化就可以直接显示），非裁剪版，输出是np
def postprocess_bayer(rawpath, img4c, is_Normal=False):
    if torch.is_tensor(img4c):
        img4c = img4c.detach()
        img4c = img4c[0].cpu().float().numpy()
    elif isinstance(img4c, np.ndarray):
        img4c = np.copy(img4c)

    img4c = np.clip(img4c, 0, 1)

    # unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern

    black_level = np.array(raw.black_level_per_channel)[:, None, None]

    white_point = 16383

    img4c = img4c * (white_point - black_level) + black_level

    img4c = depack_raw_bayer(img4c, raw_pattern)
    H, W = img4c.shape
    raw.raw_image_visible[:H, :W] = img4c

    out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1,
                          user_black=None, user_sat=None)
    if is_Normal:
        out = out / 255.
    return out


def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns


def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]


def preprocess(img, crop_size=256):
    cropx = crop_size
    cropy = crop_size

    img = crop_center(img, cropx, cropy)
    return img


def read_data(data_path):
    with rawpy.imread(data_path) as raw:
        X = pack_raw_bayer(raw)

        white_level = 16383

        black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)

        X = (X - black_level) / (white_level - black_level)
        X = np.clip(X, 0, 1)
    return X

def ratio_amply_case(input_path):
    parts = input_path.split('/')
    parts = parts[-1].split('_')
    light = parts[0]
    iso = int(parts[2][3:])
    ratio = 1

    if light == "10-1":
        if iso == 1000 or iso == 2000:
            ratio = 5
        elif iso == 4000:
            ratio = 2
    elif light == "10-2":
        if iso == 1000 or iso == 2000:
            ratio = 16
        elif iso == 4000:
            ratio = 8
        elif iso == 8000:
            ratio = 4
        elif iso == 16000:
            ratio = 2
    elif light == "10-3":
        if iso == 1000 or iso == 2000 or iso == 4000:
            ratio = 16
        elif iso == 8000:
            ratio = 8
        elif iso == 16000:
            ratio = 4

    return ratio


def compute_expo_ratio(input_fn, target_fn, camera):
    if camera == 'L118':
        return ratio_amply_case(input_fn)
    else:
        in_exposure = float(input_fn.split('_')[-1][:-5])
        gt_exposure = float(target_fn.split('_')[-1][:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        return ratio


def raw2rgb_batch(rawpath, packed_raw, is_denormal=False, camera='SonyA7S2', save_image_L118=False):
    batch_size, _, h, w = packed_raw.shape
    X = torch.empty(batch_size, 3, h, w)
    for i in range(batch_size):
        if is_denormal:
            packed_raw[i] = denormalize(packed_raw[i])
        X[i] = packed_raw2rgb(rawpath[i], packed_raw[i], camera, save_image_L118)
    return X


# for diy camera isp， output is tensor，range is 0 to 1
def packed_raw2rgb(rawpath, packed_raw, camera='SonyA7S2', save_image_L118=False ,gamma=2.2):
    with rawpy.imread(rawpath) as raw:
        """Raw2RGB pipeline (preprocess version)"""
        if camera == 'L118':
            wb = np.array(raw.daylight_whitebalance)
        else:
            wb = np.array(raw.camera_whitebalance)

        wb /= wb[1]
        cam2rgb = raw.rgb_camera_matrix[:3, :3]

        if isinstance(packed_raw, np.ndarray):
            packed_raw = torch.from_numpy(packed_raw).float()

        wb = torch.from_numpy(wb).float().to(packed_raw.device)
        cam2rgb = torch.from_numpy(cam2rgb).float().to(packed_raw.device)

        out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], camera=camera, gamma=gamma, save_image_L118=save_image_L118)[0, ...]

        return out


def process(bayer_images, wbs, cam2rgbs, camera='SonyA7S2', gamma=2.2, save_image_L118=False, CRF=None):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    # Auto White balance.
    bayer_images = apply_gains(bayer_images, wbs)
    # Binning RGBG -> RGB demonic，直接1/2
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = binning(bayer_images, camera=camera, save_image_L118=save_image_L118)

    # Color Matrix Transformation #
    images = apply_ccms(images, cam2rgbs)

    # Gamma compression. # 用于调整图像亮度的非线性操作
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images)
    return images

