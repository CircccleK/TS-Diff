from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from utils.registry import METRIC_REGISTRY


# 使用 skimage.metrics 库中的 ssim 函数，专注于直接计算图像的结构相似性，并将图像转换为灰度图
# 计算方式和ELD一致，并且data_range在tensor2img已经调整为255，并非raw的数据范围
@METRIC_REGISTRY.register()
def calculate_ssim_skimage(img, img2, data_range=255):
    ssim = structural_similarity(img2, img, data_range=data_range, multichannel=True)
    return ssim


@METRIC_REGISTRY.register()
def calculate_psnr_skimage(img, img2, data_range=255):
    psnr = peak_signal_noise_ratio(img2, img, data_range=data_range)
    return psnr


