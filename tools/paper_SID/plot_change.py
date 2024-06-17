import matplotlib.pyplot as plt

# 数据
extra_data_pairs = [3, 6, 18, 35, 70, 280]

# PSNR 数据
psnr_100 = [38.52, 38.68, 39.30, 39.31, 39.19, 39.23]
psnr_250 = [36.97, 36.78, 37.17, 37.39, 37.24, 37.15]
psnr_300 = [36.10, 36.12, 36.40, 36.71, 36.51, 36.47]

# SSIM 数据
ssim_100 = [0.897, 0.900, 0.911, 0.914, 0.912, 0.913]
ssim_250 = [0.878, 0.873, 0.876, 0.883, 0.884, 0.882]
ssim_300 = [0.862, 0.862, 0.865, 0.872, 0.873, 0.872]

# 基准线数据
psnr_baseline_100 = 39.27
ssim_baseline_100 = 0.914
psnr_baseline_250 = 37.13
ssim_baseline_250 = 0.883
psnr_baseline_300 = 36.30
ssim_baseline_300 = 0.872

# 绘制 PSNR 曲线
plt.figure(figsize=(10, 6))
plt.plot(extra_data_pairs, psnr_100, marker='o', label='TS-Diff PSNR ×100')
plt.plot(extra_data_pairs, psnr_250, marker='o', label='TS-Diff PSNR ×250')
plt.plot(extra_data_pairs, psnr_300, marker='o', label='TS-Diff PSNR ×300')

# 添加基准线
plt.axhline(y=psnr_baseline_100, color='r', linestyle='--', label='ELD PSNR ×100')
plt.axhline(y=psnr_baseline_250, color='g', linestyle='--', label='ELD PSNR ×250')
plt.axhline(y=psnr_baseline_300, color='b', linestyle='--', label='ELD PSNR ×300')

# 图表设置
plt.xlabel("Extra Data Pairs")
plt.ylabel("PSNR")
plt.grid()
plt.legend(loc='upper right')  # 图例统一放在右上角
plt.tight_layout()
plt.savefig("psnr_with_baseline.png", dpi=300)
plt.show()

# 绘制 SSIM 曲线
plt.figure(figsize=(10, 6))
plt.plot(extra_data_pairs, ssim_100, marker='o', label='TS-Diff SSIM ×100')
plt.plot(extra_data_pairs, ssim_250, marker='o', label='TS-Diff SSIM ×250')
plt.plot(extra_data_pairs, ssim_300, marker='o', label='TS-Diff SSIM ×300')

# 添加基准线
plt.axhline(y=ssim_baseline_100, color='r', linestyle='--', label='ELD SSIM ×100')
plt.axhline(y=ssim_baseline_250, color='g', linestyle='--', label='ELD SSIM ×250')
plt.axhline(y=ssim_baseline_300, color='b', linestyle='--', label='ELD SSIM ×300')

# 图表设置
plt.xlabel("Extra Data Pairs")
plt.ylabel("SSIM")
plt.grid()
plt.legend(loc='upper right')  # 图例统一放在右上角
plt.tight_layout()
plt.savefig("ssim_with_baseline.png", dpi=300)
plt.show()