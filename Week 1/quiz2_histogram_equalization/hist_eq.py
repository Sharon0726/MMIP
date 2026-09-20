"""
Quiz 2: Histogram Equalization
基礎: 提升影像明暗對比與動態範圍，繪製處理前後的灰階 Histogram
進階: 用 NumPy 手刻演算法，並與 OpenCV equalizeHist 比較
      (1) 執行速度 (2) 影像增強效果 (3) 兩種方法之間的差異
"""
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_PATH = "images/fruits.jpg"
OUT_DIR = "outputs"
N_REPEAT = 100


def make_low_contrast(gray: np.ndarray, lo=90, hi=165) -> np.ndarray:
    """
    刻意把灰階值壓縮到一個很窄的區間 [lo, hi]，
    模擬起霧 / 曝光不足導致的低對比度影像，這樣才看得出 HE 的效果。
    """
    g = gray.astype(np.float64) / 255.0
    compressed = g * (hi - lo) + lo
    return np.clip(compressed, 0, 255).astype(np.uint8)


def hist_eq_opencv(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def hist_eq_numpy(gray: np.ndarray) -> np.ndarray:
    """
    自行實作 Histogram Equalization：
    1. 統計 256 個灰階值出現的次數 (histogram)
    2. 累加成 CDF (cumulative distribution function)
    3. 把 CDF 正規化到 [0, 255]，做為每個灰階值的新對應值 (LUT)
    4. 用 LUT 對原圖每個像素做映射
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)

    cdf_min = cdf[cdf > 0].min()  # 第一個非零的累積值
    total_pixels = gray.size

    # 教科書上的標準公式: h(v) = round( (cdf(v) - cdf_min) / (total - cdf_min) * 255 )
    lut = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    return lut[gray]


def benchmark(func, img, n=N_REPEAT):
    func(img)
    t0 = time.perf_counter()
    for _ in range(n):
        result = func(img)
    t1 = time.perf_counter()
    return result, (t1 - t0) / n * 1000


def plot_histograms(before, after_np, after_cv, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, img, title in zip(
        axes,
        [before, after_np, after_cv],
        ["Before HE (low contrast)", "After HE - NumPy", "After HE - OpenCV"],
    ):
        ax.hist(img.ravel(), bins=256, range=(0, 255), color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Gray level")
        ax.set_ylabel("Pixel count")
        ax.set_xlim(0, 255)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(IMG_PATH)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    low_contrast = make_low_contrast(gray)

    eq_np, t_np = benchmark(hist_eq_numpy, low_contrast)
    eq_cv, t_cv = benchmark(hist_eq_opencv, low_contrast)

    diff = cv2.absdiff(eq_np, eq_cv)

    cv2.imwrite(f"{OUT_DIR}/gray_original.png", gray)
    cv2.imwrite(f"{OUT_DIR}/low_contrast_input.png", low_contrast)
    cv2.imwrite(f"{OUT_DIR}/eq_numpy.png", eq_np)
    cv2.imwrite(f"{OUT_DIR}/eq_opencv.png", eq_cv)
    cv2.imwrite(f"{OUT_DIR}/diff_numpy_vs_opencv_x10.png", np.clip(diff.astype(np.int32) * 10, 0, 255).astype(np.uint8))

    compare = np.hstack([low_contrast, eq_np, eq_cv])
    cv2.imwrite(f"{OUT_DIR}/compare_before_after.png", compare)

    plot_histograms(low_contrast, eq_np, eq_cv, f"{OUT_DIR}/histograms_before_after.png")

    def dynamic_range(im):
        return int(im.max()) - int(im.min())

    def std(im):
        return float(im.std())

    report = []
    report.append(f"重複執行次數 N = {N_REPEAT}\n")
    report.append("--- 動態範圍 / 對比度 (標準差) ---")
    report.append(f"處理前 (低對比輸入)        : range={dynamic_range(low_contrast)}, std={std(low_contrast):.2f}")
    report.append(f"處理後 - NumPy 手刻 HE     : range={dynamic_range(eq_np)}, std={std(eq_np):.2f}")
    report.append(f"處理後 - OpenCV equalizeHist: range={dynamic_range(eq_cv)}, std={std(eq_cv):.2f}")
    report.append("")
    report.append("--- 執行速度 (平均每次) ---")
    report.append(f"NumPy 手刻 HE      : {t_np:.4f} ms")
    report.append(f"OpenCV equalizeHist: {t_cv:.4f} ms  (NumPy 約為 OpenCV 的 {t_np / t_cv:.1f} 倍)")
    report.append("")
    report.append("--- 轉換結果差異 (NumPy vs OpenCV) ---")
    report.append(f"平均絕對誤差 MAE : {diff.mean():.6f}")
    report.append(f"最大絕對誤差 Max : {diff.max()}")
    report.append(f"完全相同的像素比例: {(diff == 0).mean() * 100:.2f}%")

    text = "\n".join(report)
    print(text)
    with open(f"{OUT_DIR}/report.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
