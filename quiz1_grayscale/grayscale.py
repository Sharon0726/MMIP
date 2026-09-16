"""
Quiz 1: 彩色影像轉灰階影像
基礎: RGB -> 灰階
進階: 用 NumPy 手刻灰階轉換演算法，並與 OpenCV cvtColor 比較
      (1) 執行速度 (2) 轉換結果 (3) 兩種方法之間的差異
"""
import time
import cv2
import numpy as np

IMG_PATH = "images/fruits.jpg"
OUT_DIR = "outputs"
N_REPEAT = 100  # 重複執行次數，取平均


def grayscale_opencv(img_bgr: np.ndarray) -> np.ndarray:
    """OpenCV 內建方法。內部同樣使用 ITU-R BT.601 加權係數。"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def grayscale_numpy_weighted(img_bgr: np.ndarray) -> np.ndarray:
    """
    自行實作：加權平均法 (Luminosity method)
    Gray = 0.299*R + 0.587*G + 0.114*B
    人眼對綠色最敏感、藍色最不敏感，因此三個通道的權重不同。
    OpenCV 讀入的影像是 BGR 順序，所以係數要對應反過來乘。
    """
    b = img_bgr[:, :, 0].astype(np.float64)
    g = img_bgr[:, :, 1].astype(np.float64)
    r = img_bgr[:, :, 2].astype(np.float64)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.round(gray).astype(np.uint8)


def grayscale_numpy_average(img_bgr: np.ndarray) -> np.ndarray:
    """自行實作：單純平均法 Gray = (R+G+B)/3，作為對照組。"""
    gray = img_bgr.astype(np.float64).mean(axis=2)
    return np.round(gray).astype(np.uint8)


def benchmark(func, img_bgr, n=N_REPEAT):
    # 先跑一次 warm-up，避免第一次呼叫的額外開銷影響計時
    func(img_bgr)
    t0 = time.perf_counter()
    for _ in range(n):
        result = func(img_bgr)
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / n * 1000
    return result, avg_ms


def main():
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        raise FileNotFoundError(IMG_PATH)
    print(f"輸入影像: {IMG_PATH}, shape={img_bgr.shape}")

    gray_cv, t_cv = benchmark(grayscale_opencv, img_bgr)
    gray_np, t_np = benchmark(grayscale_numpy_weighted, img_bgr)
    gray_avg, t_avg = benchmark(grayscale_numpy_average, img_bgr)

    diff_cv_np = cv2.absdiff(gray_cv, gray_np)
    diff_cv_avg = cv2.absdiff(gray_cv, gray_avg)

    cv2.imwrite(f"{OUT_DIR}/gray_opencv.png", gray_cv)
    cv2.imwrite(f"{OUT_DIR}/gray_numpy_weighted.png", gray_np)
    cv2.imwrite(f"{OUT_DIR}/gray_numpy_average.png", gray_avg)
    cv2.imwrite(f"{OUT_DIR}/diff_weighted_vs_opencv_x10.png", np.clip(diff_cv_np.astype(np.int32) * 10, 0, 255).astype(np.uint8))
    cv2.imwrite(f"{OUT_DIR}/diff_average_vs_opencv_x10.png", np.clip(diff_cv_avg.astype(np.int32) * 10, 0, 255).astype(np.uint8))

    # 三張圖並排比較
    compare = np.hstack([gray_cv, gray_np, gray_avg])
    cv2.imwrite(f"{OUT_DIR}/compare_side_by_side.png", compare)

    report = []
    report.append(f"重複執行次數 N = {N_REPEAT}\n")
    report.append("--- 執行速度 (平均每次) ---")
    report.append(f"OpenCV cvtColor      : {t_cv:.4f} ms")
    report.append(f"NumPy 加權平均法手刻  : {t_np:.4f} ms  (約為 OpenCV 的 {t_np / t_cv:.1f} 倍)")
    report.append(f"NumPy 單純平均法手刻  : {t_avg:.4f} ms  (約為 OpenCV 的 {t_avg / t_cv:.1f} 倍)")
    report.append("")
    report.append("--- 轉換結果差異 (NumPy 加權法 vs OpenCV) ---")
    report.append(f"平均絕對誤差 MAE : {diff_cv_np.mean():.6f}")
    report.append(f"最大絕對誤差 Max : {diff_cv_np.max()}")
    report.append(f"完全相同的像素比例: {(diff_cv_np == 0).mean() * 100:.2f}%")
    report.append("")
    report.append("--- 轉換結果差異 (NumPy 單純平均法 vs OpenCV 加權法) ---")
    report.append(f"平均絕對誤差 MAE : {diff_cv_avg.mean():.6f}")
    report.append(f"最大絕對誤差 Max : {diff_cv_avg.max()}")

    text = "\n".join(report)
    print(text)
    with open(f"{OUT_DIR}/report.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
