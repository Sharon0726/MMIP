"""
Quiz 4: 影像拼接 (Image Stitching)

基礎: 兩張有重疊區域的影像，用 SIFT 特徵偵測 + 匹配完成拼接。
進階: 調整亮度、拍攝角度、重疊範圍，觀察拼接效果，
      找出開始失敗的條件並說明原因；並嘗試前處理 (CLAHE) 提升穩定性。

手上沒有真的兩張重疊照片，所以用同一張紋理豐富的照片 (電路板) 切出
左右兩塊有重疊的區域，並對右半邊施加「模擬第二次拍攝」的擾動
(旋轉、亮度變化)，這樣就能用同一份 ground truth 來源客觀評估拼接品質，
也能有系統地掃過不同亮度/角度/重疊率組合。
"""
import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_PATH = "images/board.jpg"
OUT_DIR = "outputs"
RATIO_THRESH = 0.75  # Lowe's ratio test
RANSAC_THRESH = 4.0


def make_pair(img_bgr, overlap_ratio=0.35, angle_deg=0.0, brightness_factor=1.0):
    """從同一張來源影像切出左右兩塊有重疊的影像，並在右塊上模擬第二次拍攝的差異。"""
    h, w = img_bgr.shape[:2]
    cw = int(w / (2 - overlap_ratio))  # 每塊的寬度
    left = img_bgr[:, 0:cw].copy()
    right = img_bgr[:, w - cw:w].copy()

    # 模擬拍攝角度差異: 對右塊做小角度旋轉
    if angle_deg != 0:
        center = (right.shape[1] / 2, right.shape[0] / 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        right = cv2.warpAffine(right, M, (right.shape[1], right.shape[0]),
                                borderMode=cv2.BORDER_REPLICATE)

    # 模擬亮度/曝光差異
    if brightness_factor != 1.0:
        right = np.clip(right.astype(np.float64) * brightness_factor, 0, 255).astype(np.uint8)

    return left, right


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def match_features(left_bgr, right_bgr, preprocess=False):
    """SIFT 偵測 + KNN 匹配 + Lowe's ratio test + RANSAC 估計 homography。"""
    gray_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    if preprocess:
        gray_l = apply_clahe(gray_l)
        gray_r = apply_clahe(gray_r)

    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(gray_l, None)
    kp_r, des_r = sift.detectAndCompute(gray_r, None)

    result = {
        "n_kp_left": len(kp_l), "n_kp_right": len(kp_r),
        "n_good_matches": 0, "n_inliers": 0, "inlier_ratio": 0.0,
        "mean_reproj_error": None, "success": False,
    }

    if des_l is None or des_r is None or len(kp_l) < 4 or len(kp_r) < 4:
        return result, None, None, []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des_r, des_l, k=2)  # 把 right 的特徵去 left 裡面找對應
    good = [m for m, n in knn if m.distance < RATIO_THRESH * n.distance]
    result["n_good_matches"] = len(good)

    if len(good) < 4:
        return result, None, kp_l, kp_r

    pts_r = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_l = np.float32([kp_l[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(pts_r, pts_l, cv2.RANSAC, RANSAC_THRESH)

    if H is None:
        return result, None, kp_l, kp_r

    inliers = int(mask.sum())
    result["n_inliers"] = inliers
    result["inlier_ratio"] = inliers / len(good)

    # 平均重投影誤差 (只算 inlier)
    pts_r_h = cv2.perspectiveTransform(pts_r, H)
    errs = np.linalg.norm(pts_r_h.reshape(-1, 2) - pts_l.reshape(-1, 2), axis=1)
    inlier_mask = mask.ravel().astype(bool)
    if inlier_mask.sum() > 0:
        result["mean_reproj_error"] = float(errs[inlier_mask].mean())

    result["success"] = inliers >= 8 and result["inlier_ratio"] >= 0.5
    return result, H, (kp_l, good, mask), (kp_r, gray_l, gray_r)


def stitch(left_bgr, right_bgr, H):
    """已知 homography 後，把 right 影像 warp 到 left 的座標系並做羽化混合。"""
    h_l, w_l = left_bgr.shape[:2]
    h_r, w_r = right_bgr.shape[:2]

    corners_r = np.float32([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]]).reshape(-1, 1, 2)
    corners_r_warp = cv2.perspectiveTransform(corners_r, H)
    corners_l = np.float32([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners_l, corners_r_warp], axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    tx, ty = -min(x_min, 0), -min(y_min, 0)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    canvas_w, canvas_h = x_max + tx, y_max + ty
    canvas_w, canvas_h = int(canvas_w), int(canvas_h)

    warped_right = cv2.warpPerspective(right_bgr, T @ H, (canvas_w, canvas_h))
    canvas_left = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_left[ty:ty + h_l, tx:tx + w_l] = left_bgr

    mask_left = (canvas_left.sum(axis=2) > 0).astype(np.float32)
    mask_right = (warped_right.sum(axis=2) > 0).astype(np.float32)
    overlap = (mask_left > 0) & (mask_right > 0)

    # 重疊區域用羽化混合 (依到左邊界的距離線性加權)，避免明顯接縫
    alpha = mask_left.copy()
    if overlap.any():
        xs = np.where(overlap.any(axis=0))[0]
        x0, x1 = xs.min(), xs.max()
        ramp = np.clip((x1 - np.arange(canvas_w)) / max(x1 - x0, 1), 0, 1)
        alpha_overlap = np.tile(ramp, (canvas_h, 1))
        alpha = np.where(overlap, alpha_overlap, alpha)

    alpha3 = alpha[:, :, None]
    blended = canvas_left.astype(np.float32) * alpha3 + warped_right.astype(np.float32) * (1 - alpha3)
    blended = np.where((mask_left[:, :, None] == 0) & (mask_right[:, :, None] > 0), warped_right, blended)
    blended = np.where((mask_right[:, :, None] == 0) & (mask_left[:, :, None] > 0), canvas_left, blended)
    return np.clip(blended, 0, 255).astype(np.uint8)


def run_basic_demo(src):
    left, right = make_pair(src, overlap_ratio=0.35, angle_deg=6, brightness_factor=0.85)
    cv2.imwrite(f"{OUT_DIR}/basic_left.png", left)
    cv2.imwrite(f"{OUT_DIR}/basic_right.png", right)

    result, H, extra_l, extra_r = match_features(left, right)
    print("[基礎題] 特徵匹配結果:", json.dumps(result, ensure_ascii=False, indent=2))

    if H is not None:
        kp_l, good, mask = extra_l
        kp_r, gray_l, gray_r = extra_r
        # 只隨機挑一部分 inlier 連線畫出來，避免整張圖被線條塞滿看不清楚
        rng = np.random.default_rng(0)
        inlier_idx = np.where(mask.ravel() == 1)[0]
        show_idx = set(rng.choice(inlier_idx, size=min(40, len(inlier_idx)), replace=False).tolist())
        vis_mask = [1 if i in show_idx else 0 for i in range(len(good))]
        match_vis = cv2.drawMatches(
            right, kp_r,
            left, kp_l,
            good, None,
            matchesMask=vis_mask,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(f"{OUT_DIR}/basic_matches.png", match_vis)

        panorama = stitch(left, right, H)
        cv2.imwrite(f"{OUT_DIR}/basic_panorama.png", panorama)
    with open(f"{OUT_DIR}/basic_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def run_advanced_sweep(src):
    os.makedirs(OUT_DIR, exist_ok=True)

    sweeps = {
        "overlap_ratio": [0.5, 0.4, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03],
        "angle_deg": [0, 5, 10, 15, 20, 25, 30, 40],
        "brightness_factor": [1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12, 0.08],
    }
    defaults = {"overlap_ratio": 0.35, "angle_deg": 6, "brightness_factor": 0.85}

    all_results = {}
    for param, values in sweeps.items():
        rows = []
        for v in values:
            kwargs = dict(defaults)
            kwargs[param] = v
            left, right = make_pair(src, **kwargs)
            result, H, _, _ = match_features(left, right)
            rows.append(result)
        all_results[param] = {"values": values, "rows": rows}

    # CLAHE 前處理在「低亮度」情境下的效果比較
    clahe_rows_raw, clahe_rows_pre = [], []
    brightness_values = sweeps["brightness_factor"]
    for v in brightness_values:
        left, right = make_pair(src, overlap_ratio=defaults["overlap_ratio"],
                                 angle_deg=defaults["angle_deg"], brightness_factor=v)
        r_raw, _, _, _ = match_features(left, right, preprocess=False)
        r_pre, _, _, _ = match_features(left, right, preprocess=True)
        clahe_rows_raw.append(r_raw)
        clahe_rows_pre.append(r_pre)

    # --- 畫圖 ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    titles = {"overlap_ratio": "Overlap ratio", "angle_deg": "Rotation angle (deg)",
              "brightness_factor": "Brightness factor"}
    for ax, param in zip(axes, sweeps.keys()):
        values = all_results[param]["values"]
        rows = all_results[param]["rows"]
        matches = [r["n_good_matches"] for r in rows]
        inliers = [r["n_inliers"] for r in rows]
        ax.plot(values, matches, marker="o", label="good matches")
        ax.plot(values, inliers, marker="s", label="RANSAC inliers")
        ax.set_xlabel(titles[param])
        ax.set_ylabel("count")
        ax.set_title(f"Stitching robustness vs. {titles[param]}")
        ax.legend()
        ax.grid(alpha=0.3)
        if param == "overlap_ratio" or param == "brightness_factor":
            ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/advanced_sweep.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(brightness_values, [r["n_inliers"] for r in clahe_rows_raw], marker="o", label="No preprocessing")
    ax.plot(brightness_values, [r["n_inliers"] for r in clahe_rows_pre], marker="s", label="With CLAHE")
    ax.invert_xaxis()
    ax.set_xlabel("Brightness factor of the second shot")
    ax.set_ylabel("RANSAC inliers")
    ax.set_title("CLAHE preprocessing improves matching under low light")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/advanced_clahe_comparison.png", dpi=120)
    plt.close(fig)

    # 找出各參數第一次「失敗」(success=False) 的位置
    first_failure = {}
    for param, values in sweeps.items():
        rows = all_results[param]["rows"]
        for v, r in zip(values, rows):
            if not r["success"]:
                first_failure[param] = v
                break

    summary = {
        "defaults": defaults,
        "sweeps": {k: v["rows"] for k, v in all_results.items()},
        "sweep_values": sweeps,
        "first_failure_condition": first_failure,
        "clahe_low_light": {
            "brightness_values": brightness_values,
            "no_preprocess_inliers": [r["n_inliers"] for r in clahe_rows_raw],
            "clahe_inliers": [r["n_inliers"] for r in clahe_rows_pre],
        },
    }
    with open(f"{OUT_DIR}/advanced_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[進階題] 各參數第一次拼接失敗發生在:")
    for k, v in first_failure.items():
        print(f"  {k}: {v}")
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    src = cv2.imread(SRC_PATH)
    if src is None:
        raise FileNotFoundError(SRC_PATH)
    run_basic_demo(src)
    run_advanced_sweep(src)


if __name__ == "__main__":
    main()
