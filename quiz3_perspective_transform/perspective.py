"""
Quiz 3: 梯形校正與透視轉換 (Perspective Transformation)

基礎: 選一張斜拍影像，用 Perspective Transformation 校正為正視影像。
進階: 將演算法應用於多張不同拍攝條件的影像，自動化完成梯形校正；
      並探討不同拍攝角度下的校正效果，找出演算法開始失效的角度/條件並說明原因。

因為手上沒有真的「斜拍照片 + 已知正確答案」可以拿來驗證校正準不準，
所以採用教科書上常見的做法：自己合成一張「已知內容」的平面文件，
用相機投影幾何模擬把它斜著拍下來，再去偵測 4 個角點、做透視校正，
最後跟原始內容比對 SSIM，就能客觀量化校正得好不好、在幾度時開始壞掉。
"""
import os
import math
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

OUT_DIR = "outputs"
CANVAS = 700  # 背景畫布大小 (模擬拍照時周圍的桌面/背景)
CONTENT_SIZE = 360  # 文件內容 (正視角時) 的邊長


def make_document(content_bgr: np.ndarray) -> np.ndarray:
    """把內容圖片縮放成正方形文件，加上白色邊框，看起來像一張照片/文件。"""
    content = cv2.resize(content_bgr, (CONTENT_SIZE, CONTENT_SIZE))
    border = 18
    doc = cv2.copyMakeBorder(content, border, border, border, border,
                              cv2.BORDER_CONSTANT, value=(245, 245, 245))
    return doc


def project_tilted_plane(size, theta_deg, depth=2.4, f=1.0):
    """
    用簡化的針孔相機模型，模擬「從正前方傾斜 theta 度看一個正方形平面」
    後，這個平面 4 個角點會投影到相機影像平面上的哪個位置。
    傾斜角度越大 -> 遠離相機那側的邊越被壓縮 (透視梯形效果越明顯)。
    回傳的是「以畫布中心為原點」的角點座標 (單位: 畫布像素)。
    """
    theta = math.radians(theta_deg)
    half = 1.0  # 平面半邊長 (正規化座標)
    corners = np.array([
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ])
    # 繞 X 軸旋轉，模擬鏡頭由上往下(或由下往上)斜看這個平面
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t],
    ])
    rotated = corners @ R.T
    rotated[:, 2] += depth  # 把平面推到相機前方 depth 處

    scale = size * 0.32
    proj_x = rotated[:, 0] / rotated[:, 2] * f * scale
    proj_y = rotated[:, 1] / rotated[:, 2] * f * scale
    return np.stack([proj_x, proj_y], axis=1)


def synthesize_oblique_photo(doc_bgr: np.ndarray, theta_deg: float, canvas=CANVAS):
    """把正視文件 doc_bgr 用透視變形「貼」到一張背景畫布上，模擬斜拍照片。"""
    h, w = doc_bgr.shape[:2]
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    dst_offset = project_tilted_plane(canvas, theta_deg)
    center = np.array([canvas / 2, canvas / 2])
    dst_pts = (dst_offset + center).astype(np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 背景: 用簡單的木紋色漸層 + 雜訊，模擬拍照時的桌面背景
    rng = np.random.default_rng(0)
    bg = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    bg[:] = (60, 90, 120)
    noise = rng.integers(-15, 15, size=bg.shape, endpoint=True)
    bg = np.clip(bg.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    warped_doc = cv2.warpPerspective(doc_bgr, H, (canvas, canvas))
    mask = cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, H, (canvas, canvas))
    mask3 = cv2.merge([mask, mask, mask])

    photo = np.where(mask3 > 0, warped_doc, bg)
    true_corners = dst_pts
    return photo, true_corners


def order_points(pts: np.ndarray) -> np.ndarray:
    """把 4 個角點排序成 [左上, 右上, 右下, 左下]，透視轉換的來源點順序一定要固定。"""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left     : x+y 最小
    ordered[2] = pts[np.argmax(s)]   # bottom-right : x+y 最大
    ordered[1] = pts[np.argmin(d)]   # top-right    : x-y 最小
    ordered[3] = pts[np.argmax(d)]   # bottom-left  : x-y 最大
    return ordered


def detect_document_corners(photo_bgr: np.ndarray):
    """
    經典「文件掃描」流程: 灰階 -> 模糊去雜訊 -> Canny 邊緣 -> 膨脹補洞
    -> 找輪廓 -> 依面積排序 -> 用 approxPolyDP 找出第一個近似為四邊形的輪廓。
    找不到就回傳 None (代表這個角度校正失敗)。
    """
    gray = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.02 * photo_bgr.shape[0] * photo_bgr.shape[1]:
            return order_points(approx.astype(np.float32))
    return None


def correct_perspective(photo_bgr: np.ndarray, corners: np.ndarray, out_size=CONTENT_SIZE + 36):
    """已知 4 個角點後，算出透視變換矩陣，把斜的文件校正回正視的正方形。"""
    dst_pts = np.array([
        [0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1],
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corners, dst_pts)
    return cv2.warpPerspective(photo_bgr, H, (out_size, out_size))


def evaluate(corrected: np.ndarray, reference_doc: np.ndarray) -> float:
    ref = cv2.resize(reference_doc, (corrected.shape[1], corrected.shape[0]))
    g1 = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    return float(ssim(g1, g2))


def run_basic_demo():
    content = cv2.imread("images/building.jpg")
    doc = make_document(content)
    theta = 38
    photo, true_corners = synthesize_oblique_photo(doc, theta)

    detected = detect_document_corners(photo)
    vis = photo.copy()
    if detected is not None:
        for (x, y) in detected:
            cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.polylines(vis, [detected.astype(np.int32)], True, (0, 255, 0), 2)
        corrected = correct_perspective(photo, detected)
        score = evaluate(corrected, doc)
    else:
        corrected = np.zeros_like(doc)
        score = 0.0

    cv2.imwrite(f"{OUT_DIR}/basic_reference_document.png", doc)
    cv2.imwrite(f"{OUT_DIR}/basic_oblique_photo.png", photo)
    cv2.imwrite(f"{OUT_DIR}/basic_detected_corners.png", vis)
    cv2.imwrite(f"{OUT_DIR}/basic_corrected.png", corrected)

    print(f"[基礎題] 斜拍角度 {theta} 度 -> 校正後與原始文件 SSIM = {score:.4f}")
    return theta, score


def run_advanced_sweep():
    """對多張不同內容的影像 x 多個角度，全自動跑一次校正 pipeline。"""
    sources = {
        "building": "images/building.jpg",
        "home": "images/home.jpg",
        "fruits": "images/fruits.jpg",
    }
    angles = list(range(0, 81, 5))

    results = {name: [] for name in sources}
    fail_angle = {}

    for name, path in sources.items():
        content = cv2.imread(path)
        doc = make_document(content)
        for theta in angles:
            photo, _ = synthesize_oblique_photo(doc, theta)
            detected = detect_document_corners(photo)
            if detected is None:
                results[name].append(0.0)
                fail_angle.setdefault(name, theta)
                continue
            corrected = correct_perspective(photo, detected)
            score = evaluate(corrected, doc)
            # SSIM 太低代表雖然偵測到「一個四邊形」，但角點對應錯誤/形變太嚴重，視同失敗
            if score < 0.35:
                fail_angle.setdefault(name, theta)
            results[name].append(score)

    # 畫圖: SSIM vs 角度，每張圖一條線
    plt.figure(figsize=(8, 5))
    for name, scores in results.items():
        plt.plot(angles, scores, marker="o", label=name)
    plt.axhline(0.35, color="gray", linestyle="--", linewidth=1, label="failure threshold (SSIM=0.35)")
    plt.xlabel("Simulated oblique angle theta (deg)")
    plt.ylabel("SSIM (corrected vs. original document)")
    plt.title("Automatic perspective correction quality vs. shooting angle")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/advanced_ssim_vs_angle.png", dpi=120)
    plt.close()

    # 存一份範例：building 在幾個代表性角度下的校正結果，方便肉眼比較
    content = cv2.imread(sources["building"])
    doc = make_document(content)
    sample_thumbs = []
    for theta in [10, 40, 60, 75]:
        photo, _ = synthesize_oblique_photo(doc, theta)
        detected = detect_document_corners(photo)
        if detected is not None:
            corrected = correct_perspective(photo, detected)
        else:
            corrected = np.zeros((CONTENT_SIZE + 36, CONTENT_SIZE + 36, 3), dtype=np.uint8)
        photo_small = cv2.resize(photo, (corrected.shape[1], corrected.shape[0]))
        sample_thumbs.append(np.vstack([photo_small, corrected]))
    grid = np.hstack(sample_thumbs)
    cv2.imwrite(f"{OUT_DIR}/advanced_angle_grid_building.png", grid)

    summary = {
        "angles": angles,
        "ssim_by_source": results,
        "first_failure_angle": fail_angle,
    }
    with open(f"{OUT_DIR}/advanced_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[進階題] 各影像第一次校正失效的角度 (SSIM < 0.35):")
    for name, ang in fail_angle.items():
        print(f"  {name}: {ang} 度")
    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_basic_demo()
    run_advanced_sweep()


if __name__ == "__main__":
    main()
