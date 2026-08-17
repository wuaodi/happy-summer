"""
相机内参标定（棋盘格法）—— ZED 2 适配

用法 1 — 从已有照片标定（推荐，ZED 2 必须用这个方式）：
    python scripts/calibrate_intrinsics.py \
      --image_dir path/to/calib_photos/ --cols 12 --rows 9 --square_size 20

用法 2 — 普通 USB 相机实时采集（ZED 2 不支持，见下方说明）：
    python scripts/calibrate_intrinsics.py \
      --capture --cols 12 --rows 9 --square_size 20

ZED 2 注意事项：
    ZED 2 不是标准 UVC 摄像头 —— 它的左右两个镜头通过 ZED SDK 才能同时访问，
    OpenCV 的 VideoCapture 拿不到单个镜头的画面。所以 --capture 模式对 ZED 2 无效。
    正确做法：用 ZED 自带的工具（ZED Explorer / ZEDfu）或自己写几行 pyzed 脚本，
    拍左镜头的 20+ 张棋盘格照片放进一个文件夹，然后用 --image_dir 模式标定。

    ZED 2 左镜头参考规格（供标定后对比）：
      - 分辨率：1920×1080（HD1080）或 1280×720（HD720）
      - 标称视场角：~90°(H) × 60°(V)
      - 标称焦距：fx ≈ 1050（1080p 下），cx ≈ 960，cy ≈ 540

输出：
    标定结果保存为 calib_K_dist.json（内参矩阵 K + 畸变系数 dist + 图像尺寸）

棋盘格：
    在线生成：https://calib.io/pages/camera-calibration-pattern-generator
    参数：12×9 内角点、格子边长 20mm
    打印后贴在一块平整硬板上（不能弯），格子边长用尺子再量一次确认
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Camera intrinsic calibration (checkerboard)")
    p.add_argument("--image_dir", default="",
                   help="Existing calibration photos folder. If omitted, use --capture.")
    p.add_argument("--capture", action="store_true",
                   help="Capture photos live from camera (NOT supported for ZED 2).")
    p.add_argument("--camera_id", type=int, default=0,
                   help="Camera device ID for live capture (default 0).")
    p.add_argument("--cols", type=int, default=12,
                   help="Number of INNER corners horizontally (default 12).")
    p.add_argument("--rows", type=int, default=9,
                   help="Number of INNER corners vertically (default 9).")
    p.add_argument("--square_size", type=float, default=20.0,
                   help="Side length of one checkerboard square in mm (default 20).")
    p.add_argument("--out", default="calib_K_dist.json",
                   help="Output file path (default calib_K_dist.json).")
    p.add_argument("--ext", default=".png",
                   help="Image file extension to look for in --image_dir (default .png).")
    return p.parse_args()


# ── 棋盘格检测 ────────────────────────────────────────────────

def detect_corners(img_gray, cols, rows):
    """
    在灰度图中检测棋盘格内角点。
    返回 (ret, corners)，ret=True 表示检测成功。
    """
    # 尝试多种检测标志：先普通，再自适应阈值，再归一化
    flags_list = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        0,
    ]
    for flags in flags_list:
        ret, corners = cv2.findChessboardCorners(img_gray, (cols, rows), flags=flags)
        if ret:
            # 亚像素精化：把角点坐标从整数像素提升到小数点精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners
    return False, None


# ── 采集模式：实时拍照 ─────────────────────────────────────────

def capture_photos(cols, rows, camera_id):
    """
    打开摄像头，按 SPACE 拍照，按 ESC 退出。
    返回拍照的图像列表（灰度图 + 原图）。
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        return []

    print("=" * 50)
    print("  CAMERA CAPTURE MODE")
    print("=" * 50)
    print(f"  SPACE = take photo    ESC = finish")
    print(f"  Checkerboard: {cols}×{rows} inner corners\n")

    captured = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # 实时检测棋盘格角点
        found, corners = detect_corners(gray, cols, rows)
        if found:
            cv2.drawChessboardCorners(display, (cols, rows), corners, found)
            cv2.putText(display, "DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, "NOT DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(display, f"Captured: {len(captured)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Calibration Capture [SPACE=take  ESC=done]", display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            captured.append((frame, gray))
            frame_count += 1
            print(f"  [{frame_count}] photo taken")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Total: {len(captured)} photos captured.\n")
    return captured


# ── 主标定逻辑 ────────────────────────────────────────────────

def calibrate(cols, rows, square_size, img_size, obj_points_list, img_points_list):
    """
    调用 OpenCV calibrateCamera。
    返回 (ret, K, dist, rvecs, tvecs, mean_error)
    """
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points_list, img_points_list, img_size,
        cameraMatrix=None, distCoeffs=None,
    )

    # 计算平均重投影误差
    total_error = 0
    total_points = 0
    for i in range(len(obj_points_list)):
        projected, _ = cv2.projectPoints(obj_points_list[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(img_points_list[i], projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(obj_points_list[i])
    mean_error = np.sqrt(total_error / total_points)

    return ret, K, dist, rvecs, tvecs, mean_error


def main():
    args = parse_args()

    # 构建 3D 目标点（棋盘格角点在其实物坐标系中的坐标，Z=0 平面）
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    obj_points_list = []   # 每张图的 3D 目标点（都相同）
    img_points_list = []   # 每张图的 2D 角点（检测结果）

    # ── 获取图像 ──
    if args.capture:
        frames = capture_photos(args.cols, args.rows, args.camera_id)
        if not frames:
            return
        grays = [g for _, g in frames]
    elif args.image_dir:
        folder = Path(args.image_dir)
        if not folder.is_dir():
            print(f"ERROR: Directory not found: {folder}")
            return
        paths = sorted(folder.glob(f"*{args.ext}")) + sorted(folder.glob(f"*{args.ext.upper()}"))
        if not paths:
            print(f"ERROR: No {args.ext} images found in {folder}")
            return
        print(f"Found {len(paths)} images in {folder}")
        grays = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                print(f"  SKIP: cannot read {p.name}")
                continue
            grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    else:
        print("ERROR: Specify --image_dir or --capture.")
        return

    # ── 逐张检测角点 ──
    print("\nDetecting checkerboard corners...")
    found_count = 0
    for i, gray in enumerate(grays):
        ret, corners = detect_corners(gray, args.cols, args.rows)
        if ret:
            obj_points_list.append(objp)
            img_points_list.append(corners)
            found_count += 1
            print(f"  [{found_count}] OK  (image {i+1})")
        else:
            print(f"  [--] FAIL (image {i+1})")

    print(f"\nDetection success: {found_count}/{len(grays)}")

    if found_count < 15:
        print(f"WARNING: Only {found_count} valid views. "
              f"Recommend ≥ 20 for reliable calibration. Take more photos "
              f"with varied angles and distances.")

    if found_count < 5:
        print("ERROR: Too few valid views. Cannot calibrate.")
        return

    # ── 标定 ──
    h, w = grays[0].shape[:2]
    print(f"\nCalibrating with {found_count} views, image size {w}×{h} ...")
    ret, K, dist, rvecs, tvecs, mean_error = calibrate(
        args.cols, args.rows, args.square_size, (w, h),
        obj_points_list, img_points_list,
    )

    # ── 输出结果 ──
    print("\n" + "=" * 50)
    print("  CALIBRATION RESULTS")
    print("=" * 50)
    print(f"\n  Camera matrix K:")
    print(f"    fx={K[0,0]:.2f}  fy={K[1,1]:.2f}")
    print(f"    cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"    [[{K[0,0]:.3f}, 0.000, {K[0,2]:.3f}]")
    print(f"     [0.000, {K[1,1]:.3f}, {K[1,2]:.3f}]")
    print(f"     [0.000, 0.000, 1.000]]")
    print(f"\n  Distortion coefficients (k1,k2,p1,p2,k3):")
    print(f"    {dist.ravel()}")
    print(f"\n  Mean reprojection error: {mean_error:.4f} px")
    print(f"  RMS: {ret:.4f}")

    # 质量判断
    if mean_error < 0.3:
        print("  ✓ Excellent (< 0.3 px)")
    elif mean_error < 0.5:
        print("  ✓ Good (< 0.5 px)")
    elif mean_error < 1.0:
        print("  △ Acceptable (< 1.0 px), consider retaking some views")
    else:
        print("  ✗ Poor (> 1.0 px). Possible causes:")
        print("      - Checkerboard not flat (warped paper)")
        print("      - Square size entered incorrectly")
        print("      - Too few views or all from similar angles")
        print("      - Motion blur — use faster shutter")

    # 水平视场角（用于后续写 transforms JSON）
    camera_angle_x = 2 * np.arctan(w / (2 * K[0, 0]))
    print(f"\n  Horizontal FOV (camera_angle_x): {camera_angle_x:.6f} rad = "
          f"{np.degrees(camera_angle_x):.2f}°")

    # ZED 2 参考值对比（1080p 下，帮助判断标定是否离谱）
    if abs(w - 1920) <= 10:
        print(f"\n  ZED 2 参考值 (1080p): fx≈1050, fy≈1050, cx≈960, cy≈540, FOV≈90°")
        if abs(K[0, 0] - 1050) > 300:
            print("  ⚠ fx 偏离 ZED 2 标称值较远，请检查棋盘格尺寸或照片质量")
        if abs(K[0, 2] - 960) > 300:
            print("  ⚠ cx 偏离 ZED 2 标称值较远，请检查棋盘格尺寸或照片质量")

    # ── 保存 ──
    out_path = Path(args.out)
    result = {
        "K": K.tolist(),
        "dist": dist.tolist(),
        "image_width": w,
        "image_height": h,
        "camera_angle_x_rad": float(camera_angle_x),
        "mean_reprojection_error_px": float(mean_error),
        "checkerboard": f"{args.cols}x{args.rows}, square={args.square_size}mm",
        "num_views": found_count,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  Saved to: {out_path.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
