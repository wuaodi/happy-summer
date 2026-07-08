"""
可视化 Aqua 数据集的 100 帧相机位姿。

画出相机在 3D 空间中的位置分布和朝向，确认：
- 相机都围绕原点分布（卫星在原点附近）
- 相机朝向都指向原点
- 覆盖大半个球面（多视角绕飞）

用法：
    python scripts/visualize_poses.py

输出：
    交互式 matplotlib 3D 窗口，可旋转/缩放观察。
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_poses(data_dir: str):
    """读取 train + test 两个 JSON 的 c2w 矩阵，返回 (positions, forwards, ups, labels)。"""
    data_dir = Path(data_dir)
    all_positions = []
    all_forwards = []
    all_ups = []
    all_labels = []  # 'train' or 'test'

    for split, filename in [("train", "transforms_train.json"),
                             ("test", "transforms_test.json")]:
        with open(data_dir / filename) as f:
            data = json.load(f)

        for frame in data["frames"]:
            c2w = np.array(frame["transform_matrix"])  # 4×4, OpenGL convention

            # 相机在世界系中的位置 = 第 4 列前 3 行
            pos = c2w[:3, 3]

            # OpenGL/Blender 相机约定：相机朝 -Z 看，+Y 朝上
            # 局部 Z 轴在世界系中的方向 = c2w[:3, 2]，相机前向 = -Z
            forward = -c2w[:3, 2]  # 镜头朝向
            up = c2w[:3, 1]        # 相机上方

            all_positions.append(pos)
            all_forwards.append(forward)
            all_ups.append(up)
            all_labels.append(split)

    return (np.array(all_positions), np.array(all_forwards),
            np.array(all_ups), all_labels)


def plot_poses(positions, forwards, ups, labels):
    """绘制相机位姿 3D 图：位置 + 朝向箭头 + 上方短箭头。"""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    train_mask = np.array([l == "train" for l in labels])
    test_mask = ~train_mask

    # --- 相机位置：用散点画在球面上 ---
    for mask, color, name, marker in [
        (train_mask, "#2196F3", "Train (87 frames)", "o"),
        (test_mask, "#FF5722", "Test (13 frames)", "s"),
    ]:
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            positions[mask, 2],
            c=color, marker=marker, s=50, alpha=0.85,
            label=name, edgecolors="white", linewidth=0.5,
            depthshade=True,
        )

    # --- 相机朝向：用 quiver 画箭头 ---
    arrow_len = 8  # 朝向箭头长度
    up_arrow_len = 3.0  # 上方箭头长度

    for i in range(len(positions)):
        is_train = labels[i] == "train"
        color = "#2196F3" if is_train else "#FF5722"
        alpha = 0.35 if is_train else 0.75  # test 更突出

        p = positions[i]
        f = forwards[i]
        u = ups[i]

        # 主朝向箭头（相机看的方向）
        ax.quiver(
            p[0], p[1], p[2],
            f[0], f[1], f[2],
            length=arrow_len, color=color, alpha=alpha,
            linewidth=1.2, arrow_length_ratio=0.15,
            normalize=True,
        )

        # 上方短箭头（相机顶部方向）
        ax.quiver(
            p[0], p[1], p[2],
            u[0], u[1], u[2],
            length=up_arrow_len, color="#4CAF50", alpha=alpha * 0.7,
            linewidth=0.8, arrow_length_ratio=0.2,
            normalize=True,
        )

    # --- 原点标记（卫星所在位置） ---
    ax.scatter([0], [0], [0], c="#E91E63", marker="*", s=300,
               label="Origin (satellite)", edgecolors="white", linewidth=0.8,
               depthshade=False)

    # --- 坐标轴 ---
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        "Aqua Camera Poses — 100 Frames\n"
        "Blue/Orange arrows = look direction | Green arrows = camera up | ★ = origin",
        fontsize=13, pad=20,
    )

    # 等比例轴，避免球面变形
    _set_axes_equal(ax)

    # 图例：手动添加朝向和上方箭头说明
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=10, label="Train (87 frames)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF5722",
               markersize=10, label="Test (13 frames)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#E91E63",
               markersize=14, label="Origin (satellite)"),
        Line2D([0], [0], color="#2196F3", linewidth=2, label="Look direction"),
        Line2D([0], [0], color="#4CAF50", linewidth=2, label="Camera up"),
    ]
    ax.legend(handles=custom_lines, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.show()


def _set_axes_equal(ax):
    """让 3D 图三个轴等比例缩放，球面不变形。"""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    mid = np.mean(limits, axis=1)
    span = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([mid[0] - span, mid[0] + span])
    ax.set_ylim3d([mid[1] - span, mid[1] + span])
    ax.set_zlim3d([mid[2] - span, mid[2] + span])


def print_statistics(positions, forwards, labels):
    """打印位姿的统计信息，帮助确认数据合理性。"""
    train_mask = np.array([l == "train" for l in labels])
    test_mask = ~train_mask

    # 相机到原点的距离
    dists = np.linalg.norm(positions, axis=1)

    # 相机朝向与"指向原点方向"的夹角
    dirs_to_origin = -positions  # 从相机指向原点
    dirs_to_origin = dirs_to_origin / np.linalg.norm(dirs_to_origin, axis=1, keepdims=True)
    cos_angles = np.sum(forwards * dirs_to_origin, axis=1)
    angles_deg = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))

    print("=" * 60)
    print("Camera Pose Statistics")
    print("=" * 60)
    print(f"Total frames: {len(positions)} (train {train_mask.sum()}, "
          f"test {test_mask.sum()})")
    print()
    print(f"Distance to origin (camera radius):")
    print(f"  Mean:   {dists.mean():.2f}")
    print(f"  Range:  [{dists.min():.2f}, {dists.max():.2f}]")
    print()
    print(f"Angle between look-direction and origin-direction "
          f"(0° = perfectly pointing at origin):")
    print(f"  Mean:   {angles_deg.mean():.2f}°")
    print(f"  Max:    {angles_deg.max():.2f}°")
    print(f"  Min:    {angles_deg.min():.2f}°")
    print()

    # 分布判断
    if angles_deg.max() < 15:
        print("✓ All cameras point near the origin — looks correct.")
    else:
        print("⚠ Some cameras deviate notably from the origin — check the data.")

    # 检查球面覆盖：计算相机在球面上的角分布
    # 把位置投影到单位球面，看经纬度范围
    sph_pos = positions / dists[:, None]
    # 球坐标：theta = azimuth (XZ), phi = elevation (Y)
    theta = np.degrees(np.arctan2(sph_pos[:, 0], sph_pos[:, 2]))  # -180..180
    phi = np.degrees(np.arcsin(sph_pos[:, 1]))                     # -90..90

    print(f"Spherical coverage of camera positions:")
    print(f"  Azimuth (XZ):  [{theta.min():.0f}°, {theta.max():.0f}°]")
    print(f"  Elevation (Y): [{phi.min():.0f}°, {phi.max():.0f}°]")
    print()

    if theta.max() - theta.min() > 180:
        print("✓ Azimuth coverage > 180° — good spherical coverage.")
    else:
        print("⚠ Azimuth coverage < 180° — may not cover enough views.")

    if phi.max() - phi.min() > 60:
        print("✓ Elevation coverage > 60° — decent vertical spread.")
    else:
        print("⚠ Elevation coverage narrow — cameras mostly at same height.")

    print("=" * 60)


def main():
    # 脚本位置: happy-summer/scripts/visualize_poses.py
    # 数据位置: happy-summer/data/Aqua/
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_dir = repo_root / "data" / "Aqua"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please download Aqua data to data/Aqua/ first.")
        return

    positions, forwards, ups, labels = load_poses(str(data_dir))

    print_statistics(positions, forwards, labels)
    plot_poses(positions, forwards, ups, labels)


if __name__ == "__main__":
    main()
