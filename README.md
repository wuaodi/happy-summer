# happy-summer 卫星新视角合成实习项目

用 2DGS（2D Gaussian Splatting）在带位姿真值的卫星图像上做新视角合成。先用仿真渲染数据跑通全流程，再到实验室真机采集数据并验证。实习任务清单见 [goal_todo.md](goal_todo.md)。

## 仓库结构

```
happy-summer/
├── 2d-gaussian-splatting/   # 官方 2DGS 代码（hbb1/2d-gaussian-splatting @ 335ad61）
│   └── submodules/          # CUDA 扩展源码已内置（diff-surfel-rasterization、simple-knn），克隆无需 --recursive
├── data/                    # 数据集（不入库，从百度网盘下载后放到这里）
│   └── Aqua/                # Aqua 卫星仿真数据（100 帧，Blender Cycles 渲染）
├── experiments/             # 实验记录（experiments.md，最新在前）与结果图片
├── goal_todo.md             # 实习任务清单
└── README.md
```

## 数据说明

`data/Aqua/` 是 NASA Aqua 卫星的绕飞仿真数据，Blender Cycles 路径追踪渲染，相机位姿是精确真值。

百度网盘链接：（待填）

```
data/Aqua/
├── images/                  # 100 张 1024×1024 RGBA PNG，alpha 通道 = 前景 mask（背景为太空，纯黑）
├── transforms_train.json    # 训练集 87 帧：camera_angle_x + 每帧 c2w 位姿
└── transforms_test.json     # 测试集 13 帧（每 8 帧留 1 帧）
```

格式为标准 NeRF-synthetic（Blender）格式，要点：

- `transform_matrix` 是 4×4 的 **c2w**（相机到世界），OpenGL/Blender 相机约定：相机朝 **-Z** 看，**+Y** 朝上。第 4 列前 3 行是相机在世界系中的位置。
- `camera_angle_x` 是水平视场角（弧度），焦距换算：`fx = 0.5 * W / tan(0.5 * camera_angle_x)`，主点在图像中心。
- `file_path` 是不带扩展名的相对路径，读图时补 `.png`。

后续实验室真机采集的数据也统一转成这个格式（见 goal_todo.md 阶段 3），这样训练代码一行不用改。

## 环境需求

- Ubuntu（实测 22.04），**必须有 NVIDIA 显卡**。2DGS 的光栅化器是纯 CUDA 算子（diff-surfel-rasterization），没有 CPU 实现，CPU 上无法训练和渲染。你可以现在你的电脑上尝试验证，到所里实验室会给你准备一台装好ubuntu的RTX4090显卡的电脑
- 显存：Aqua 数据 1024×1024 训练实测峰值约 4~6 GB，8 GB 以上显卡即可（RTX 4090 24 GB）。
- 显卡驱动 ≥ 520（能跑 CUDA 11.8 即可，`nvidia-smi` 能看到显卡就行，不需要系统装 CUDA，CUDA 工具链装在 conda 环境里）。
- gcc/g++ 11：CUDA 11.8 的 nvcc 不支持 gcc 12 及以上，Ubuntu 22.04 默认是 gcc 12，需要装 gcc-11：

```bash
sudo apt install gcc-11 g++-11
```

- conda（Anaconda 或 Miniconda 均可）。

## 安装步骤

```bash
# 1. 克隆仓库（子模块源码已内置，不需要 --recursive）
git clone https://github.com/wuaodi/happy-summer.git
cd happy-summer

# 2. 创建环境：Python 3.10 + PyTorch 2.1 (cu118)
conda create -n 2dgs python=3.10 -y
conda activate 2dgs
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# 3. 把 CUDA 11.8 工具链（nvcc 等）装进 conda 环境，避免依赖系统 CUDA
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y

# 4. 其余 Python 依赖（版本约束都是踩过坑的：torch 2.1 不兼容 numpy 2.x，
#    编扩展需要 pkg_resources 所以 setuptools<70，plyfile 新版本强制 numpy>=2 故降级，
#    pillow 11.x 会让 dataset_readers 里的 Image.fromarray 报错所以钉 10.2.0）
pip install "numpy<2" "setuptools<70" "plyfile<1.1" "pillow==10.2.0" opencv-python open3d lpips scikit-image tqdm trimesh mediapy

# 5. 编译两个 CUDA 扩展（关键一步）
cd 2d-gaussian-splatting
export CUDA_HOME=$CONDA_PREFIX          # 让编译用环境里的 nvcc 11.8，而不是系统 CUDA
export PATH=$CUDA_HOME/bin:$PATH
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
# --no-build-isolation 必须加：不加的话 pip 会开一个没有 torch 的临时环境去编译，报 No module named 'torch'
pip install --no-build-isolation ./submodules/diff-surfel-rasterization ./submodules/simple-knn
```

验证安装：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出: 2.1.0+cu118 True
python -c "import diff_surfel_rasterization, simple_knn; print('CUDA 扩展 OK')"
```

### 常见坑

| 现象 | 原因与解决 |
|---|---|
| 编译报 `No module named 'torch'` | 漏了 `--no-build-isolation` |
| 编译报 `No module named 'pkg_resources'` | setuptools 版本太新，`pip install "setuptools<70"` |
| 编译报 `unsupported GNU version! gcc versions later than 11 are not supported` | 没生效 gcc-11，确认 `export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11` 后重装 |
| 编译报 `nvcc fatal: Unsupported gpu architecture 'compute_89'` | 用到了系统里的老版本 nvcc（如 11.5），确认 `CUDA_HOME=$CONDA_PREFIX` 且 `which nvcc` 指向 conda 环境 |
| 读数据报 `TypeError: Cannot handle this data type: (1, 1, 3), \|i1` | Pillow 版本太新，`pip install "pillow==10.2.0"` |
| 训练时 `CUDA out of memory` | 加 `-r 2` 把图像降到一半分辨率训练 |
| `ModuleNotFoundError: diff_surfel_rasterization` | 第 5 步没装成功，回去看编译日志 |

## 训练与评估

以下命令都在 `2d-gaussian-splatting/` 目录下、`2dgs` 环境里执行。

```bash
# 训练（约 30k 次迭代，4090 上约半小时；--eval 表示留出测试集）
python train.py -s ../data/Aqua -m output/Aqua --eval

# 渲染训练/测试视角 + 提取 mesh
python render.py -m output/Aqua -s ../data/Aqua

# 计算测试集新视角合成指标 PSNR / SSIM / LPIPS
python metrics.py -m output/Aqua
```

产物在 `output/Aqua/`：`point_cloud/` 是训好的高斯模型，`test/ours_30000/renders/` 是测试视角渲染图（可与 `gt/` 对比），`train/ours_30000/fuse_post.ply` 是提取的 mesh，`results.json` 是指标。

本机参考指标（Aqua，30k 迭代，测试集 13 帧）：见 goal_todo.md 阶段 2，跑完应与之接近。

常用训练参数：`--lambda_normal`（法线一致性正则）、`--lambda_dist`（深度畸变正则）、`--depth_ratio`（0 均值深度 / 1 中值深度）、`-r 2`（半分辨率）。

## 上游代码说明

`2d-gaussian-splatting/` 来自官方仓库 [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)（commit 335ad61），子模块 [diff-surfel-rasterization](https://github.com/hbb1/diff-surfel-rasterization) 与 [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) 的源码直接放在 `submodules/` 下（glm 只保留了编译需要的头文件）。除此之外没有改动，算法原理与用法看该目录下的 `README.md` 和论文 [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://arxiv.org/abs/2403.17888)。
