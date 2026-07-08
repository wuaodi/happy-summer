# 实习任务清单

总目标：掌握 2DGS 新视角合成全流程，先在仿真渲染的卫星数据上跑通，再设计并完成实验室真机数据采集，用真实数据训练并定量评估。所有代码提交到本仓库，数据不入库（放百度网盘），实验记录和结果图片放 `experiments/`（写在 `experiments/experiments.md`，最新在前）。

## 阶段 1：数据与位姿

- [ √ ] 下载 Aqua 仿真数据放到 `data/Aqua/`（百度网盘链接：待填）
- [ √ ] 读懂数据格式：`transforms_train.json` 里每帧的 `transform_matrix` 是 c2w（OpenGL 约定，相机看 -Z、up +Y），`camera_angle_x` 换算焦距，alpha 通道是前景 mask，具体见 README「数据说明」
- [ √ ] 写一个脚本可视化 100 帧相机位姿真值：画出相机在球面上的位置分布和朝向（matplotlib 3D 或 open3d 都行），确认相机都朝向原点、覆盖大半个球面。脚本放 `scripts/` 并提交

## 阶段 2：跑通 2DGS

硬件结论（不用再验证）：必须 Ubuntu + NVIDIA 显卡。2DGS 的光栅化器 diff-surfel-rasterization 是纯 CUDA 算子，没有 CPU 实现，CPU 上训练和渲染都跑不了，不存在「用 CPU 慢慢跑」的选项。显存 8 GB 以上即可。

- [ ] 按 README「安装步骤」配好 `2dgs` 环境，编译通过两个 CUDA 扩展
- [ ] 用 Aqua 数据训练 30k 迭代并评估，指标应接近参考值（见下方「Aqua 参考结果」，明显偏低说明哪里不对）
- [ ] 2DGS 算法上手：改动 `--lambda_normal`、`--depth_ratio`、`-r` 等参数各跑一两组，观察渲染质量和 mesh 质量的变化，记录结论
- [ ] 2DGS 算法理解：读论文 [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://arxiv.org/abs/2403.17888)，重点弄清三件事——为什么用 2D surfel 替代 3D 椭球、透视正确的光栅化怎么做、深度畸变与法线一致性两个正则各解决什么问题。写一页笔记或做一次组会分享

## 阶段 3：实验室真机采集与验证

- [ ] 动捕系统上手：给一个小球贴 marker，实时读出它的位置，感受精度、丢帧、坐标系定义
- [ ] 设计采集方案并过一遍评审，方案要回答：
  - 相机和卫星模型各自如何装 marker、动捕如何同时输出两者的 6DoF
  - 相机内参怎么标（棋盘格/标定板）
  - 动捕刚体坐标系到相机光心坐标系的外参怎么标（手眼标定，这是方案的核心难点）
  - 图像帧和动捕位姿怎么做时间同步
  - 如何把「相机位姿 + 卫星位姿」换算成卫星体坐标系下的相机 c2w
- [ ] 实际采集：绕卫星模型多视角拍摄（参考 Aqua 的球面覆盖），光照保持固定，背景尽量暗且干净
- [ ] 把采集数据转换成与 `data/Aqua/` 完全相同的格式：RGBA PNG（alpha 放前景 mask，暗背景阈值抠或用 SAM）+ `transforms_train.json` / `transforms_test.json`。格式对齐后训练代码一行不用改，后续也能直接用于实验室的其他仓库
- [ ] 用采集的数据训练 2DGS
- [ ] 评估新视角合成精度：留出测试视角算 PSNR / SSIM / LPIPS，与 Aqua 仿真结果对比，分析差距来源（位姿误差、mask 质量、光照变化、运动模糊等），写成小结

## Aqua 参考结果

本机 RTX 4090、默认参数 30k 迭代、测试集 13 帧，`metrics.py` 输出：

| PSNR | SSIM | LPIPS | 训练用时 |
|---|---|---|---|
| 31.53 | 0.984 | 0.022 | 约 8 分钟 |

一个值得思考的现象：7k 迭代时测试 PSNR 有 35.9，30k 反而降到 31.7。这不是 bug——2DGS 在训练中后段开启了深度畸变和法线一致性正则，用一部分光度精度换几何质量（mesh 更干净）。读论文时可以结合这个现象理解两个正则的作用。

## 工作方式

- 本仓库为公开仓库，实习生以协作者身份开发（接受邮件邀请后即有 push 权限）。
- 分支模型：main 是稳定分支，受保护，不直接 push。实习生在 `dev_wgb` 分支上开发，每完成一个阶段性功能就从 `dev_wgb` 向 main 发 Pull Request，review 通过后合入；合并后记得把 main 同步回 `dev_wgb`（`git merge main`）再继续。
- PR 描述里写清楚做了什么、为什么这么设计、怎么验证；review 意见在 PR 页面里讨论并留痕，改完再合。
- commit 信息写清楚做了什么；实验记录及时写进 `experiments/experiments.md`。
