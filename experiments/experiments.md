# 实验记录

最新在前。每条格式：日期 + 做了什么 + 结论/数字 + 遗留问题。结果图片直接放本文件夹，用相对路径引用。

---

## 2026-07-06 仓库初始化，Aqua 跑通验证

- 环境：RTX 4090 24G，conda 环境 `2dgs`（Python 3.10 + torch 2.1.0 cu118），按 README 步骤配置，CUDA 扩展编译通过。
- 数据：`data/Aqua/`，100 帧 1024×1024（87 训练 / 13 测试）。
- 命令：`python train.py -s ../data/Aqua -m output/Aqua --eval`，30k 迭代。
- 结果：（跑完回填：PSNR / SSIM / LPIPS / 训练用时）
