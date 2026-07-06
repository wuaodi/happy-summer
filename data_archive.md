# 数据归档说明

数据集不进 git，统一打包放百度网盘。下载后在**仓库根目录**直接解压即可还原目录结构：

```bash
unzip data_Aqua_20260706.zip     # 解压出 data/Aqua/
```

## 网盘位置

百度网盘 `Alvin/git大文件/happy-summer_archive/`

## 内容清单

| zip 名称 | 内容 | 大小 |
|---|---|---|
| data_Aqua_20260706.zip | `data/Aqua/`：images/ 100 张 1024×1024 RGBA PNG + transforms_train.json（87 帧）+ transforms_test.json（13 帧） | 7.8M |

数据格式说明见 README「数据说明」。注意首次训练后 `data/Aqua/` 下会多出一个 `points3d.ply`，是 2DGS 自动生成的随机初始化点云，不属于数据本体，删掉会自动重建。

## 打包记录

- 2026-07-06：`data_Aqua_20260706.zip`，覆盖 `data/Aqua/` 全部内容（剔除自动生成的 points3d.ply），store 模式打包。
