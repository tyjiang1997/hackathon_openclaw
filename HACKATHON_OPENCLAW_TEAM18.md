# OpenClaw 赛道 · x4claw（第 18 组）— 项目综述

## 团队成员

- Peng Yun
- Qing Lian
- Tianyuan Jiang
- Jinghang Li
- Feiyi chen

## 概述

本项目面向黑客松 **OpenClaw** 赛道，构建 **双层机器人系统**：**上层** 基于 **Nanobot** 负责任务分解与分发；**底层** 基于 **OpenPI 系视觉-语言-动作（VLA）模型 π₀.₅（Pi0.5）**，在真实桌面上完成抓取、归类与丢弃等原子操作。展示场景为 **桌面清理**：日常用品（如记号笔）归入储物区域，纸团等垃圾投入垃圾桶。

本仓库以 [LeRobot](https://github.com/huggingface/lerobot) 为训练与推理主干，策略实现与 [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi) 对齐，便于在统一数据格式下微调与部署。

---

## 系统架构

| 层级 | 组件 | 职责 |
|------|------|------|
| 上层 | Nanobot | 高层任务规划、子任务序列化与调度（与底层策略通过接口衔接） |
| 底层 | Pi0.5（OpenPI 兼容） | 单步/片段级视觉运动控制：抓取、放置、清理桌面等 |

数据流可概括为：**自然语言 / 结构化子任务 → 观测（图像 + 机器人状态）→ Pi0.5 动作序列 → 机械臂执行**。

---

## 展示任务：桌面清理

- **可回收 / 可归类物品**（如记号笔、纸盒等）：放入指定 **储物框 / 支架**。
- **垃圾**（如纸团、易拉罐等）：投入 **垃圾桶**。
- **复合目标**：在「整桌清理」类任务中，模型需根据语言指令与视觉上下文连续完成多步操作。

当前训练数据清单中的任务命名与上述语义一致（示例见下文「数据与配置」）。

---

## 数据采集

- **方式**：**遥操作（Teleoperation）** 采集演示数据，格式为 LeRobot 数据集（Parquet + 视频等）。
- **规模**：**7 类原子任务 × 30 组** Episode（赛题口径；具体目录以各环境 `trainset_config` 中路径为准）。
- **聚合训练**：多数据集根目录通过 **`dataset_list_file`** 一次性喂给 `lerobot-train`，便于混合多任务微调。

---

## 模型训练（LeRobot + 云端分布式）

- **策略类型**：`policy.type=pi05`，对应仓库内 `PI05Policy` / `PI05Config`（`src/lerobot/policies/pi05/`）。
- **初始化**：默认自 **`lerobot/pi05_base`** 加载预训练权重，在自有数据上做 **全参数微调**。
- **训练入口**：`lerobot.scripts.lerobot_train`（CLI 即 `lerobot-train`）。
- **分布式**：脚本 `train_pi05_xle.sh` 使用 **`accelerate launch`**，并读取云端训练平台注入的多机环境变量（如 `MLP_WORKER_NUM`、`MLP_ROLE_INDEX`、`MLP_WORKER_0_HOST` 等）做 **NCCL + PyTorch 分布式**，支持单机多卡与多节点。
- **典型超参**（可在脚本中调整）：
  - 每 GPU `batch_size`、总 `steps`、`save_freq`
  - `policy.dtype=bfloat16`，可选 `gradient_checkpointing` 节省显存
  - 优化器与学习率调度与 OpenPI 侧设计对齐（见 `configuration_pi05.py` 中 `optimizer_*`、`scheduler_*`）

**与本项目直接相关的仓库文件：**

- `train_pi05_xle.sh` — xlerobot 多数据集微调一键脚本（含云端多节点适配）。
- `trainset_config/xlerobot.txt` — 数据集根路径列表（每行一个 LeRobot 数据集目录）。
- `train_pi05_auto.sh` — 通用本地 Pi0.5 微调脚本（可参考对比）。

**数据清单示例**（`trainset_config/xlerobot.txt` 中任务 ID 与语义示例）：

- `task00` — 纸团入垃圾桶  
- `task01` — 罐类入垃圾桶  
- `task04` — 纸盒等放入支架/收纳位  
- `task07` — 桌面整体清理（多段轨迹）

---

## 模型评估与离线验证

- **脚本**：`examples/training/infer_pi05_trainset.py`  
  从检查点目录读取 `train_config.json` 与 `model.safetensors`，重建与训练一致的数据管道，在 **eval 模式**下前向计算 **flow-matching 训练损失**（无反向），用于快速 sanity check 与过拟合/欠拟合诊断。

示例：

```bash
python examples/training/infer_pi05_trainset.py \
  --checkpoint /path/to/checkpoints/XXXXX/pretrained_model \
  --max_batches 200
```

---

## 部署与推理

- **目标环境**：在 **AMD 笔记本电脑** 侧通过 **Docker** 封装运行环境，使用 **CUDA** 栈做 GPU 推理（常见为 AMD 平台机型搭配 NVIDIA 独显；纯核显场景需按实际驱动与 PyTorch 构建调整）。
- **镜像参考**：仓库提供 `docker/Dockerfile.user`（Python + `uv` + `pip install ".[all]"`），可作为推理/开发容器的基础；实际部署时可在此基础上挂载检查点、数据集与相机驱动依赖。

推理侧可沿用 LeRobot 的 `make_policy` 与 Pi0.5 专用前后处理（`make_pi05_pre_post_processors`），与训练配置保持一致，减少 train/serve 漂移。

---

## 代码地图（Pi0.5 相关）

| 路径 | 说明 |
|------|------|
| `src/lerobot/policies/pi05/configuration_pi05.py` | `PI05Config`：图像分辨率、chunk 长度、归一化（QUANTILES）、RTC、优化器与调度等 |
| `src/lerobot/policies/pi05/modeling_pi05.py` | `PI05Policy` / 核心网络，标注与 OpenPI 对齐 |
| `src/lerobot/policies/pi05/processor_pi05.py` | 数据集与策略之间的预处理/后处理管道 |
| `src/lerobot/policies/factory.py` | `policy.type=pi05` 的注册与构建入口 |
| `src/lerobot/scripts/lerobot_train.py` | 训练主流程 |

---

## 小结

**x4claw**（第 18 组）在 OpenClaw 赛道上实现了 **Nanobot 任务层 + Pi0.5 控制层** 的分工，以 **桌面清理** 为展示，用 **遥操 7×30 原子任务** 数据，在 **LeRobot** 中基于 **`lerobot/pi05_base` 全参微调**，并在 **云端** 完成分布式训练；验证与部署分别依托 **`infer_pi05_trainset.py`** 与 **AMD 笔记本上的 Docker/CUDA** 推理环境，形成从数据到实机的完整闭环。

---

*文档由项目组根据当前仓库状态整理；数据路径与硬件细节请按实际部署环境调整。*
