# LLM-CogMap
**LLM-CogMap: 面向具身智能的轻量级 3D 语义拓扑建图与认知系统**

![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue.svg) ![C++17](https://img.shields.io/badge/C++-17-00599C.svg) ![Python3](https://img.shields.io/badge/Python-3.8+-FFD43B.svg)

> **"将 3D 稠密点云压缩为轻量级语义拓扑，赋予大语言模型（LLM）真正的三维空间物理直觉。"**

##  项目简介

`LLM-CogMap` 是一个面向边缘计算平台（如 NVIDIA Jetson 系列）设计的端云异构具身智能导航框架。项目针对传统 3D 稠密语义点云导航**带宽占用大、极易丢失目标、大模型缺乏空间几何概念**等痛点，提出了一套从“底层像素级感知”到“高层自然语言决策”的完整解耦架构。

目前，本系统已基于 **TUM RGB-D 数据集**完成了全链路 Software-in-the-Loop (SIL) 闭环验证。

---

## 系统架构图

本项目严格遵循“算力层与业务层解耦”的工程哲学，分为两大核心节点：

### 1. 底层感知与记忆引擎 (C++ Node)
* **高性能几何处理**：基于 `cv::connectedComponents` 与形态学操作，实现 2D 掩码到 3D 实例包围盒（AABB）的极速转换。
* **背景剥离与防粘连**：通过 $O(1)$ 级深度跳变检测与掩码重构，彻底解决 YOLO 掩码溢出导致的背景与物体粘连问题。
* **全局长时记忆 (GC 机制)**：自主实现带有 EMA（指数移动平均）滤波和多级信用评级（Hit/Miss Count）的实例管理器。有效过滤视觉瞬时幻觉，实现跨帧、抗遮挡的全局空间状态持久化。
* **低带宽输出**：将百万级稠密点云压缩为极度轻量的 JSON 语义拓扑数据（包含质心、AABB、深度等），通过 ROS 2 话题高频广播。

### 2. 具身认知中枢 (Python Node)
* **语义翻译中间件**：内置 COCO 标准字典，将底层的 `semantic_id` 实时翻译为 LLM 易读的自然语言实体。
* **空间几何直觉注入**：通过特制的 System Prompt，向大模型注入机器人 FLU（前-左-上）绝对坐标系规则，克服了传统 LLM“左右不分”的坐标系幻觉。
* **零代码闭环控制**：接收人类自然语言模糊指令（如*“去左边那台显示器那里”*），LLM 自主进行逻辑链（CoT）推理，提取目标 3D 坐标，计算安全避障偏移量，并下发标准 `PoseStamped` 导航指令驱动底盘（兼容 Nav2）。

---

## 核心亮点 (Why this project stands out?)

- **极低的算力与带宽消耗**：彻底摒弃传递点云或体素给大模型的传统方案，依靠高精 AABB JSON 拓扑，使端云通信带宽下降 90% 以上，极度适配边缘算力平台。
- **跨越“断电”的长时记忆**：底层 C++ 维持 RAM 级高频状态机，上层 Python 建立跨帧字典索引，即使物体移出相机视野，机器人依然“记得”它在全局坐标系中的绝对位置。
- **极其精准的自然语言交互**：完美支持多重约束指令查找（如：*“去最远的那把椅子”*、*“去地上那个水瓶”*），大模型自主完成空间比对。

---

### 环境要求与核心依赖

本项目涉及跨语言（C++/Python）与异构计算，为保证系统顺利编译运行，请确保您的环境满足以下依赖：

#### 1. 操作系统与基础中间件

* **OS**: Ubuntu 22.04 LTS
* **ROS 2**: Humble Hawksbill (需安装 `ros-humble-desktop`)
* **Build Tools**: CMake (>= 3.16) 与支持 C++17 的编译器 (GCC/G++ >= 9.0)

#### 2. 视觉感知与深度学习 (2D Vision & AI)

* **OpenCV**: 4.x (用于基础图像处理与 2D 连通域计算)
* **Ultralytics YOLOv8**: 用于高频像素级语义分割与目标检测
* **CUDA & cuDNN**: (强烈推荐) 用于 YOLOv8 模型推理的 GPU 硬件加速

#### 3. 空间几何与 SLAM 后端 (3D Geometry & SLAM)

* **ORB-SLAM3**: 用于提供高精度的相机位姿（$t_{wc}$）与全局参考系
* **PCL (Point Cloud Library)**: 1.12+ (用于 3D 点云数据结构的过滤、存储与包围盒计算)
* **Eigen3**: 3.3+ (底层数学库，ORB-SLAM3 与 PCL 的前置刚需依赖，用于高频矩阵运算)

#### 4. 认知中枢与大模型接口 (Cognitive Brain)

* **Python**: 3.8+ 
* **ROS 2 Bridge**: `cv_bridge` (用于实现 ROS Image 消息与 OpenCV/NumPy 阵列的无缝转换)
* **LLM SDK**: `openai` (用于兼容所有标准 OpenAI API 格式的大语言模型服务)

### 编译构建

```bash
mkdir -p ~/semantic_slam_ws/src
cd ~/semantic_slam_ws/src
git clone https://github.com/atat1010/LLM-CogMap.git
cd ..
colcon build --symlink-install
source install/setup.bash
```

### VLM交互

```bash
ros2 topic pub --once /vlm/prompt std_msgs/msg/String "{data: 'your_msg'}"
```

本项目支持任何兼容 OpenAI 接口的大模型（推荐使用 Qwen 等多模态模型）：

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

同时修改`vlm_brain_node.py`中的`model_name`变量以指定使用的模型。

### 运行

```bash
ros2 launch orb_slam3_ros2 semantic_slam.launch.py
```

### 运行示例

**点云地图**：
![点云地图](image.png)

**原始动态场景**：
![原始动态场景](<without yolo.png>)

**YOLO动态剔除**：
![YOLO动态剔除](<with yolo.png>)

**VLM交互**：
![alt text](msg.png)
![alt text](reply.png)

---

## TODO

- [ ] **实机部署**：将系统迁移至 NVIDIA Jetson Orin 平台，结合 TensorRT 对 YOLOv8 进行 INT8 量化加速。
- [ ] **深度图修复**：引入孔洞填充算法，应对真实世界（如玻璃、反光材质）造成的深度图缺失。
- [ ] **Nav2 深度集成**：结合代价地图（Costmap），将大模型输出的粗略目标点交由局部规划器进行精细化碰撞检测与微调。
- [ ] **人机交互**：脱离终端，构建真正方便的 Web UI，支持远程监控与指令输入。


