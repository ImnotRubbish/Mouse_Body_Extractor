# 小鼠行为学分析一站式处理平台

## 项目概述
本项目基于DeepLabCut（DLC）和Flask框架，构建了一个用于小鼠行为学分析的交互式网页平台。用户可以通过该平台完成从视频处理到模型训练的全流程操作。

## 主要功能
1. **项目创建与加载**
   - 创建新项目：输入项目名称、用户名称，选择源视频目录和工作目录
   - 加载已有项目：通过选择配置文件加载已有项目

2. **视频处理**
   - 视频帧提取：设置帧提取间隔，从视频中提取关键帧
   - 标记点注册：在提取的帧上进行标记点标注
   - 标记点检查：验证标记点标注的准确性

3. **模型训练**
   - 创建训练数据集
   - 训练行为分析模型
   - 模型评估

4. **结果导出**
   - 导出带标记点的行为视频
   - 支持多种视频格式（.mp4, .avi, .mov）

5. **配置管理**
   - 修改项目配置：包括bodyparts、skeleton、dotsize等参数
   - 实时保存配置更改

## 配置要求
1. 操作系统
    - Windows 10/11 （推荐）

2. Python版本
    - Python 3.7+ （推荐3.8或更高版本）  

3. 依赖库
    - TensorFlow 2.x （推荐2.4或更高版本）
    - DeepLabCut （推荐2.2或更高版本）
    - Flask （推荐2.0或更高版本）
    - OpenCV （推荐4.5或更高版本）
    - NumPy （推荐1.19或更高版本）
    - YAML （推荐5.4或更高版本）  

4. 硬件要求
    - CPU : 至少4核处理器（推荐8核或更高）
    - GPU : NVIDIA GPU（推荐RTX 2060或更高，支持CUDA 11.0+）
    - 内存 : 至少16GB（推荐32GB或更高）
    - 存储 : 至少50GB可用空间（推荐SSD）  

5. 软件环境
    - CUDA : 11.0或更高版本（如果使用GPU）
    - cuDNN : 8.0或更高版本（如果使用GPU）
    - NVIDIA驱动 : 450.80.02或更高版本（如果使用GPU）  

6. 视频格式支持
    - MP4 （推荐）
    - AVI
    - MOV  

7. 浏览器兼容性
    - Chrome （推荐最新版本）
    - Firefox （推荐最新版本）
    - Edge （推荐最新版本）

## 安装与运行
1. 克隆本项目：
   ```bash
   git clone https://github.com/your-repo/Mouse_Body_Extractor.git

2. 安装依赖：
   ```bash
   pip install -r requirements.txt

3. 启动Flask应用：
   ```bash
   python app.py    


4. 访问URL_ADDRESS4. 访问http://localhost:5000在浏览器中打开平台。


## 文件结构
    ```bash
    project/
    ├── app.py                # Flask应用主文件
    ├── dlc_module.py         # DLC功能实现
    ├── templates/            # HTML模板
    │   ├── create_project.html
    │   ├── export_video.html
    │   ├── extract_frames.html
    │   ├── label_frames.html
    │   ├── load_project.html
    │   ├── modify_config.html
    │   ├── train_model.html
    │   └── welcome.html
    └── README.md             # 项目说明文件
    ```

## 使用说明
- 访问欢迎页面，选择"创建新项目"或"加载项目"
- 按照页面提示完成各项操作
- 所有操作结果和日志信息都会实时显示在页面上
- 导出的视频和模型文件会保存在项目目录下的video文件夹中

## 注意事项
- 确保您的系统满足系统要求。
- 确保您的网络环境正常。
- 确保您的GPU支持TensorFlow 2.x。
- 确保您的DeepLabCut版本与项目要求兼容。
- 确保您的视频格式与项目支持的格式一致。
- 确保您的标记点标注准确。
- 确保您的配置文件正确。
- 确保您的项目名称和用户名称唯一。
- 确保您的项目目录存在。
- 确保您的项目目录下有config.yaml文件。