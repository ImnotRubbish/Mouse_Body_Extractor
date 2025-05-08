# KeyPoint MoSeq Web App

这是一个基于 FastAPI 和 Jupyter MoSeq 的网页应用，允许用户上传 `.h5` 格式的 DeepLabCut 关键点数据，并在后台运行 KeyPoint MoSeq 分析流程，提取行为序列并可视化结果。

## 📦 功能

- 支持上传 `.h5` 格式的关键点数据
- 在后台运行 KeyPoint MoSeq 分析流程
- 自动生成轨迹图、聚类视频网格图、树状图等
- 可导出 CSV 行为序列数据

## 🧰 技术栈

- FastAPI
- Jinja2 模板引擎
- HTML/CSS 前端
- KeyPoint MoSeq Python 库

## 🚀 快速开始

 克隆仓库

   ```bash
   git clone https://github.com/yourname/keypoint-moseq-web-app.git
   cd keypoint-moseq-web-app
   ```
安装依赖：
pip install -r requirements.txt

运行服务：
uvicorn app:app --reload

打开浏览器访问：
http://localhost:8000