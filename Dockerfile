# 基础镜像：轻量 Python
FROM python:3.11-slim

# 安装 aria2（这个工具运行下载任务必需的依赖）
RUN apt-get update && \
    apt-get install -y --no-install-recommends aria2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 装 Python 依赖，利用 Docker 缓存层，改代码不用重新装依赖
RUN pip install --no-cache-dir flask requests

# 拷贝主程序
COPY app.py .

# 工具默认监听的端口
EXPOSE 5678

# 显式标记"这是在 Docker 容器里跑"，供 app.py 判断是否要接管 SIGTERM
ENV RUNNING_IN_DOCKER=1

# 容器内默认保存目录，实际使用时通过 docker-compose 的 volumes 映射到 NAS 真实存储路径
ENV HF_DOWNLOADER_SAVE_DIR=/downloads
VOLUME ["/downloads"]

CMD ["python3", "app.py"]
