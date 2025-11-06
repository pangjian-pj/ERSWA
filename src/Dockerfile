# 使用官方 Python 3.11 slim 镜像作为基础镜像
FROM mirror.gcr.io/library/python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
# 防止 Python 生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
# 确保 Python 输出直接发送到终端，不经过缓冲
ENV PYTHONUNBUFFERED=1

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .
COPY analysis.py .
COPY explain.py .
COPY recommend.py .

# 复制数据文件
COPY data/u.data data/u.data
COPY data/u.item data/u.item

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 健康检查（可选）
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 启动 Streamlit 应用
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]