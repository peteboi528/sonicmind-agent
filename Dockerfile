# MusicAgent (SonicMind) —— 单镜像，无 key 也能启动（自动降级到 MockLLM）
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_API_KEY=""

# 先装依赖（COPY 只在依赖描述变化时失效层缓存）
COPY pyproject.toml README.md ./
COPY app/ ./app/
# 生产镜像不装 [dev]（pytest/ruff）以缩小体积；以非 root 用户 app 运行。
RUN pip install "." && mkdir -p data/store data/media \
    && useradd -m -r app && chown -R app:app /app
USER app

EXPOSE 8000

# 健康检查打到 /health 端点
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
