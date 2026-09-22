FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PROJECT_ROOT=/workspace
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt requirements-common.txt requirements-docker.txt ./
ARG INSTALL_ML_DEPS=true
RUN if [ "$INSTALL_ML_DEPS" = "true" ]; then \
      python -m pip install --upgrade pip && \
      python -m pip install -r requirements-docker.txt && \
      python -m pip check && \
      python -c "import networkx, numpy, ogb, openai, pandas, requests, sklearn, torch, torch_geometric"; \
    fi
COPY . .
EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=3s --retries=3 CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1
CMD ["python","-u","dashboard/server.py"]
