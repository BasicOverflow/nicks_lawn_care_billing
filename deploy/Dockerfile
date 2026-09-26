FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "ray[default]>=2.9" || true

COPY app ./app
COPY ocr ./ocr
COPY static ./static
COPY run.py .

ENV HOST=0.0.0.0
ENV PORT=8787
ENV PYTHONUNBUFFERED=1
EXPOSE 8787

CMD ["python", "-u", "run.py"]
