FROM python:3.10-slim

# system dependencies required by RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY . .

# run fastapi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
