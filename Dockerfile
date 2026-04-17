FROM python:3.10-slim

# system dependencies for RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# set workdir
WORKDIR /app

# copy project
COPY . /app

# install python deps
RUN pip install --no-cache-dir -r requirements.txt

# expose port
EXPOSE 8000

# start app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
