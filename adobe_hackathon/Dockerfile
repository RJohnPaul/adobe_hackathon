FROM python:3.10-slim

# Install system dependencies for OpenCV, PyMuPDF, and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libmupdf-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main script and model files
COPY process_pdfs.py .
COPY yolo-doclaynet.onnx .
COPY doclaynet.yaml .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set environment variable to avoid matplotlib GUI backend issues
ENV MPLBACKEND=Agg

CMD ["python", "process_pdfs.py"]
