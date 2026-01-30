FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download common NLTK assets (optional but convenient)
RUN python -m nltk.downloader punkt stopwords wordnet

# (Optional) spaCy language model — uncomment if you use it
# RUN python -m spacy download en_core_web_sm

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
