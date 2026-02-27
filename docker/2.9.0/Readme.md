
# 🚀 Build and Publish the Docker Image

This guide explains how to build a Docker image and push it to Docker Hub.   

This Docker image is designed to run the Jupyter notebooks with **PyTorch 2.9.0**, **CUDA 12.8**, and all dependencies for this project. 
It supports **GPU acceleration if available**, and falls back to CPU otherwise.

A pre-build and ready-to-use docker image for this project is available at:
```
docker pull jchen8000/pytorch_gpu_jupyterlab:latest
```

## Features

- PyTorch 2.9.0 with CUDA 12.8 + cuDNN 9
- Hugging Face Transformers, Datasets, Accelerate, PEFT
- NLP utilities: NLTK, spaCy, sentencepiece
- LangChain ecosystem support
- JupyterLab for notebook development
- FAISS (CPU) and other utilities

---

## ✅ 1. Files in This Folder
- **Dockerfile** – Defines the image build steps.
- **requirements.txt** – Lists pinned Python packages for reproducibility.

## ✅ 2. Prerequisites
- Docker installed on your machine.
- (optionsl) Docker Hub account and credentials.
- (optional) NVIDIA drivers and Docker GPU runtime for GPU support.


## ✅ 3. Log in to Docker Hub (Optional)
```
docker login
```
Enter your Docker Hub username and password.
Verify login:
```
docker info | grep Username
```

## ✅ 4. Build the Docker Image
Navigate to the folder containing your Dockerfile and requirements.txt.

Build the image
```
cd ./docker
docker build -t dockerhub-username/pytorch_gpu_jupyterlab .
```

## ✅ 5. Test the image locally
CPU test
```
docker run -it -p 8888:8888 \
  -v $(pwd):/workspace \
  dockerhub-username/pytorch_gpu_jupyterlab
```

GPU test
```
docker run -it --gpus all -p 8888:8888 \
  -v $(pwd):/workspace \
  dockerhub-username/pytorch_gpu_jupyterlab
```

Verify the version of the packages
```
pip list
```


## ✅ 6. Tag as latest (Optional)
```
docker tag dockerhub-username/pytorch_gpu_jupyterlab:2.9.0 \
           dockerhub-username/pytorch_gpu_jupyterlab:latest
```

## ✅ 7. Push to Docker Hub (Optional)
```
docker push dockerhub-username/pytorch_gpu_jupyterlab:2.9.0
docker push dockerhub-username/pytorch_gpu_jupyterlab:latest
```

## ✅ 8. Logout (Optional)
```
docker logout
```

## ✅ 9. How to Run the Docker Image

### Create a `.env` file

This file stores your API keys and secrets. 

Example `.env` file:

```env
# Hugging Face API token
HF_TOKEN=huggingface_token

# GROQ API key
GROQ_API_KEY=groq_api_key
```

**Important:**
- Do not commit your ```.env``` file with tokens to version control.
- Add ```.env``` to ```.gitignore```.


### Run the Docker container

CPU-only machine
```
docker run -it --rm -p 8888:8888 \
  --env-file .env \
  -v $(pwd):/workspace \
  dockerhub-username/pytorch_gpu_jupyterlab:latest
```

GPU machine
```
docker run -it --rm --gpus all -p 8888:8888 \
  --env-file .env \
  -v $(pwd):/workspace \
  dockerhub-username/pytorch_gpu_jupyterlab:latest
```


### Access JupyterLab

After running the container, you will see a message like:

http://127.0.0.1:8888/lab?token=abc123...

Open this URL in your browser to start using JupyterLab.


### Verify GPU availability (optional)

Inside a notebook, you can check GPU access with:
```
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

### Access .env file

Inside a notebook, you can check GPU access with:
```
import os

hf_token = os.environ.get("HF_TOKEN")
groq_key = os.environ.get("GROQ_API_KEY")

print(hf_token[:4] + "...")
print(groq_key[:4] + "...")
```
