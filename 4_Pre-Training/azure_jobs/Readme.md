# Training on Azure Machine Learning

If you have a Azure subscription, you might want to submit the following two training scripts to Azure as training jobs.

- **`en2de_translation_job.py`** — English → German translation training script 
- **`en2fr_translation_job.py`** — English → French translation training script

Each script trains a Transformer model, logs training metrics with MLflow, and writes artifacts (checkpoints, vocab, metrics, plots) which are automatically captured by Azure ML.

---

## 1. Prerequisites

Before running the jobs, you should already have:

- An **Azure subscription**  
- A **Resource Group** and an **Azure ML Workspace**
- A **GPU Compute Cluster**  
- The Azure ML CLI v2 extension:
    ```
    az extension add -n ml -y
    ```

## 2. Choose a PyTorch + GPU Environment

Azure ML provides curated environments for PyTorch, but their names and versions differ depending on subscription, region, and workspace.  
You might want to list all the curated environments available to your subscription:
```bash
az ml environment list --registry-name azureml -o table
```
Pick an environment that support **PyTorch** and **CUDA(GPU support)**.

## 3. YAML Job Files

Azure ML CLI v2 requires all jobs to be submitted via a YAML job definition. This folder includes two YAML files:

- ```en2de-job.yml```
- ```en2fr-job.yml```

You might have to modify the two YAML files based on your subscription, environment, and compute cluster name, etc. You might also want to modify the parameters of training script, such as epochs, batch-size, learning-rate(lr), etc.

## 4.Submit the Jobs

Submit jobs using the Azure ML CLI:

```bash
az ml job create -f en2de-job.yml
az ml job create -f en2fr-job.yml
```

## 5. Monitor Training

Then you can go to the Azure portal to view the jobs.