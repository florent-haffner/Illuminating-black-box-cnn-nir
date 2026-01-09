# Illuminating the black-box: An explainability study (XAI) of a Deep CNN for near-infrared spectroscopy

The code for the paper "IPA under the light", review in progress at "Chemometrics and Intelligent Labs system"

This study use two environments : one using PyTorch for calculating SHAP values, another one using TensorFlow for calculating Grad-CAM activation. 

## Environment SHAP (PyTorch)
Copy and paste each lines

```bash
conda create -n xai-shap python=3.10

conda activate xai-shap

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```

To open an environment and jupyterlab :
```bash
conda activate xai-shap
jupyter lab .
```

## Environment - Grad-CAM (TensorFlow)
Copy and paste each lines

```bash
conda create -n xai-gradcam python=3.10

conda activate xai-gradcam

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

pip install "tensorflow<2.11"
pip install -r requirements.txt
```

To open an environment and jupyterlab :
```bash
conda activate xai-gradcam
jupyter lab .
```
