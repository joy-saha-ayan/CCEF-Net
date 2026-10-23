import torch
import torchvision
import transformers
import datasets
import sentence_transformers
import cv2
import PIL
import numpy as np
import pandas as pd
import sklearn
import faiss
import streamlit
import gradio
import captum

print("All core imports successful")

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("opencv:", cv2.__version__)
print("Pillow:", PIL.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))