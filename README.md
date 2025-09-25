# Audio SSL Anomaly Detection

This project implements **self-supervised audio anomaly detection** using **BYOL (Bootstrap Your Own Latent)** in PyTorch Lightning. It learns to represent normal audio patterns and detects anomalies in new audio recordings without requiring labeled data.

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Folder Structure](#folder-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [1. Preprocess Data](#1-preprocess-data)  
  - [2. Train BYOL Encoder](#2-train-byol-encoder)  
  - [3. Extract Embeddings](#3-extract-embeddings)  
  - [4. Train Anomaly Detector](#4-train-anomaly-detector)  
  - [5. Run Demo Inference](#5-run-demo-inference)  
- [Web API Deployment](#web-api-deployment)  


---

## Overview

The pipeline consists of:

1. **Self-supervised BYOL model**: Learns compact embeddings of normal audio using mel-spectrograms.  
2. **Anomaly detection model**: Uses embeddings to detect unusual audio patterns (e.g., machine faults, abnormal environmental sounds).  
3. **Demo and API**: Allows inference on new audio files and provides anomaly scores.

---

## Features

- Train BYOL on normal audio files only  
- Extract embeddings for all audio files  
- Train an anomaly detector (Isolation Forest or similar)  
- Real-time anomaly inference  
- Optional web API via FastAPI  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Bhargavram882/audio-ssl-anomaly-detection.git
cd audio-ssl-anomaly-detection
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
python3 -m src.preprocess
python3 -m src.ssl_models.train_byol
python3 -m src.extract_embeddings
python3 -m src.demo.demo_inference data/raw/anomaly/anom_000.wav

