# Audio SSL Anomaly Detection
1. `python src/data/generate_synthetic.py` – make synthetic audio  
2. `python src/preprocess.py` – compute log-mel features  
3. `python src/ssl/train_byol.py` – train BYOL encoder (5 epochs)  
4. `python src/extract_embeddings.py` – save embeddings  
5. `python src/train_anomaly.py` – train IsolationForest & report AUC  
6. `python src/demo_inference.py data/raw/anomaly/anom_000.wav` – score a file
