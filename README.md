# Federated Learning for DDoS Attack Detection in Cybersecurity
## Project Overview

DDoS attacks are a significant threat in today’s digital landscape. Traditional detection systems often rely on centralized data aggregation, which can compromise user privacy and lead to scalability issues. This project addresses these challenges by:
- Implementing a decentralized federated learning system.
- Training a deep neural network model (named `DDoSNet`) using local data on simulated clients.
- Aggregating model updates on a central server using the FedAvg strategy.

## Repository Contents

- **client.py**  
  Implements the Flower client including:
  - A deep learning model (`DDoSNet`) built with PyTorch.
  - Functions for data loading, preprocessing, training, and evaluation.
  - Command-line arguments to specify data partitions for simulating multiple clients.  
  :contentReference[oaicite:0]{index=0}

- **server.py**  
  Implements the federated learning server using:
  - The Flower framework’s `ServerApp` and FedAvg strategy for model aggregation.
  - Custom aggregation of client metrics to compute a weighted accuracy.  
  :contentReference[oaicite:1]{index=1}

## Requirements

- Python 3.8+
- PyTorch
- Flower
- Pandas
- tqdm
- argparse
- Other standard Python libraries
