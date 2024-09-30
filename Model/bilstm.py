import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, mean_squared_error
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRMSNorm, MixtralPreTrainedModel
import fire
# from transformers.models.mixtral import MixtralDecoderLayer, MixtralRMSNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "")


# Define the dataset
class SNPDataset(Dataset):
    def __init__(self, snp_data, risk_scores):
        self.risk_scores = torch.FloatTensor(risk_scores).to(gpu_device)
        self.snp_data = torch.FloatTensor(snp_data).to(gpu_device)

    def __len__(self):
        return len(self.snp_data)

    def __getitem__(self, idx):
        return self.snp_data[idx], self.risk_scores[idx]


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for snps, risk_scores in train_loader:
            optimizer.zero_grad()
            snps, risk_scores = snps.to(gpu_device), risk_scores.to(gpu_device)
            # outputs, scores = model(snps) # for MoEModel2
            scores = model(snps)
            loss = criterion(scores, risk_scores.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch: {epoch}; ite loss: {loss}")
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')


# Function to evaluate the model
def evaluate_model(model, test_loader, threshold=0.5):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for snps, risk_scores in test_loader:
            snps, risk_scores = snps.to(gpu_device), risk_scores.to(gpu_device)
            # outputs, scores = model(snps)
            scores = model(snps)
            all_preds.extend(scores.squeeze().tolist())
            all_targets.extend(risk_scores.tolist())
            print(f"Predict: {all_preds}; Target: {all_targets}")

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    roc_auc = roc_auc_score(all_targets, all_preds)

    # Convert predictions to binary (0 or 1) based on threshold
    binary_preds = (all_preds > threshold).astype(int)

    f1 = f1_score(all_targets, binary_preds)
    precision = precision_score(all_targets, binary_preds)
    recall = recall_score(all_targets, binary_preds)

    return {
        'MSE': mse,
        'ROC AUC': roc_auc,
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall
    }


# Function to load and preprocess data from TSV file
def load_data(file_path, risk_score_column):
    # Load the TSV file
    data = pd.read_csv(file_path, sep='\t', header=None)

    # Separate features (SNP data) and target (risk scores)
    X = data.drop(columns=[0, 1, risk_score_column])
    y = data[risk_score_column]
    X_id = data[1]

    # Convert risk scores to binary (0 or 1) based on median value
    # y_binary = (y > y.median()).astype(int)

    return X, y, X_id

# 2. GRU Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.gru = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.gru(x)
        print(f"hidden size: {hidden.shape}; hidden squeeze: {hidden.squeeze(0).shape}")
        out = self.fc(hidden.squeeze(0))
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # * 2 for bidirectional

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # * 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Main function to run the entire process
def main(
        hidden_size: int = 256,
        embedding_dim: int = 32,
        output_size: int = 1,
        num_layers: int = 2,
        num_epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 0.0001,
        test_size: float = 0.2,
        random_state: int = 42,
        file_path: str="../Data/dataset/Breast/data.txt",
        cancer: str="Breast",
):
    # Hyperparameters
    # hidden_size = 256
    # embedding_dim = 32  # Dimension of SNP embeddings
    # output_size = 1
    # num_experts = 5
    # num_epochs = 20
    # batch_size = 64
    # learning_rate = 0.0001
    # test_size = 0.2
    # random_state = 42
    print(
        f"Training MoE for Cancer Risk Prediction:\n"
        f"hidden_size: {hidden_size}\n"
        f"embedding_dim: {embedding_dim}\n"
        f"output_size: {output_size}\n"
        f"num_layers: {num_layers}\n"
        f"num_epochs: {num_epochs}\n"
        f"batch_size: {batch_size}\n"
        f"learning_rate: {learning_rate}\n"
        f"test_size: {test_size}\n"
        f"random_state: {random_state}\n"
        f"file_path: {file_path}\n"
        f"cancer: {cancer}\n"
    )

    # Load and preprocess data
    # file_path = '../Data/dataset/Breast/data.txt'  # Replace with your actual file path
    risk_score_column = 2  # Replace with the actual column name for risk scores
    X, y, X_id = load_data(file_path, risk_score_column)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create datasets and data loaders
    train_dataset = SNPDataset(X_train_scaled, y_train.values)
    test_dataset = SNPDataset(X_test_scaled, y_test.values)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = X.shape[1]  # Number of SNP features

    # Model
    # model = RNNModel(input_size=X_train.shape[1], hidden_size=64, output_size=1)
    model = BiLSTM(input_size, hidden_size, num_layers, 1, dropout=0.2)

    model = model.to(gpu_device)
    # try1-- BCEWithLogitsLoss
    # criterion = nn.BCEWithLogitsLoss()
    # try2-- MSELoss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    metrics = evaluate_model(model, test_loader)

    # Print the evaluation metrics
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')

    # save model
    torch.save(model.state_dict(), f'model/{cancer}_model.pth')

if __name__ == "__main__":
    fire.Fire(main)




