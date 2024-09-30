import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


# Define an expert network
class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the gating network
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_size, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)


# Define the Mixture of Experts model
class MoEModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_experts):
        super(MoEModel, self).__init__()
        self.gating = GatingNetwork(input_size, num_experts)
        self.experts = nn.ModuleList([Expert(input_size, hidden_size, output_size) for _ in range(num_experts)])

    def forward(self, x):
        gates = self.gating(x)
        final_output = torch.zeros(x.size(0), 1).to(gpu_device)
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            final_output += gates[:, i].unsqueeze(1) * expert_output
        return final_output


# Define the Mixture of Experts model with SNP embeddings
class MoEModelWithEmbeddings(nn.Module):
    def __init__(self, num_snps, embedding_dim, hidden_size, output_size, num_experts):
        super(MoEModelWithEmbeddings, self).__init__()
        self.snp_embeddings = nn.Embedding(num_snps, embedding_dim)
        self.gating = GatingNetwork(num_snps * embedding_dim, num_experts)
        self.experts = nn.ModuleList(
            [Expert(num_snps * embedding_dim, hidden_size, output_size) for _ in range(num_experts)])

    def forward(self, x):
        # x shape: (batch_size, num_snps)
        batch_size, num_snps = x.shape

        # Get embeddings for each SNP
        embedded_snps = self.snp_embeddings(torch.arange(num_snps).to(gpu_device))  # (num_snps, embedding_dim)

        # Multiply input data with corresponding SNP embeddings
        x_embedded = x.unsqueeze(2) * embedded_snps.unsqueeze(0)  # (batch_size, num_snps, embedding_dim)

        # Flatten the embedded input
        x_flat = x_embedded.view(batch_size, -1)  # (batch_size, num_snps * embedding_dim)

        # Apply gating and experts
        gates = self.gating(x_flat)
        final_output = torch.zeros(batch_size, 1).to(gpu_device)
        for i, expert in enumerate(self.experts):
            expert_output = expert(x_flat)
            final_output += gates[:, i].unsqueeze(1) * expert_output
        return final_output

# Lucas try Mixtral model for cancer risk prediction
# 有问题：目前预测的y一直是0
class MoEModel2(MixtralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MixtralDecoderLayer`]

    Args:
        config: MixtralConfig
    """
    # def __init__(self, config, num_snps, embedding_dim):
    #     super(MoEModel2, self).__init__()
    def __init__(self, config: MixtralConfig, num_snps, embedding_dim):
        super().__init__(config)
    # def __init__(self, config: MixtralConfig):
    #     super().__init__(config)
        self.config = config
        self.snp_embeddings = nn.Embedding(num_snps, embedding_dim)
        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.score = nn.Linear(config.hidden_size*num_snps, 1, bias=False)
        self.relu = nn.ReLU()

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Ignore copy
    def forward(self, x, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                output_router_logits: Optional[bool] = None,
                ):
        # x shape: (batch_size, num_snps)
        batch_size, num_snps = x.shape

        # Get embeddings for each SNP
        embedded_snps = self.snp_embeddings(torch.arange(num_snps).to(gpu_device))  # (num_snps, embedding_dim)

        # Multiply input data with corresponding SNP embeddings
        x_embedded = x.unsqueeze(2) * embedded_snps.unsqueeze(0)  # (batch_size, num_snps, embedding_dim)

        # Flatten the embedded input
        x_flat = x_embedded.view(batch_size, -1)  # (batch_size, num_snps * embedding_dim)

        # Apply gating and experts
        # gates = self.gating(x_flat)
        # final_output = torch.zeros(batch_size, 1).to(gpu_device)
        # for i, expert in enumerate(self.experts):
        #     expert_output = expert(x_flat)
        #     final_output += gates[:, i].unsqueeze(1) * expert_output
        # return final_output

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        inputs_embeds = x_embedded
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
            )
            hidden_states = layer_outputs[0]
            # add hidden states from the last decoder layer

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        hidden_states = hidden_states.view(batch_size, -1)
        score = self.score(hidden_states)
        score = self.relu(score)
        return hidden_states, score


class TransformerExpert(nn.Module):
    def __init__(self, num_snps, model_dim=64, num_heads=4, num_layers=2):
        super(TransformerExpert, self).__init__()
        self.snp_embeddings = nn.Embedding(num_embeddings=num_snps, embedding_dim=model_dim)  # Assuming SNPs are categorical
        self.positional_encoding = nn.Parameter(torch.zeros(1,num_snps, model_dim))  # Sequence length 100
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim*2)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_snps)
        batch_size, num_snps = x.shape

        # Get embeddings for each SNP
        embedded_snps = self.snp_embeddings(torch.arange(num_snps).to(gpu_device))  # (num_snps, embedding_dim)

        # Multiply input data with corresponding SNP embeddings
        x_embedded = x.unsqueeze(2) * embedded_snps.unsqueeze(0)  # (batch_size, num_snps, embedding_dim)

        # Flatten the embedded input
        x_flat = x_embedded.view(batch_size, -1)  # (batch_size, num_snps * embedding_dim)

        x = x_embedded + self.positional_encoding
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  # Pooling across sequence dimension
        x = self.fc(x)
        return x


class MoEModel3(nn.Module):
    def __init__(self, num_experts=3, num_snps=50):
        super(MoEModel3, self).__init__()
        # self.experts = nn.ModuleList([TransformerExpert(num_snps, model_dim=64, num_heads=4, num_layers=2) for _ in range(num_experts)])
        self.experts = TransformerExpert(num_snps, model_dim=64, num_heads=4, num_layers=2)
        # self.gate = nn.Linear(num_snps * 100, num_experts)  # Gate based on input SNP data (flattened)
        # self.gate = GatingNetwork(num_snps, num_experts)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # Flatten the SNP data
        # gate_weights = F.softmax(self.gate(x_flat), dim=1)  # Compute gating weights

        # expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch_size, num_experts, 1)
        expert_outputs = self.experts(x)
        # weighted_sum = torch.sum(gate_weights.unsqueeze(2) * expert_outputs, dim=1)  # Weighted sum of expert outputs
        # return weighted_sum.squeeze(1)

        return expert_outputs

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs, test_loader):
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

        # Evaluate the model
        metrics = evaluate_model(model, test_loader)

        # Print the evaluation metrics
        for metric, value in metrics.items():
            print(f'Epoch: {epoch}; Result: {metric}: {value:.4f}')


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


# Main function to run the entire process
def main(
        hidden_size: int = 256,
        embedding_dim: int = 32,
        output_size: int = 1,
        num_experts: int = 5,
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
        f"num_experts: {num_experts}\n"
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

    # Initialize the model, loss function, and optimizer
    # try1-- MoEModel-->MSE: 141.4823 ROC AUC: 0.6091 F1 Score: 0.5601 Precision: 0.5705 Recall: 0.5501
    # input_size = X.shape[1]  # Number of SNP features
    # model = MoEModel(input_size, hidden_size, output_size, num_experts)

    # try2-- MoEModelWithEmbeddings--->
    # num_snps = X.shape[1]  # Number of SNP features
    # model = MoEModelWithEmbeddings(num_snps, embedding_dim, hidden_size, output_size, num_experts).to(gpu_device)

    # try3-- MoEModel2 with Mixtral model
    # 预测y一直为0，有问题
    # num_snps = X.shape[1]  # Number of SNP features
    # config = MixtralConfig.from_pretrained('config')
    # print(f"MoEModel2 config: {config}")
    # print(f"num_snps: {num_snps}; ")
    # model = MoEModel2(config, num_snps, embedding_dim).to(gpu_device)
    input_size = X.shape[1]  # Number of SNP features
    model = MoEModel3(num_experts, input_size)

    model = model.to(gpu_device)
    # try1-- BCEWithLogitsLoss
    # criterion = nn.BCEWithLogitsLoss()
    # try2-- MSELoss
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs, test_loader)



    # save model
    torch.save(model.state_dict(), f'model/{cancer}_model.pth')


if __name__ == "__main__":
    fire.Fire(main)