# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 20:50:56 2025

@author: julio
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
import time
import matplotlib.pyplot as plt

torch.manual_seed(42)

# --- INÍCIO DA MODIFICAÇÃO PARA GPU ---
# 1. Definir o dispositivo (CPU ou GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA está disponível! Usando a GPU.")
else:
    device = torch.device("cpu")
    print("CUDA não está disponível. Usando a CPU.")
# --- FIM DA MODIFICAÇÃO PARA GPU ---


# Gerando 1000 amostras com 4 variáveis de entrada
X = torch.randn(100000, 4) * torch.tensor([2.0, 1.0, 1.5, 2.5])
y = (
    1.2 * torch.sin(X[:, 0]) +
    0.8 * X[:, 1]**2 -
    0.5 * X[:, 2] * X[:, 3] +
    0.7 * torch.exp(-X[:, 1] * X[:, 3]) +
    0.3 * X[:, 0]**3 +
    torch.randn(100000) * 0.6
) > 0
y = y.long()

# --- INÍCIO DA MODIFICAÇÃO PARA GPU ---
# 2. Mover os dados de entrada X e y para a GPU antes de criar o TensorDataset e DataLoader
X = X.to(device)
y = y.to(device)
# --- FIM DA MODIFICAÇÃO PARA GPU ---

dataset = TensorDataset(X, y)
# O DataLoader não precisa ser modificado diretamente, ele entregará os tensores que já estão na GPU.
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class SecureModel(nn.Module):
    def __init__(self):
        super(SecureModel, self).__init__()
        self.camada1 = nn.Linear(4, 128)
        self.ativacao1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.camada2 = nn.Linear(128, 256)
        self.ativacao2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.camada3 = nn.Linear(256, 480)
        self.ativacao3 = nn.ReLU()
        self.camada_saida = nn.Linear(480, 2)

    def forward(self, x):
        x = self.ativacao1(self.camada1(x))
        x = self.dropout1(x)
        x = self.ativacao2(self.camada2(x))
        x = self.dropout2(x)
        x = self.ativacao3(self.camada3(x))
        x = self.camada_saida(x)
        return x

# Função de treinamento
def train(model, optimizer, data_loader, noise_multiplier, max_grad_norm):
    model.train()
    criterion = nn.CrossEntropyLoss()

    # --- INÍCIO DA MODIFICAÇÃO PARA GPU ---
    # 3. O critério de perda não precisa ser explicitamente movido, mas o modelo e os dados já estarão na GPU.
    # --- FIM DA MODIFICAÇÃO PARA GPU ---

    privacy_engine = PrivacyEngine()

    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    start_time = time.time()
    for epoch in range(3):
        for X_batch, y_batch in data_loader:
            # --- INÍCIO DA MODIFICAÇÃO PARA GPU ---
            # 4. Mover cada batch para o dispositivo correto (GPU) dentro do loop
            #    Se os dados já foram movidos para a GPU no início (X, y),
            #    o DataLoader já os entregará na GPU. Mas é uma boa prática garantir:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # --- FIM DA MODIFICAÇÃO PARA GPU ---

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    elapsed_time = time.time() - start_time
    return elapsed_time, privacy_engine

# Função de avaliação
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            # --- INÍCIO DA MODIFICAÇÃO PARA GPU ---
            # 5. Mover cada batch para o dispositivo correto (GPU) dentro do loop de avaliação
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # --- FIM DA MODIFICAÇÃO PARA GPU ---

            output = model(X_batch)
            predicted = output.argmax(dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

# Comparação de diferentes noise_multipliers
noise_multipliers = [1.0, 4.0, 10.0,]
times = []
accuracies = []
epsilons = []

for noise in noise_multipliers:
    model = SecureModel()
    # --- INÍCIO DA MODIFICAÇÃO PARA GPU ---
    # 6. Mover o modelo para a GPU ANTES de passá-lo para o otimizador e PrivacyEngine
    model.to(device)
    # --- FIM DA MODIFICAÇÃO PARA GPU ---
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    elapsed_time, privacy_engine = train(model, optimizer, data_loader, noise, 1.0)
    # --- INÍCIO DA MODIFICAÇÃO PARA GPU ---
    # 7. Criar um DataLoader para avaliação (se for diferente do treinamento)
    #    e garantir que os dados estejam no device correto (já feito acima para X, y)
    eval_dataset = TensorDataset(X, y) # Reutiliza os tensores X e y que já estão na GPU
    eval_data_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    accuracy = evaluate(model, eval_data_loader)
    # --- FIM DA MODIFICAÇÃO PARA GPU ---

    # Estimando o epsilon corretamente
    delta = 1e-5
    epsilon = privacy_engine.get_epsilon(delta)

    times.append(elapsed_time)
    accuracies.append(accuracy)
    epsilons.append(epsilon)

    print(f"Noise: {noise}, Tempo: {elapsed_time:.2f}s, Acurácia: {accuracy:.4f}, Epsilon: {epsilon:.2f}")

# Plotando resultados (gráficos separados para melhor visualização)
def plot_results(noise_multipliers, times, accuracies, epsilons):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: Tempo vs. Noise Multiplier
    axes[0].plot(noise_multipliers, times, marker="o", linestyle="-", color="tab:red", label="Tempo de Treinamento")
    axes[0].set_xlabel("Noise Multiplier")
    axes[0].set_ylabel("Tempo (s)")
    axes[0].set_title("Tempo de Treinamento vs. Noise Multiplier")
    axes[0].grid(True)
    axes[0].legend()

    # Gráfico 2: Acurácia e Epsilon vs. Noise Multiplier
    axes[1].plot(noise_multipliers, accuracies, marker="s", linestyle="--", color="tab:blue", label="Acurácia")
    axes[1].plot(noise_multipliers, epsilons, marker="^", linestyle="-.", color="tab:green", label="Epsilon (Privacidade)")
    axes[1].set_xlabel("Noise Multiplier")
    axes[1].set_ylabel("Valor")
    axes[1].set_title("Acurácia e Epsilon vs. Noise Multiplier")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

plot_results(noise_multipliers, times, accuracies, epsilons)