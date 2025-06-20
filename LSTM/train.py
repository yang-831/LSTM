from operator import truediv
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix,vstack, hstack
from scipy.sparse.linalg import spsolve, factorized
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
a = 0
b = 1
c = 0
d = 1
epsl = 0.025
Nx = 100
Ny = 100
dx = (b - a) / Nx
dy = (d - c) / Ny
h = (d - c) / Nx
TT = 1
dt = 0.0001
Ndt = round(TT / dt)
dt = TT / Ndt

class CustomActivation(nn.Module):
    def forward(self, x, out, dx):
        inputini = x[:, 0, :]
        W_b = 1 - out ** 2
        W_b_sum = torch.sum(W_b, dim=1) * dx
        penalty_sum = torch.sum(inputini - out, dim=1) * dx
        output = out + (W_b / W_b_sum.unsqueeze(1)) * penalty_sum.unsqueeze(1)
        return output

class LSTMModelWithPolyBias(nn.Module):
    def __init__(self, input_dim=2500, hidden1_dim=300, hidden2_dim=300, output_dim=2500):
        super(LSTMModelWithPolyBias, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim
        self.lstm1 = nn.LSTM(input_dim, hidden1_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1_dim, hidden2_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden2_dim, hidden2_dim, batch_first=True)
        self.fc = nn.Linear(hidden2_dim, output_dim)
        self.tanh = nn.Tanh()
        self.activation = CustomActivation()

    def polynomial_fit_and_correct(self, output, X_batch, degree=1):
        batch_size, output_dim = output.shape
        corrected_output = torch.zeros_like(output)
        for i in range(batch_size):
            x = np.array([3, 4, 5])
            y = X_batch[i, -3:, :].squeeze(0).detach().cpu().numpy()
            coeffs = np.polyfit(x, y, degree)
            corrected_output[i] = torch.tensor(np.polyval(coeffs, 6))
        return corrected_output

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out3, _ = self.lstm3(lstm_out2)
        out = self.fc(lstm_out3[:, -1, :])
        poly_bias = self.polynomial_fit_and_correct(out, x)
        out_with_bias = out + poly_bias
        out = self.activation(x, out_with_bias, dx)
        return out

def compute_gradient(u, dx):
    grad_u = torch.zeros_like(u)
    grad_u[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx
    grad_u[:, -1] = (u[:, -1] - u[:, -2]) / dx
    return grad_u

def custom_loss(X_batch, y_batch, y_pred, dx, epsl):
    x_last = X_batch[:, -1, :]
    y_true = y_batch
    F_c_true = 0.25 * (y_true ** 2 - 1) ** 2
    F_c_pred = 0.25 * (y_pred ** 2 - 1) ** 2
    F_c_last_true = 0.25 * (x_last ** 2 - 1) ** 2
    grad_true = compute_gradient(y_true, dx)
    grad_pred = compute_gradient(y_pred, dx)
    grad_last_true = compute_gradient(x_last, dx)
    row_sum_true = torch.sum(F_c_true + 0.5 * epsl ** 2 * grad_true ** 2, dim=1)
    row_sum_pred = torch.sum(F_c_pred + 0.5 * epsl ** 2 * grad_pred ** 2, dim=1)
    row_sum_last_true = torch.sum(F_c_last_true + 0.5 * epsl ** 2 * grad_last_true ** 2, dim=1)
    energy_pred_sub = row_sum_pred * dx
    energy_last_true_sub = row_sum_last_true * dx
    difference_E = energy_pred_sub - energy_last_true_sub
    energy_penalty = torch.mean(torch.square(difference_E))
    mse = torch.mean(torch.square(y_true - y_pred))
    print(f'MSE: {mse.item()}, Energy Penalty: {energy_penalty.item()}')
    LOSS = mse
    return LOSS

def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size, dx, epsl):
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0015)
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size].requires_grad_(True)
            y_batch = y_train[i:i + batch_size].requires_grad_(True)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = custom_loss(X_batch, y_batch, output, dx, epsl)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = custom_loss(X_test, y_test, test_output, dx, epsl)
        print(f'Test Loss: {test_loss.item()}')
    return losses

data = np.load('data.npz')
X = data['data_u_values']
y = data['data_u_final']
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
model = LSTMModelWithPolyBias(input_dim=2500, hidden1_dim=300, hidden2_dim=300, output_dim=2500)
losses = train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=400, dx=dx, epsl=epsl)

model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
torch.save(model.state_dict(), 'lstm_model.pth')
