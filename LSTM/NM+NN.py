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

X = np.zeros((Nx, 1))
Y = np.zeros((Ny, 1))

for i in range(Nx):
    X[i] = a + h / 2 + (b - a) / Nx * i
for i in range(Ny):
    Y[i] = c + h / 2 + (d - c) / Ny * i
X, Y = np.meshgrid(X, Y)

X = X.flatten()
Y = Y.flatten()

Boundary = np.zeros(Nx * Ny)
for i in range(Nx * Ny):
    if abs(X.flatten()[i] - (a + h / 2)) < h ** 3 or abs(X.flatten()[i] - (b - h / 2)) < h ** 3 or abs(Y.flatten()[i] - (c + h / 2)) < h ** 3 or abs(Y.flatten()[i] - (d - h / 2)) < h ** 3:
        Boundary[i] = 1
polygon = Delaunay(np.column_stack((X, Y)))

N = len(X)
K = lil_matrix((N, N))
A = lil_matrix((N, N))
for i in range(N):
    A[i, i] = 1
    if Boundary[i] == 0:
        DiffusionWeight = [1, 1, -4, 1, 1]
        if i % Nx != 0:
            K[i, i - 1] = DiffusionWeight[0] / h**2
        if (i + 1) % Nx != 0:
            K[i, i + 1] = DiffusionWeight[1] / h**2
        if i - Nx >= 0:
            K[i, i - Nx] = DiffusionWeight[3] / h**2
        if i + Nx < N:
            K[i, i + Nx] = DiffusionWeight[4] / h**2
        K[i, i] = DiffusionWeight[2] / h**2
    elif Boundary[i] == 1:
        if abs(X[i] - (a+h/2)) < h**3 and abs(Y[i] - (c+h/2)) > h**3 and abs(Y[i] - (d-h/2)) > h**3:
            K[i, [i+Nx-1,i+1,i,i-Nx,i+Nx]] = np.array([1, 1, -4, 1, 1]) / h**2
        if abs(X[i] - (a+h/2)) < h**3 and abs(Y[i] - (c+h/2)) < h**3:
            K[i, [i+Nx-1,i+1,i,i+(Ny-1)*Nx,i+Nx]] = np.array([1,1,-4,1,1]) / h**2
        if abs(X[i] - (a+h/2)) < h**3 and abs(Y[i] - (d-h/2)) < h**3:
            K[i, [i+Nx-1,i+1,i,i-(Ny-1)*Nx,i-Nx]] = np.array([1, 1, -4, 1, 1]) / h**2
        if abs(X[i] - (b-h/2)) < h**3 and abs(Y[i] - (c+h/2)) > h**3 and abs(Y[i] - (d-h/2)) > h**3:
            K[i, [i-1,i-Nx+1,i,i-Nx,i+Nx]] = np.array([1, 1, -4, 1, 1]) / h**2
        if abs(X[i] - (b-h/2)) < h**3 and abs(Y[i] - (c+h/2)) < h**3:
            K[i, [i-1,i-Nx+1,i,i+(Ny-1)*Nx,i+Nx]] = np.array([1,1,-4,1,1]) / h**2
        if abs(X[i] - (b-h/2)) < h**3 and abs(Y[i] - (d-h/2)) < h**3:
            K[i, [i-1,i-Nx+1,i,i-(Ny-1)*Nx,i-Nx]] = np.array([1, 1, -4, 1, 1]) / h**2
        if abs(Y[i] - (c+h/2)) < h**3 and abs(X[i] - (a+h/2)) > h**3 and abs(X[i] - (b-h/2)) > h**3:
            K[i, [i-1,i+1,i,i+(Ny-1)*Nx,i+Nx]] = np.array([1, 1, -4, 1, 1]) / h**2
        if abs(Y[i] - (d-h/2)) < h**3 and abs(X[i] - (a+h/2)) > h**3 and abs(X[i] - (b-h/2)) > h**3:
            K[i, [i-1,i+1,i,i-(Ny-1)*Nx,i-Nx]] = np.array([1, 1, -4, 1, 1]) / h**2
u = np.load('u.npy')

T = []
Mass = [np.sum(u) * h ** 2]
G = lil_matrix((2 * N, 2 * N))
G_upper_left = (A / dt ).tocsr()
G_upper_right = -K.tocsr()
G_lower_left = (epsl ** 2 * K).tocsr()
G_lower_right = A.tocsr()
G = vstack([
    hstack([G_upper_left, G_upper_right]),
    hstack([G_lower_left, G_lower_right])
])

G_csr = csr_matrix(G)
solve_G = factorized(G_csr)
time_values = []
u_values = []
energy_values = []
mass_values = []
energy_rate_values = []

def compute_energy(u, K, dx, dy, epsl):
    grad_u = -u @ K @ u
    F_u = 0.25 * (u ** 2 - 1) ** 2
    E = (0.5 * epsl ** 2 * grad_u + np.sum(F_u)) * dx * dy
    return E

def compute_mass(u,h):
    M = np.sum( u )*h**2
    return M

def crank_nicolson(u, N):
    u = u.flatten()
    F = np.zeros(2 * N)
    F[:N] = (u / dt ).flatten()
    F[N:] = (u ** 3 - u).flatten()
    U = solve_G(F)
    u_new = U[:N]
    return u_new

prev_energy = compute_energy(u, K, dx, dy, epsl)
window_size1 = 400
window_size2 = 5
nn_count = 0
valid_steps = 0
threshold = 0.5

for n in range(Ndt):
    t = n * dt
    T.append(t)
    current_energy = compute_energy(u, K, dx, dy, epsl)
    energy_rate = 0

    if n < window_size1:
        u_new = crank_nicolson(u, N)
    else:
        energy_rate = (current_energy - prev_energy) / dt
        prev_energy = current_energy

        if abs(energy_rate) < threshold:
            valid_steps += 1
        else:
            valid_steps = 0
        if valid_steps < 1:
            u_new = crank_nicolson(u, N)
        else:
            if nn_count >= 1:
                u_new = crank_nicolson(u, N)
                nn_count = 0
            else:
                input_data = np.array(u_values[-window_size2:])
                input_data = input_data.reshape((1, window_size2, N))
                input_data_tensor = torch.tensor(input_data, dtype=torch.float32)
                with torch.no_grad():
                    predictions = model(input_data_tensor)
                u_new = predictions.numpy().reshape(2500)
                nn_count += 1
    u = u_new
    current_energy = compute_energy(u, K, dx, dy, epsl)
    if n > 0:
        energy_rate = (prev_energy - current_energy) / dt
        energy_rate_values.append(energy_rate)
    prev_energy = current_energy
    current_mass = compute_mass(u, h)
    energy_values.append(current_energy)
    energy_rate_values.append(energy_rate)
    mass_values.append(current_mass)
    time_values.append(n * dt)
    u_values.append(u)
u_values = np.array(u_values)
print(u_values.shape)