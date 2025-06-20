import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, vstack, hstack
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

num_iterations = 500
num_samples_per_iteration = 40
total_samples = num_iterations * num_samples_per_iteration
data_u_values = np.zeros((total_samples, 5, N))
data_u_final = np.zeros((total_samples, N))

def compute_energy(u, K, dx, dy, epsl):
    grad_u = -u @ K @ u
    F_u = 0.25 * (u ** 2 - 1) ** 2
    E = (0.5 * epsl ** 2 * grad_u + np.sum(F_u)) * dx * dy
    return E

sample_index = 0  # To track the current sample index
for iteration in range(num_iterations):
    u = -0.3 + 0.05 * (2 * np.random.rand(N) - 1)
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
    energy_rate_values = []
    mass_values = []
    prev_energy = compute_energy(u, K, dx, dy, epsl)

    for n in range(Ndt):
        t = n * dt
        u = u.flatten()
        F = np.zeros(2 * N)
        F[:N] = (u / dt ).flatten()
        F[N:] = (u ** 3 - u).flatten()
        U = solve_G(F)
        u = U[:N]

        current_energy = compute_energy(u, K, dx, dy, epsl)
        energy_values.append(current_energy)
        time_values.append(n * dt)
        u_values.append(u)

        if n > 0:
            energy_rate = (prev_energy - current_energy) / dt
            energy_rate_values.append(energy_rate)
        prev_energy = current_energy

    u_values = np.array(u_values)
    energy_rate_values = np.array(energy_rate_values)

    selected_indices_high = []
    for i in range(len(energy_rate_values) - 5):
        if np.all(energy_rate_values[i:i + 6] > 0.5):
            selected_indices_high.append(i)

    selected_indices_low = []
    for i in range(len(energy_rate_values) - 5):
        if np.all(energy_rate_values[i:i + 6] < 0.5):
            selected_indices_low.append(i)

    num_selected_high = int(0.7 * num_samples_per_iteration)
    if selected_indices_high:
        num_selected_high = min(num_selected_high, len(selected_indices_high))
        random_selected_indices_high = np.random.choice(selected_indices_high, num_selected_high, replace=False)

        for index in random_selected_indices_high:
            if sample_index < total_samples:
                data_u_values[sample_index] = u_values[index:index + 5]
                data_u_final[sample_index] = u_values[index + 5]
                sample_index += 1

    num_selected_low = num_samples_per_iteration - num_selected_high
    if selected_indices_low:
        num_selected_low = min(num_selected_low, len(selected_indices_low))
        random_selected_indices_low = np.random.choice(selected_indices_low, num_selected_low, replace=False)

        for index in random_selected_indices_low:
            if sample_index < total_samples:
                data_u_values[sample_index] = u_values[index:index + 5]
                data_u_final[sample_index] = u_values[index + 5]
                sample_index += 1

    print(f"Iteration {iteration + 1}/{num_iterations} completed.")
np.savez('data.npz', data_u_values=data_u_values, data_u_final=data_u_final)
