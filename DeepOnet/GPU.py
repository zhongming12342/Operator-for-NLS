#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import math
import itertools
import warnings

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(1234)
np.random.seed(1234)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(1234)


# Data generator
class DataGenerator(Dataset):
    def __init__(self, u, y, s, batch_size):
        'Initialization'
        self.u = torch.tensor(u, dtype=torch.float32).to(device)  # input sample
        self.y = torch.tensor(y, dtype=torch.float32).to(device)  # location
        self.s = torch.tensor(s, dtype=torch.float32).to(device)  # labeled data evaluated at y

        self.N = u.shape[0]
        self.batch_size = batch_size

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        """返回单个样本"""
        return (self.u[index], self.y[index]), self.s[index]


# Define the neural net
class MLP(nn.Module):
    def __init__(self, layers, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            # 正确的初始化方式
            nn.init.xavier_normal_(layer.weight)  # 初始化权重
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)  # 初始化偏置，注意是bias不是basis
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        # No activation for the last layer
        x = self.layers[-1](x)
        return x


# Define the model
class DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super(DeepONet, self).__init__()
        self.branch = MLP(branch_layers, activation=nn.Tanh())
        self.trunk = MLP(trunk_layers, activation=nn.Tanh())

        # 将模型移动到GPU
        self.to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)
        self.loss_log = []
        self.itercount = itertools.count()

    def _initialize_weights(self):
        # Already handled in MLP class with Xavier init
        pass

    # Define DeepONet architecture
    def operator_net(self, u, x, t):
        y = torch.stack([x, t], dim=-1)
        B = self.branch(u)
        T = self.trunk(y)
        outputs = torch.sum(B * T, dim=-1)
        return outputs

    # Define operator loss
    def loss_operator(self, batch):
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        s_pred = self.operator_net(u, y[:, 0], y[:, 1])
        # Compute loss
        loss = torch.mean((outputs.flatten() - s_pred.flatten()) ** 2)
        return loss

    # Training step
    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self.loss_operator(batch)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    # Optimize parameters in a loop
    def train(self, dataloader, val_dataloader, num_epochs):
        with tqdm(range(num_epochs), desc="Epochs") as pbar_epoch:
            for epoch in pbar_epoch:
                # 训练阶段（带批次进度条）
                # self.model.train()
                train_loss = 0.0
                for batch in dataloader:
                    self.optimizer.zero_grad()
                    loss = self.loss_operator(batch)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    # batch_pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

                # 验证阶段（无进度条）
                # self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_dataloader:
                        val_loss += self.loss_operator(batch).item()

                # 更新主进度条
                pbar_epoch.set_postfix({
                    'train_loss': f"{train_loss / len(dataloader):.4f}",
                    'val_loss': f"{val_loss / len(val_dataloader):.4f}"
                })

    # Evaluates predictions at test points
    def predict_s(self, U_star, Y_star):
        with torch.no_grad():
            U_star = torch.tensor(U_star, dtype=torch.float32).to(device)
            Y_star = torch.tensor(Y_star, dtype=torch.float32).to(device)
            s_pred = self.operator_net(U_star, Y_star[:, 0], Y_star[:, 1])
        return s_pred.cpu().numpy()


# RBF kernel function
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


# A diffusion-reaction numerical solver (using numpy as before)
def solve_ADR(key, Nx, Nt, P, length_scale):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    np.random.seed(key)
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    k = lambda x: 0.01 * np.ones_like(x)
    v = lambda x: np.zeros_like(x)
    g = lambda u: 0.01 * u ** 2
    dg = lambda u: 0.02 * u
    u0 = lambda x: np.zeros_like(x)

    # Generate a GP sample
    N = 512
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    gp_sample = np.dot(L, np.random.normal(size=(N,)))
    # Create a callable interpolation function
    f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute coefficients and forcing
    k = k(x)
    v = v(x)
    f = f_fn(x)

    # Compute finite difference operators
    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    # Time-stepping update
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)

    # Input sensor locations and measurements
    xx = np.linspace(xmin, xmax, m)
    u_input = f_fn(xx)
    # Output sensor locations and measurements
    idx = np.random.randint(0, max(Nx, Nt), size=(P, 2))
    y = np.concatenate([x[idx[:, 0]][:, None], t[idx[:, 1]][:, None]], axis=1)
    s = u[idx[:, 0], idx[:, 1]]
    return (x, t, u), (u_input, y, s)


# Generate training data corresponding to one input sample
def generate_one_training_data(key, P):
    # Numerical solution
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx, Nt, P, length_scale)
    u = u.flatten()

    # 平铺u使其与y的样本数匹配
    u_tiled = np.tile(u, (P, 1))  # 形状 (P, m)

    # 确保s是列向量 (P, 1)
    s = s.reshape(-1, 1)

    return u_tiled, y, s


# Generate test data corresponding to one input sample
def generate_one_test_data(key, P):
    Nx = P
    Nt = P
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx, Nt, P, length_scale)
    XX, TT = np.meshgrid(x, t)
    u_test = np.tile(u, (P ** 2, 1))
    y_test = np.hstack([XX.flatten()[:, None], TT.flatten()[:, None]])
    s_test = UU.T.flatten()
    return u_test, y_test, s_test


# Generate training data corresponding to N input samples
def generate_training_data(key, N, P):
    u_train, y_train, s_train = [], [], []
    for i in range(N):
        u, y, s = generate_one_training_data(key + i, P)
        u_train.append(u)
        y_train.append(y)
        s_train.append(s)

    u_train = np.float32(np.vstack(u_train))
    y_train = np.float32(np.vstack(y_train))
    s_train = np.float32(np.vstack(s_train))
    return u_train, y_train, s_train


# Generate test data corresponding to N input samples
def generate_test_data(key, N, P):
    u_test, y_test, s_test = [], [], []
    for i in range(N):
        u, y, s = generate_one_test_data(key + i, P)
        u_test.append(u)
        y_test.append(y)
        s_test.append(s)

    u_test = torch.tensor(np.array(u_test))
    y_test = torch.tensor(np.array(y_test))
    s_test = torch.tensor(np.array(s_test))
    u_test = np.float32(u_test.reshape(N * P ** 2, -1))
    y_test = np.float32(y_test.reshape(N * P ** 2, -1))
    s_test = np.float32(s_test.reshape(N * P ** 2, -1))
    return u_test, y_test, s_test


# Compute relative l2 error over N test samples
def compute_error(model, key, P):
    # Generate one test sample
    u_test, y_test, s_test = generate_test_data(key, 1, P)
    # Predict
    error_s = model.predict_s(u_test, y_test)[:, None]
    # Compute relative l2 error
    error_s = np.linalg.norm(s_test - error_s) / np.linalg.norm(s_test)
    return error_s


# Main execution
if __name__ == "__main__":
    key = 0

    # GRF length scale
    length_scale = 0.2

    # Resolution of the solution
    Nx = 100
    Nt = 100
    m = Nx  # number of input sensors

    N = 50  # number of input samples
    P_train = 100  # number of output sensors

    u_train, y_train, s_train = generate_training_data(key, N, P_train)

    # Initialize model
    branch_layers = [m, 50, 50, 50, 50, 50]
    trunk_layers = [2, 50, 50, 50, 50, 50]
    model = DeepONet(branch_layers, trunk_layers)

    # Create data loader
    batch_size = 100
    dataset = DataGenerator(u_train, y_train, s_train, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Test data
    N_test = 1  # number of input samples
    P_test = 100  # number of sensors
    key_test = 1234567

    u_test, y_test, s_test = generate_test_data(key, N_test, P_test)
    val_dataset = DataGenerator(u_test, y_test, s_test, 1)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Train
    model.train(dataloader, val_dataloader, num_epochs=500)

    # Compute error
    errors = []
    for i in range(N_test):
        errors.append(compute_error(model, key_test + i, P_test))
    error_s = np.array(errors)

    print('mean of relative L2 error of s: {:.2e}'.format(error_s.mean()))
    print('std of relative L2 error of s: {:.2e}'.format(error_s.std()))

    # Plot loss function
    plt.figure(figsize=(6, 5))
    plt.plot(model.loss_log, lw=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # Generate one test sample

    # Predict
    s_pred = model.predict_s(u_test, y_test)

    # Generate an uniform mesh
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    XX, TT = np.meshgrid(x, t)

    # Grid data
    S_pred = griddata(y_test, s_pred.flatten(), (XX, TT), method='cubic')
    S_test = griddata(y_test, s_test.flatten(), (XX, TT), method='cubic')

    # Compute the relative l2 error
    error = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
    print('Relative l2 error: {:.3e}'.format(error))

    # Plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(XX, TT, S_test, cmap='seismic')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Exact $s(x,t)$')
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(XX, TT, S_pred, cmap='seismic')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Predict $s(x,t)$')
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(XX, TT, S_pred - S_test, cmap='seismic')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('Absolute error')
    plt.tight_layout()
    plt.show()
