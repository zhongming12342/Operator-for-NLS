import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import math

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Data generator
class DataGenerator(Dataset):
    def __init__(self, u, y, s, batch_size=64):
        'Initialization'
        self.u = torch.FloatTensor(u).to(device)  # input sample
        self.y = torch.FloatTensor(y).to(device)  # location
        self.s = torch.FloatTensor(s).to(device)  # labeled data evaluated at y

        self.N = u.shape[0]
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the total number of samples'
        return self.N

    def __getitem__(self, index):
        'Generate one batch of data'
        return (self.u[index], self.y[index]), self.s[index]


# Define the neural net
class MLP(nn.Module):
    def __init__(self, layers, activation=nn.Tanh()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            # Xavier initialization
            nn.init.xavier_normal_(self.layers[-1].weight, gain=1.0)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)  # No activation on last layer
        return x


# Define the model
class PI_DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super(PI_DeepONet, self).__init__()

        # Network initialization
        self.branch = MLP(branch_layers).to(device)
        self.trunk = MLP(trunk_layers).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)

        # Loggers
        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []
        self.itercount = 0

    # Define DeepONet architecture
    def operator_net(self, u, y):
        x = y[:, 0:1]
        t = y[:, 1:2]
        B = self.branch(u)
        T = self.trunk(torch.cat([x, t], dim=1))
        outputs = torch.sum(B * T, dim=1, keepdim=True)
        return outputs

    # Define residual calculation with autograd
    def residual_net(self, u, y):
        x = y[:, 0:1]
        t = y[:, 1:2]

        # Requires grad for autodiff
        x.requires_grad_(True)
        t.requires_grad_(True)
        y_with_grad = torch.cat([x, t], dim=1)

        s = self.operator_net(u, y_with_grad)

        # Compute gradients
        s_t = torch.autograd.grad(s.sum(), t, create_graph=True)[0]
        s_x = torch.autograd.grad(s.sum(), x, create_graph=True)[0]
        s_xx = torch.autograd.grad(s_x.sum(), x, create_graph=True)[0]

        res = s_t - 0.01 * s_xx - 0.01 * s ** 2
        return res

    # Define boundary loss
    def loss_bcs(self, batch):
        inputs, outputs = batch
        u, y = inputs
        s_pred = self.operator_net(u, y)
        loss = torch.mean((outputs - s_pred) ** 2)
        return loss

    # Define residual loss
    def loss_res(self, batch):
        inputs, outputs = batch
        u, y = inputs
        pred = self.residual_net(u, y)
        loss = torch.mean((outputs - pred) ** 2)
        return loss

    # Define total loss
    def loss(self, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(bcs_batch)
        loss_res = self.loss_res(res_batch)
        total_loss = loss_bcs + loss_res
        return total_loss, loss_bcs, loss_res

    # Training step
    def train_step(self, bcs_batch, res_batch):
        self.optimizer.zero_grad()
        total_loss, loss_bcs, loss_res = self.loss(bcs_batch, res_batch)
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Log losses
        if self.itercount % 100 == 0:
            self.loss_log.append(total_loss.item())
            self.loss_bcs_log.append(loss_bcs.item())
            self.loss_res_log.append(loss_res.item())

        self.itercount += 1
        return total_loss.item(), loss_bcs.item(), loss_res.item()

    # Train the model
    def train(self, bcs_loader, res_loader, nIter=10000):
        # self.train()

        # Create iterators that cycle indefinitely
        bcs_iter = iter(bcs_loader)
        res_iter = iter(res_loader)

        pbar = tqdm(range(nIter))
        for it in pbar:
            try:
                bcs_batch = next(bcs_iter)
            except StopIteration:
                bcs_iter = iter(bcs_loader)
                bcs_batch = next(bcs_iter)

            try:
                res_batch = next(res_iter)
            except StopIteration:
                res_iter = iter(res_loader)
                res_batch = next(res_iter)

            total_loss, loss_bcs, loss_res = self.train_step(bcs_batch, res_batch)

            if it % 100 == 0:
                pbar.set_postfix({'Loss': total_loss,
                                  'loss_bcs': loss_bcs,
                                  'loss_physics': loss_res})

    # Evaluates predictions at test points
    def predict_s(self, U_star, Y_star):
        # self.eval()
        with torch.no_grad():
            s_pred = self.operator_net(U_star, Y_star)
        return s_pred.cpu().numpy()

    def predict_res(self, U_star, Y_star):
        # self.eval()
        with torch.no_grad():
            r_pred = self.residual_net(U_star, Y_star)
        return r_pred.cpu().numpy()


# RBF kernel for GP sampling
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


# A diffusion-reaction numerical solver
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
    gp_sample = np.dot(L, np.random.randn(N))
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
def generate_one_training_data(key, P, Q):
    # Numerical solution
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx, Nt, P, length_scale)

    # Sample points from the boundary and the initial conditions
    x_bc1 = np.zeros((P // 3, 1))
    x_bc2 = np.ones((P // 3, 1))
    x_bc3 = np.random.uniform(size=(P // 3, 1))
    x_bcs = np.vstack((x_bc1, x_bc2, x_bc3))

    t_bc1 = np.random.uniform(size=(P // 3 * 2, 1))
    t_bc2 = np.zeros((P // 3, 1))
    t_bcs = np.vstack([t_bc1, t_bc2])

    # Training data for BC and IC
    u_train = np.tile(u, (P, 1))
    y_train = np.hstack([x_bcs, t_bcs])
    s_train = np.zeros((P, 1))

    # Sample collocation points
    x_r_idx = np.random.choice(np.arange(Nx), size=(Q, 1))
    x_r = x[x_r_idx]
    t_r = np.random.uniform(size=(Q, 1))

    # Training data for the PDE residual
    u_r_train = np.tile(u, (Q, 1))
    y_r_train = np.hstack([x_r, t_r])
    s_r_train = u[x_r_idx]

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train


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
def generate_training_data(key, N, P, Q):
    u_train = []
    y_train = []
    s_train = []
    u_r_train = []
    y_r_train = []
    s_r_train = []

    for i in range(N):
        u_tr, y_tr, s_tr, u_r_tr, y_r_tr, s_r_tr = generate_one_training_data(key + i, P, Q)
        u_train.append(u_tr)
        y_train.append(y_tr)
        s_train.append(s_tr)
        u_r_train.append(u_r_tr)
        y_r_train.append(y_r_tr)
        s_r_train.append(s_r_tr)

    u_train = np.float32(np.vstack(u_train))
    y_train = np.float32(np.vstack(y_train))
    s_train = np.float32(np.vstack(s_train))

    u_r_train = np.float32(np.vstack(u_r_train))
    y_r_train = np.float32(np.vstack(y_r_train))
    s_r_train = np.float32(np.vstack(s_r_train))

    return u_train, y_train, s_train, u_r_train, y_r_train, s_r_train


# Generate test data corresponding to N input samples
def generate_test_data(key, N, P):
    u_test = []
    y_test = []
    s_test = []

    for i in range(N):
        u_te, y_te, s_te = generate_one_test_data(key + i, P)
        u_test.append(u_te)
        y_test.append(y_te)
        s_test.append(s_te)

    u_test = np.float32(np.vstack(u_test))
    y_test = np.float32(np.vstack(y_test))
    s_test = np.float32(np.vstack(s_test))

    return u_test, y_test, s_test


# Compute relative l2 error over N test samples
def compute_error(model, key, P):
    # Generate one test sample
    u_test, y_test, s_test = generate_test_data(key, 1, P)

    # Convert to torch tensors
    u_test_t = torch.FloatTensor(u_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    s_test_t = torch.FloatTensor(s_test).to(device)

    # Predict
    s_pred = model.predict_s(u_test_t, y_test_t)

    s_pred = np.transpose(s_pred)
    # Compute relative l2 error
    error_s = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
    return error_s


# Main parameters
key = 0
length_scale = 0.2
Nx = 100
Nt = 100
m = Nx

# Training data
N = 10  # number of input samples
P_train = 300  # number of output sensors
Q_train = 100  # number of collocation points for each input sample

u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train = generate_training_data(key, N, P_train,
                                                                                                      Q_train)

# Initialize model
branch_layers = [m, 50, 50, 50, 50, 50]
trunk_layers = [2, 50, 50, 50, 50, 50]
model = PI_DeepONet(branch_layers, trunk_layers).to(device)

# Create data loaders
batch_size = 100
bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

bcs_loader = DataLoader(bcs_dataset, batch_size=batch_size, shuffle=True)
res_loader = DataLoader(res_dataset, batch_size=batch_size, shuffle=True)

# Train
model.train(bcs_loader, res_loader, nIter=12)

# Test data
N_test = 100  # number of input samples
P_test = m  # number of sensors
key_test = 1234567

# Compute error
errors = []
for i in range(N_test):
    errors.append(compute_error(model, key_test + i, P_test))

error_s = np.array(errors)
print('Mean of relative L2 error of s: {:.2e}'.format(error_s.mean()))
print('Std of relative L2 error of s: {:.2e}'.format(error_s.std()))

# Plot loss function
plt.figure(figsize=(6, 5))
plt.plot(model.loss_bcs_log, lw=2, label='bcs')
plt.plot(model.loss_res_log, lw=2, label='res')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

# Generate one test sample
key = 12345
P_test = 100
u_test, y_test, s_test = generate_test_data(key, 1, P_test)

# Predict
u_test_t = torch.FloatTensor(u_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)
s_pred = model.predict_s(u_test_t, y_test_t)

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