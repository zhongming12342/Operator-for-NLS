import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import scipy.io

# 设置随机种子（可复现）
torch.manual_seed(42)

# 设置设备（GPU 如果可用， 否则使用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data():
    data = scipy.io.loadmat('Data/NLSG_modify_2.mat')
    x = data['x']  # (256, 1)
    t = data['t']  # (100, 1)
    usol = data['usol2']  # (256, 100) complex

    X, T = np.meshgrid(x, t, indexing='ij')  # X, T 都是 (256, 100)

    # 拆解复数解为实部和虚部
    Exact_u = np.real(usol)  # (256, 100)
    Exact_v = np.imag(usol)  # (256, 100)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)  # (256, 100)

    # 整理输入输出
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # (25600, 2)
    u_star = Exact_u.flatten()[:, None]  # (25600, 1)
    v_star = Exact_v.flatten()[:, None]  # (25600, 1)
    h_star = Exact_h.flatten()[:, None]  # (25600, 1)

    # 初始条件 t=0
    N0 = 500
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    t0 = t.min() * np.ones((N0, 1))
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    # 边界条件 x=x_min 或 x=x_max
    N_b = 400
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    xb1 = x[0] * np.ones((N_b, 1))  # x = x_min
    ub1 = Exact_u[0, idx_t].reshape(-1, 1)
    vb1 = Exact_v[0, idx_t].reshape(-1, 1)

    xb2 = x[-1] * np.ones((N_b, 1))  # x = x_max
    ub2 = Exact_u[-1, idx_t].reshape(-1, 1)
    vb2 = Exact_v[-1, idx_t].reshape(-1, 1)

    xb = np.vstack([xb1, xb2])
    tb = np.vstack([tb, tb])
    ub = np.vstack([ub1, ub2])
    vb = np.vstack([vb1, vb2])

    # 残差点 X_f：拉丁超立方采样
    N_f = 5000
    lb = X_star.min(0)
    ub_bound = X_star.max(0)
    X_f = lb + (ub_bound - lb) * lhs(2, N_f)

    X0 = np.hstack((x0, t0))  # shape: (N0, 2)

    X_f = np.vstack((X_f, X0))  # shape: (N_f + N0, 2)

    # 转为 PyTorch tensor
    return {
        'X_f': torch.tensor(X_f, dtype=torch.float32).to(device),
        'x0': torch.tensor(x0, dtype=torch.float32).to(device),
        't0': torch.tensor(t0, dtype=torch.float32).to(device),
        'u0': torch.tensor(u0, dtype=torch.float32).to(device),
        'v0': torch.tensor(v0, dtype=torch.float32).to(device),
        'xb': torch.tensor(xb, dtype=torch.float32).to(device),
        'tb': torch.tensor(tb, dtype=torch.float32).to(device),
        'ub': torch.tensor(ub, dtype=torch.float32).to(device),
        'vb': torch.tensor(vb, dtype=torch.float32).to(device),
        'X_star': torch.tensor(X_star, dtype=torch.float32).to(device),
        'u_star': torch.tensor(u_star, dtype=torch.float32).to(device),
        'v_star': torch.tensor(v_star, dtype=torch.float32).to(device),
        'h_star': torch.tensor(h_star, dtype=torch.float32).to(device)
    }


class NLS_PINN(nn.Module):
    def __init__(self, layers):
        super(NLS_PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        outputs = self.layers[-1](inputs)
        u_real = outputs[:, 0:1]
        u_imag = outputs[:, 1:2]
        return u_real, u_imag


def compute_losses(model, X_f, u0, v0, xb, tb, ub, vb, lambda_ic, lambda_r, lambda_bc, epsilon):
    """
    计算各时间步的损失和因果权重
    参数:
        model: PINN模型
        t, x: 时间和空间坐标 (Tensor, shape=(N,1))
        u_true: 真实解 (Tensor, shape=(N,1))
        lambda_*: 各损失项的权重系数
        epsilon: 因果权重衰减系数
    返回:
        total_loss: 加权总损失
        losses: 各时间步损失列表 (用于可视化)
    """

    t = X_f[:, 1:2]
    x = X_f[:, 0:1]

    # 时间演化损失 (t>0)
    unique_t, inverse_indices = torch.unique(t, return_inverse=True, sorted=True)
    Nt = unique_t.shape[0]

    # 全部残差点启用梯度
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    # 前向传播
    u, v = model(x, t)

    # 梯度计算
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]

    modulus_squared = u ** 2 + v ** 2

    f_u = u_t + 0.5 * v_xx + modulus_squared * v
    f_v = v_t - 0.5 * u_xx - modulus_squared * u

    # 每个时间步的残差loss累积
    losses = torch.zeros(Nt, device=device)
    for i in range(1, Nt):  # 从 t=1 开始
        mask = (inverse_indices == i)
        losses[i] = torch.mean(f_u[mask] ** 2 + f_v[mask] ** 2)

    # 初始条件损失（t = min）
    t0 = t.min()
    mask_t0 = (t == t0)
    t0 = t[mask_t0].unsqueeze(-1)
    x0 = x[mask_t0].unsqueeze(-1)
    lic = lambda_ic * (torch.mean((u0 - model(x0, t0)[0]) ** 2) +
                       torch.mean((v0 - model(x0, t0)[1]) ** 2))
    losses[0] = lic

    # 因果权重
    weights = torch.ones_like(losses)
    for i in range(1, Nt):
        cumulative_loss = torch.sum(losses[0:i])
        weights[i] = torch.exp(-epsilon * cumulative_loss)

    # 边界损失
    u_left, v_left = model(xb[0:len(xb) // 2], tb[0:len(xb) // 2])
    u_right, v_right = model(xb[len(xb) // 2:], tb[len(xb) // 2:])
    loss_bc = lambda_bc * (torch.mean((u_left - u_right) ** 2) + torch.mean((v_left - v_right) ** 2))

    # 总损失
    total_loss = (losses[0] + torch.sum(weights[1:] * losses[1:])) / Nt + loss_bc

    return total_loss, losses, weights, loss_bc, losses[0]


def train_model():
    data = load_data()

    lambda_ic = 10.0  # 初始条件权重
    lambda_r = 1.0  # PDE残差权重
    lambda_bc = 1.0  # 边界条件权重
    epsilon = 0.5  # 因果权重衰减系数

    X_f, X_star = data['X_f'], data["X_star"]
    x0, t0, u0, v0 = data['x0'], data['t0'], data['u0'], data['v0']
    xb, tb, ub, vb = data['xb'], data['tb'], data['ub'], data['vb']
    u_star, v_star, h_star = data["u_star"], data["v_star"], data["h_star"],
    # 定义模型
    layers = [2] + 8 * [100] + [2]
    model = NLS_PINN(layers).to(device)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 50000

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, losses, weights, loss_bc, loss_ic = compute_losses(
            model, X_f, u0, v0, xb, tb, ub, vb, lambda_ic, lambda_r, lambda_bc, epsilon
        )
        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())
        if epoch % 10 == 0:
            u_pred, v_pred = model(X_star[:, 0:1], X_star[:, 1:2])
            pred = torch.complex(u_pred.squeeze(), v_pred.squeeze())  # (N,) 复数向量
            true = torch.complex(u_star.squeeze(), v_star.squeeze())
            # 相对 L2 误差：||pred - true|| / ||true||
            l2_error = torch.norm(pred - true, p=2) / torch.norm(true, p=2)
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4e},Loss_bc: {loss_bc.item():.4e},loss_ic: {loss_ic.item():.4e},Rel L2: {l2_error.item():.4e}")


if __name__ == "__main__":
    train_model()
