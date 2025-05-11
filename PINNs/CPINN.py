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


# ===================== 1. 数据预处理 =====================
def load_data():
    data = scipy.io.loadmat('Data/NLSG_modify_3.mat')
    x = data['x']  # (256, 1)
    t = data['t']  # (100, 1)
    usol = data['usol3']  # (256, 100) complex

    X, T = np.meshgrid(x, t, indexing='ij')  # X, T: (256, 100)
    Exact_u = np.real(usol)
    Exact_v = np.imag(usol)

    # 全域输入
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # 初始条件 t=0
    N0 = 200
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    t0 = t.min() * np.ones((N0, 1))
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    # 边界条件 x=x_min 或 x=x_max
    N_b = 200
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
    N_f = 20000
    lb = X_star.min(0)
    ub_bound = X_star.max(0)
    X_f = lb + (ub_bound - lb) * lhs(2, N_f)

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
        'vb': torch.tensor(vb, dtype=torch.float32).to(device)
    }


# ===================== 2. 定义神经网络 =====================
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


# ===================== 3. 定义损失函数 =====================
def compute_loss(model, X_f, x0, t0, u0, v0, xb, tb, ub, vb, lambda_ic=100.0, lambda_bc=1.0, lambda_causal=5.0):
    # 初始条件损失
    u0_pred_real, u0_pred_imag = model(x0, t0)
    loss_ic = lambda_ic * (torch.mean((u0_pred_real - u0) ** 2) +
                           torch.mean((u0_pred_imag - v0) ** 2))

    # 边界条件损失
    ub_pred_real, ub_pred_imag = model(xb, tb)
    loss_bc = lambda_bc * (torch.mean((ub_pred_real - ub) ** 2) +
                           torch.mean((ub_pred_imag - vb) ** 2))

    # PDE残差损失（无监督）
    x_f = X_f[:, 0:1].clone().detach().requires_grad_(True)
    t_f = X_f[:, 1:2].clone().detach().requires_grad_(True)
    u_real, u_imag = model(x_f, t_f)

    u_real_t = torch.autograd.grad(u_real.sum(), t_f, create_graph=True)[0]
    u_real_x = torch.autograd.grad(u_real.sum(), x_f, create_graph=True)[0]
    u_real_xx = torch.autograd.grad(u_real_x.sum(), x_f, create_graph=True)[0]

    u_imag_t = torch.autograd.grad(u_imag.sum(), t_f, create_graph=True)[0]
    u_imag_x = torch.autograd.grad(u_imag.sum(), x_f, create_graph=True)[0]
    u_imag_xx = torch.autograd.grad(u_imag_x.sum(), x_f, create_graph=True)[0]

    # Schrödinger 方程残差项
    residual_real = -u_imag_t + 0.5 * u_real_xx + (u_real ** 2 + u_imag ** 2) * u_real
    residual_imag = u_real_t + 0.5 * u_imag_xx + (u_real ** 2 + u_imag ** 2) * u_imag

    causal_weights = torch.exp(-lambda_causal * (t_f - t_f.min()))
    loss_pde = torch.mean(causal_weights * (residual_real ** 2 + residual_imag ** 2))

    # 总损失
    total_loss = loss_ic + loss_bc + loss_pde
    return total_loss, loss_ic, loss_bc, loss_pde


def train_model():
    data = load_data()
    X_f = data['X_f']
    x0, t0, u0, v0 = data['x0'], data['t0'], data['u0'], data['v0']
    xb, tb, ub, vb = data['xb'], data['tb'], data['ub'], data['vb']

    # 定义模型
    layers = [2] + 8 * [100] + [2]
    model = NLS_PINN(layers).to(device)

    # 使用 Adam 优化器初始训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(40000):  # 你可以先跑少一些 epochs
        optimizer.zero_grad()
        total_loss, loss_ic, loss_bc, loss_pde = compute_loss(
            model, X_f, x0, t0, u0, v0, xb, tb, ub, vb)
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | Total Loss: {total_loss.item():.3e} | IC: {loss_ic.item():.3e} | BC: {loss_bc.item():.3e} | PDE: {loss_pde.item():.3e}")

    # 使用 L-BFGS 进一步优化
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),
                                        max_iter=50000,
                                        tolerance_grad=1e-9,
                                        tolerance_change=1e-9,
                                        history_size=50,
                                        line_search_fn="strong_wolfe")

    lbfgs_iter = [0]  # 使用 list 是为了在 closure 内部修改

    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss, loss_ic, loss_bc, loss_pde = compute_loss(
            model, X_f, x0, t0, u0, v0, xb, tb, ub, vb)
        total_loss.backward()
        if lbfgs_iter[0] % 10 == 0:
            print(
                f"[LBFGS] Iter {lbfgs_iter[0]} | Total Loss: {total_loss.item():.3e} | IC: {loss_ic.item():.3e} | BC: {loss_bc.item():.3e} | PDE: {loss_pde.item():.3e}")
        lbfgs_iter[0] += 1
        return total_loss

    print("Starting L-BFGS optimization...")
    optimizer_lbfgs.step(closure)

    return model


# ===================== 5. 训练与可视化 =====================

model = train_model()

data = scipy.io.loadmat('Data/NLSG_modify_3.mat')
x = data['x']  # (256, 1)
t = data['t']  # (100, 1)
usol = data['usol3']  # (256, 100) complex

X, T = np.meshgrid(x, t, indexing='ij')  # X, T: (256, 100)
Exact_u = np.real(usol)
Exact_v = np.imag(usol)

X_star = torch.tensor(np.hstack((X.flatten()[:, None], T.flatten()[:, None])), dtype=torch.float32).to(device)
# 预测结果
with torch.no_grad():
    u_real_pred, u_imag_pred = model(X_star[:, 0:1], X_star[:, 1:2])
    u_pred = u_real_pred.cpu().numpy() + 1j * u_imag_pred.cpu().numpy()
    u_pred = u_pred.reshape(len(x), len(t))

# 绘制结果
plt.plot()
plt.imshow(np.real(u_pred) ** 2 + np.imag(u_pred) ** 2, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto',
           cmap='jet')
plt.title("Predicted (Imaginary Part)")
plt.xlabel("x")
plt.ylabel("t")

plt.tight_layout()
plt.show()

scipy.io.savemat('Data/NLS_PINNs_Pred_RW.mat', {           # 坐标矩阵 (N, 2)
    'u_pred': u_pred,           # 预测解 (N, 2) [实部, 虚部]
 # L2误差历史 (M,)
})