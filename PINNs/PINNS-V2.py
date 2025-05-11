import torch
import numpy as np
import scipy.io
from collections import OrderedDict
from pyDOE import lhs
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义DNN模型
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        return self.layers(x)


# 定义物理信息神经网络（NLS版本）
class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f, layers, lb, ub, alpha, beta):
        # 边界条件
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # 数据
        # Split the training data into initial condition and boundary conditions
        # Initial condition (t=0)
        idx_init = np.where(X_u[:, 1] == lb[1])[0]
        self.x_init = torch.tensor(X_u[idx_init, 0:1], requires_grad=True).float().to(device)
        self.t_init = torch.tensor(X_u[idx_init, 1:2], requires_grad=True).float().to(device)
        self.u_init = torch.tensor(u[idx_init, :]).float().to(device)

        # Boundary conditions (x=lb and x=ub)
        idx_bc = np.where(X_u[:, 1] != lb[1])[0]
        self.x_bc = torch.tensor(X_u[idx_bc, 0:1], requires_grad=True).float().to(device)
        self.t_bc = torch.tensor(X_u[idx_bc, 1:2], requires_grad=True).float().to(device)
        self.u_bc = torch.tensor(u[idx_bc, :]).float().to(device)

        # Collocation points
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)

        self.layers = layers
        self.alpha = alpha  # NLS方程中的参数
        self.beta = beta  # NLS方程中的参数

        # 神经网络
        self.dnn = DNN(layers).to(device)

        # 优化器
        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        self.optimizer_lbfgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        self.iter = 0
        self.loss_history = []
        self.loss_init_history = []
        self.loss_bc_history = []
        self.loss_f_history = []
        self.l2_error_history = []
        self.use_adam = True

    def net_u(self, x, t):
        X = torch.cat([x, t], dim=1)
        out = self.dnn(X)
        psi_real = out[:, 0:1]
        psi_imag = out[:, 1:2]
        return psi_real, psi_imag

    def net_f(self, x, t):
        """ 计算NLS方程的残差 """
        psi_real, psi_imag = self.net_u(x, t)

        # 计算一阶导数
        psi_real_t = torch.autograd.grad(psi_real, t, grad_outputs=torch.ones_like(psi_real),
                                         retain_graph=True, create_graph=True)[0]
        psi_real_x = torch.autograd.grad(psi_real, x, grad_outputs=torch.ones_like(psi_real),
                                         retain_graph=True, create_graph=True)[0]
        psi_imag_t = torch.autograd.grad(psi_imag, t, grad_outputs=torch.ones_like(psi_imag),
                                         retain_graph=True, create_graph=True)[0]
        psi_imag_x = torch.autograd.grad(psi_imag, x, grad_outputs=torch.ones_like(psi_imag),
                                         retain_graph=True, create_graph=True)[0]

        # 计算二阶导数
        psi_real_xx = torch.autograd.grad(psi_real_x, x, grad_outputs=torch.ones_like(psi_real_x),
                                          retain_graph=True, create_graph=True)[0]
        psi_imag_xx = torch.autograd.grad(psi_imag_x, x, grad_outputs=torch.ones_like(psi_imag_x),
                                          retain_graph=True, create_graph=True)[0]

        # NLS方程: iψ_t + αψ_xx + β|ψ|²ψ = 0
        modulus_squared = psi_real ** 2 + psi_imag ** 2
        f_real = -psi_imag_t + self.alpha * psi_real_xx + self.beta * modulus_squared * psi_real
        f_imag = psi_real_t + self.alpha * psi_imag_xx + self.beta * modulus_squared * psi_imag

        return f_real, f_imag

    def loss_func(self):
        if self.use_adam:
            self.optimizer_adam.zero_grad()
        else:
            self.optimizer_lbfgs.zero_grad()

        # 初始条件损失 (t=0)
        psi_real_init, psi_imag_init = self.net_u(self.x_init, self.t_init)
        loss_init = torch.mean((self.u_init[:, 0:1] - psi_real_init) ** 2 +
                               (self.u_init[:, 1:2] - psi_imag_init) ** 2)

        # 周期边界条件损失 (x=lb and x=ub)
        psi_real_bc, psi_imag_bc = self.net_u(self.x_bc, self.t_bc)

        # 计算边界上的解在x=lb和x=ub处的差值
        # 首先需要分离出x=lb和x=ub的点
        idx_lb = torch.where(self.x_bc == self.lb[0])[0]
        idx_ub = torch.where(self.x_bc == self.ub[0])[0]

        # 确保我们有匹配的边界点对
        n_pairs = min(len(idx_lb), len(idx_ub))
        idx_lb = idx_lb[:n_pairs]
        idx_ub = idx_ub[:n_pairs]

        # 计算周期边界差异
        loss_bc = torch.mean((psi_real_bc[idx_lb] - psi_real_bc[idx_ub]) ** 2 +
                             (psi_imag_bc[idx_lb] - psi_imag_bc[idx_ub]) ** 2)

        # 物理约束损失
        f_real_pred, f_imag_pred = self.net_f(self.x_f, self.t_f)
        loss_f = torch.mean(f_real_pred ** 2 + f_imag_pred ** 2)

        # 总损失
        loss = loss_init + loss_bc + loss_f

        # 记录各项损失
        self.loss_history.append(loss.item())
        self.loss_init_history.append(loss_init.item())
        self.loss_bc_history.append(loss_bc.item())
        self.loss_f_history.append(loss_f.item())

        loss.backward()

        # 计算L2误差
        if self.iter % 100 == 0:
            with torch.no_grad():
                x_test = torch.tensor(X_star[:, 0:1], requires_grad=True).float().to(device)
                t_test = torch.tensor(X_star[:, 1:2], requires_grad=True).float().to(device)
                u_test = torch.tensor(u_star).float().to(device)

                psi_real_test, psi_imag_test = self.net_u(x_test, t_test)
                l2_error = torch.sqrt(torch.mean((psi_real_test - u_test[:, 0:1]) ** 2 +
                                                 (psi_imag_test - u_test[:, 1:2]) ** 2)) / \
                           torch.sqrt(torch.mean(u_test[:, 0:1] ** 2 + u_test[:, 1:2] ** 2))
                self.l2_error_history.append(l2_error.item())

                print(f'Iter {self.iter}, Loss: {loss.item():.5e}, '
                      f'Loss_init: {loss_init.item():.5e}, '
                      f'Loss_bc: {loss_bc.item():.5e}, '
                      f'Loss_f: {loss_f.item():.5e}, '
                      f'L2 Error: {l2_error.item():.5e}')

        self.iter += 1
        return loss

    def train(self, n_iter):
        self.dnn.train()
        for it in range(n_iter):
            if self.use_adam:
                self.optimizer_adam.step(self.loss_func)
            else:
                self.optimizer_lbfgs.step(self.loss_func)

            # 在一定的训练迭代次数后切换到L-BFGS优化器
            if self.iter > 10000 and self.use_adam:
                print("Switching to L-BFGS optimizer...")
                self.use_adam = False

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        psi_real, psi_imag = self.net_u(x, t)
        f_real, f_imag = self.net_f(x, t)

        psi_real = psi_real.detach().cpu().numpy()
        psi_imag = psi_imag.detach().cpu().numpy()
        f_real = f_real.detach().cpu().numpy()
        f_imag = f_imag.detach().cpu().numpy()

        return np.hstack([psi_real, psi_imag]), np.hstack([f_real, f_imag])


# 配置参数
alpha = 0.5
beta = 1.0
noise = 0.0

N_u = 600  # 边界数据点数
N_f = 10000  # 内部残差点数
layers = [2] + [100] * 4 + [2]

# 加载数据
data = scipy.io.loadmat('Data/NLSG_modify.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact_real = np.real(data['usol1']).T
Exact_imag = np.imag(data['usol1']).T

X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = np.hstack((Exact_real.flatten()[:, None], Exact_imag.flatten()[:, None]))

# 定义域边界
lb = X_star.min(0)
ub = X_star.max(0)

# 生成训练数据
# 初始条件 (t=0)
xx_init = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu_init = np.hstack((Exact_real[0:1, :].T, Exact_imag[0:1, :].T))

# 边界条件 (x=lb and x=ub)
xx_bc_lb = np.hstack((X[:, 0:1], T[:, 0:1]))  # x=lb
uu_bc_lb = np.hstack((Exact_real[:, 0:1], Exact_imag[:, 0:1]))
xx_bc_ub = np.hstack((X[:, -1:], T[:, -1:]))  # x=ub
uu_bc_ub = np.hstack((Exact_real[:, -1:], Exact_imag[:, -1:]))

# 合并边界条件数据
X_bc = np.vstack([xx_bc_lb, xx_bc_ub])
u_bc = np.vstack([uu_bc_lb, uu_bc_ub])

# 合并所有训练数据
X_u_train = np.vstack([xx_init, X_bc])
u_train = np.vstack([uu_init, u_bc])

# 内部残差点
X_f_train = lb + (ub - lb) * lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))

# 随机选择训练点
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx, :]

# 创建模型
model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, alpha, beta)

# 训练模型
model.train(50000)

# 预测整个域的解
u_pred, f_pred = model.predict(X_star)

# 计算相对L2误差
error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Relative L2 Error: %e' % error_u)

# 保存结果
scipy.io.savemat('results/NLS_PINNs1.mat', {
    'X_star': X_star,
    'u_pred': u_pred,
    'u_exact': u_star,
    'loss_history': np.array(model.loss_history),
    'loss_init_history': np.array(model.loss_init_history),
    'loss_bc_history': np.array(model.loss_bc_history),
    'loss_f_history': np.array(model.loss_f_history),
    'l2_error_history': np.array(model.l2_error_history)
})

# 绘制损失曲线
plt.figure(figsize=(12, 8))
plt.semilogy(model.loss_history, label='Total Loss')
plt.semilogy(model.loss_init_history, label='Initial Condition Loss')
plt.semilogy(model.loss_bc_history, label='Boundary Condition Loss')
plt.semilogy(model.loss_f_history, label='PDE Loss')
plt.semilogy(model.l2_error_history, label='L2 Error')
plt.xlabel('Iteration (x100)')
plt.ylabel('Value')
plt.title('Training History')
plt.legend()
plt.grid(True)
plt.show()

# 可视化结果
U_pred = griddata(X_star, u_pred[:, 0], (X, T), method='cubic')
V_pred = griddata(X_star, u_pred[:, 1], (X, T), method='cubic')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.pcolor(X, T, U_pred, cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted Real Part')

plt.subplot(1, 2, 2)
plt.pcolor(X, T, V_pred, cmap='jet')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted Imaginary Part')
plt.show()