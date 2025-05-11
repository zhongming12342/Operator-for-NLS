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
import torch
import numpy as np
from scipy.stats import qmc
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation =  torch.nn.Tanh  # 改用Tanh激活函数

        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        return self.layers(x)


class PhysicsInformedNN():
    def __init__(self, X_u, u, X_f, layers, lb, ub, alpha, beta):
        # 边界条件
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)

        # 数据, 第一列对应于初始点的x和边界点的x
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
        self.u = torch.tensor(u, dtype=torch.float32).to(device)

        # 内部残差点
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)

        self.layers = layers
        self.alpha = alpha
        self.beta = beta

        # 神经网络
        self.dnn = DNN(layers).to(device)

        # 优化器
        lambda1 = lambda step: 1.0 if step < 1000 else \
            0.9 ** ((step - 1000) // 1000)

        self.optimizer_adam = torch.optim.Adam(self.dnn.parameters(), lr=0.001)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer_adam, lr_lambda=lambda1)

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
        self.l2_error_history = []
        self.use_adam = True

    def net_u(self, x, t):
        X = torch.cat([x, t], dim=1)
        out = self.dnn(X)
        psi_real = out[:, 0:1]
        psi_imag = out[:, 1:2]
        return psi_real, psi_imag

    def net_f(self, x, t):
        psi_real, psi_imag = self.net_u(x, t)

        # 计算一阶导数
        psi_real_t = torch.autograd.grad(psi_real.sum(), t, create_graph=True)[0]
        psi_real_x = torch.autograd.grad(psi_real.sum(), x, create_graph=True)[0]
        psi_imag_t = torch.autograd.grad(psi_imag.sum(), t, create_graph=True)[0]
        psi_imag_x = torch.autograd.grad(psi_imag.sum(), x, create_graph=True)[0]

        # 计算二阶导数
        psi_real_xx = torch.autograd.grad(psi_real_x.sum(), x, create_graph=True)[0]
        psi_imag_xx = torch.autograd.grad(psi_imag_x.sum(), x, create_graph=True)[0]

        # NLS方程
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

        x_t0 = self.x_u[0:N_initial]
        t_t0 = self.t_u[0:N_initial]
        u_t0 = self.u[0:N_initial]

        psi_real_t0, psi_imag_t0 = self.net_u(x_t0, t_t0)
        loss_t0 = torch.mean((u_t0[:, 0:1] - psi_real_t0) ** 2 + (u_t0[:, 1:2] - psi_imag_t0) ** 2)

        # 周期边界条件损失
        # 创建边界点并确保它们需要梯度
        t_lb = self.t_u[N_initial : N_initial + N_boundary//2].requires_grad_(True)
        t_rb = self.t_u[N_initial + N_boundary//2:].requires_grad_(True)
        x_lb = self.lb[0] * torch.ones_like(t_lb).requires_grad_(True)
        x_ub = self.ub[0] * torch.ones_like(t_rb).requires_grad_(True)

        # 计算边界上的解
        psi_real_lb, psi_imag_lb = self.net_u(x_lb, t_lb)
        psi_real_ub, psi_imag_ub = self.net_u(x_ub, t_rb)

        # 周期边界损失 - 解相同
        loss_periodic = torch.mean((psi_real_lb - psi_real_ub) ** 2 + (psi_imag_lb - psi_imag_ub) ** 2)

        # 计算边界上的导数
        psi_real_lb_x = torch.autograd.grad(psi_real_lb.sum(), x_lb, create_graph=True)[0]
        psi_imag_lb_x = torch.autograd.grad(psi_imag_lb.sum(), x_lb, create_graph=True)[0]
        psi_real_ub_x = torch.autograd.grad(psi_real_ub.sum(), x_ub, create_graph=True)[0]
        psi_imag_ub_x = torch.autograd.grad(psi_imag_ub.sum(), x_ub, create_graph=True)[0]

        # 周期边界导数损失 - 导数相同
        loss_periodic_deriv = torch.mean((psi_real_lb_x - psi_real_ub_x) ** 2 + (psi_imag_lb_x - psi_imag_ub_x) ** 2)

        # 物理约束损失
        f_real_pred, f_imag_pred = self.net_f(self.x_f, self.t_f)
        loss_f = torch.mean(f_real_pred ** 2 + f_imag_pred ** 2)

        # 总损失
        # 动态权重设计
        # w_t0 = 1.0 + 0.5 * torch.sigmoid(torch.tensor((self.iter - 2000) / 1000.))
        # w_p = 1.0 + 1.0 * torch.sigmoid(torch.tensor((self.iter - 1000) / 1000.))
        # w_pd = 0.5 + 2.0 * torch.sigmoid(torch.tensor((self.iter - 3000) / 1000.))
        # w_f = 1.0 + 0.5 * torch.sigmoid(torch.tensor((5000 - self.iter) / 2000.))

        w_t0 = 1.0
        w_p = 1.0
        w_pd = 1.0
        w_f = 1.0
        # 总损失
        loss = (
                w_t0 * loss_t0 +
                w_p * loss_periodic +
                w_pd * loss_periodic_deriv +
                w_f * loss_f
        )

        loss.backward()

        # 记录和打印
        if self.iter % 100 == 0:
            with torch.no_grad():
                x_test = torch.tensor(X_star[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
                t_test = torch.tensor(X_star[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)
                u_test = torch.tensor(u_star, dtype=torch.float32).to(device)

                psi_real_test, psi_imag_test = self.net_u(x_test, t_test)
                l2_error = torch.sqrt(torch.mean((psi_real_test - u_test[:, 0:1]) ** 2 +
                                                 (psi_imag_test - u_test[:, 1:2]) ** 2)) / \
                           torch.sqrt(torch.mean(u_test[:, 0:1] ** 2 + u_test[:, 1:2] ** 2))

                self.l2_error_history.append(l2_error.item())
                self.loss_history.append(loss.item())

                print(f'Iter {self.iter}, Loss: {loss.item():.5e}, Loss_t0: {loss_t0.item():.5e}, '
                      f'Loss_periodic: {loss_periodic.item():.5e}, Loss_deriv: {loss_periodic_deriv.item():.5e}, '
                      f'Loss_f: {loss_f.item():.5e}, L2 Error: {l2_error.item():.5e}')

        self.iter += 1
        return loss

    def train(self, n_iter):
        self.dnn.train()
        for it in range(n_iter):
            if self.use_adam:
                self.optimizer_adam.step(self.loss_func)
                # self.scheduler.step()
            else:
                def closure():
                    return self.loss_func()

                self.optimizer_lbfgs.step(closure)

            if self.iter > n_iter // 2 and self.use_adam:
                print("Switching to L-BFGS optimizer...")
                self.use_adam = False

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], dtype=torch.float32, requires_grad=True).to(device)
        t = torch.tensor(X[:, 1:2], dtype=torch.float32, requires_grad=True).to(device)

        self.dnn.eval()
        psi_real, psi_imag = self.net_u(x, t)
        f_real, f_imag = self.net_f(x, t)

        psi_real = psi_real.detach().cpu().numpy()
        psi_imag = psi_imag.detach().cpu().numpy()
        f_real = f_real.detach().cpu().numpy()
        f_imag = f_imag.detach().cpu().numpy()

        return np.hstack([psi_real, psi_imag]), np.hstack([f_real, f_imag])


def generate_adaptive_residual_points(lb, ub, N_f, X_u_train):
    # 基础拉丁超立方采样
    sampler = qmc.LatinHypercube(d=2)
    X_f = lb + (ub - lb) * sampler.random(int(N_f * 0.7))

    # 在边界附近增强采样
    boundary_width = 0.1 * (ub - lb)
    X_f_boundary = np.vstack([
        np.c_[lb[0] + boundary_width[0] * np.random.rand(int(N_f * 0.15)),
              np.random.uniform(lb[1], ub[1], int(N_f * 0.15))],
        np.c_[ub[0] - boundary_width[0] * np.random.rand(int(N_f * 0.15)),
              np.random.uniform(lb[1], ub[1], int(N_f * 0.15))]
    ])

    return np.vstack([X_f, X_f_boundary])
# 配置参数


alpha = 0.5  # NLS方程中的扩散系数
beta = 1.0  # NLS方程中的非线性系数
N_f = 10000  # 内部残差点数
layers = [2] + [100] * 6 + [2]  # 网络结构

# 加载数据
data = scipy.io.loadmat('Data/NLSG_modify_3.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact_real = np.real(data['usol3']).T
Exact_imag = np.imag(data['usol3']).T

X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = np.hstack((Exact_real.flatten()[:, None], Exact_imag.flatten()[:, None]))

# 定义域边界
lb = X_star.min(0)
ub = X_star.max(0)

# 生成训练数据
# 初始条件 (t=0)
N_initial = 200  #
xx_initial = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu_initial = np.hstack((Exact_real[0:1, :].T, Exact_imag[0:1, :].T))
idx_initial = np.random.choice(xx_initial.shape[0], N_initial, replace=False)
xx_initial = xx_initial[idx_initial]
uu_initial = uu_initial[idx_initial]

# 周期边界条件 (x=lb和x=ub)
tt_boundary = t
N_boundary = 400
# 设置采样数（左右各一半）
N_boundary_half = N_boundary // 2

# 左边界 (x=lb)
xx_left_all = np.hstack((lb[0] * np.ones_like(tt_boundary), tt_boundary))
uu_left_all = np.hstack([
    np.interp(tt_boundary.flatten(), T[:,0], Exact_real[:,0])[:, None],
    np.interp(tt_boundary.flatten(), T[:,0], Exact_imag[:,0])[:, None]
])

idx_left = np.random.choice(xx_left_all.shape[0], N_boundary_half, replace=False)
xx_left = xx_left_all[idx_left]
uu_left = uu_left_all[idx_left]

# 右边界 (x=ub)
xx_right_all = np.hstack((ub[0] * np.ones_like(tt_boundary), tt_boundary))
uu_right_all = np.hstack([
    np.interp(tt_boundary.flatten(), T[:,0], Exact_real[:,-1])[:, None],
    np.interp(tt_boundary.flatten(), T[:,0], Exact_imag[:,-1])[:, None]
])

idx_right = np.random.choice(xx_right_all.shape[0], N_boundary_half, replace=False)
xx_right = xx_right_all[idx_right]
uu_right = uu_right_all[idx_right]

# 合并边界点
X_bc = np.vstack([xx_left, xx_right])
u_bc = np.vstack([uu_left, uu_right])

# 最终训练数据：初始条件 + 边界条件
X_u_train = np.vstack([xx_initial, X_bc])
u_train = np.vstack([uu_initial, u_bc])



# 内部残差点 - 使用自适应采样
X_f_train = generate_adaptive_residual_points(lb, ub, N_f, X_u_train)

# 确保没有重叠点


# 创建模型
model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, alpha, beta)

# 训练模型
model.train(40000)

# 预测整个域的解
u_pred, f_pred = model.predict(X_star)

# 计算相对L2误差
error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
print('Relative L2 Error: %e' % error_u)

# 保存结果
# if not os.path.exists('results'):
#     os.makedirs('results')

# 保存预测解
scipy.io.savemat('Data/NLS_PINNs_Pred_RW.mat', {
    'X_star': X_star,           # 坐标矩阵 (N, 2)
    'u_pred': u_pred,           # 预测解 (N, 2) [实部, 虚部]
    'u_exact': u_star,          # 精确解 (N, 2) [实部, 虚部]
    'loss_history': np.array(model.loss_history),      # 损失历史 (M,)
    'l2_error_history': np.array(model.l2_error_history)  # L2误差历史 (M,)
})

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.semilogy(model.loss_history, label='Total Loss')
plt.semilogy(model.l2_error_history, label='L2 Error')
plt.xlabel('Iteration (x100)')
plt.ylabel('Value')
plt.title('Training History')
plt.legend()
plt.grid(True)
# plt.savefig('results/training_history.png')
plt.show()

# 可视化结果
U_pred = griddata(X_star, u_pred[:, 0], (X, T), method='cubic')  # 实部
V_pred = griddata(X_star, u_pred[:, 1], (X, T), method='cubic')  # 虚部

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
# plt.savefig('results/NLS_solution.png')
plt.show()