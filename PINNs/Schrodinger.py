#!/usr/bin/env python
# coding: utf-8

# **Install deepxde**  
# Tensorflow and all other dependencies are already installed in Colab terminals

# **Problem setup**  
#   
# We are going to solve the non-linear Schrödinger equation given by  
# $i h_t + \frac{1}{2} h_{xx} + |h|^2h = 0$  
#   
# with periodic boundary conditions as  
# $x \in [-5,5], \quad t \in [0, \pi/2]$  
# $h(t, -5) = h(t,5)$  
# $h_x(t, -5) = h_x(t,5)$  
#   
# and initial condition equal to  
# $h(0,x) = 2 sech(x)$
# 
# 

# Deepxde only uses real numbers, so we need to explicitly split the real and imaginary parts of the complex PDE.  
#   
# In place of the single residual  
# $f = ih_t + \frac{1}{2} h_{xx} +|h|^2 h$  
#   
# we get the two (real valued) residuals  
# $f_{\mathcal{R}} = u_t + \frac{1}{2} v_{xx} + (u^2 + v^2)v$  
# $f_{\mathcal{I}} = v_t - \frac{1}{2} u_{xx} - (u^2 + v^2)u$  
#   
# where u(x,t) and v(x,t) denote respectively the real and the imaginary part of h.  
# 

# In[1]:


import numpy as np
import scipy.io
import deepxde as dde

# For plotting
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# In[2]:


data = scipy.io.loadmat("Data/NLS_PINN1.mat")

x = data["x"]  # shape: (256, 1)
t = data["t"].flatten()  # shape: (100,1)
u = data["usol1"]  # shape: (256, 100), complex-valued

# 若不是复数类型但以实部+虚部分开提供，请改为：
# u = data["u_real"] + 1j * data["u_imag"]

# 2. 提取实部和虚部
u_real = np.real(u)
u_imag = np.imag(u)

# 2. 提取初始条件 t=0

X_init = np.hstack((x, np.full_like(x, t[0])))  # (256, 2)
Y_init_u = u_real[:, 0:1]  # (256, 1)
Y_init_v = u_imag[:, 0:1]

# 3. 定义几何区域
space_domain = dde.geometry.Interval(x.min(), x.max())
time_domain = dde.geometry.TimeDomain(t[0], t[-1])
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)


def pde(x, y):
    """
    INPUTS:
        x: x[:,0] is x-coordinate
           x[:,1] is t-coordinate
        y: Network output, in this case:
            y[:,0] is u(x,t) the real part
            y[:,1] is v(x,t) the imaginary part
    OUTPUT:
        The pde in standard form i.e. something that must be zero
    """

    u = y[:, 0:1]
    v = y[:, 1:2]

    # In 'jacobian', i is the output component and j is the input component
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    v_t = dde.grad.jacobian(y, x, i=1, j=1)

    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)

    # In 'hessian', i and j are both input components. (The Hessian could be in principle something like d^2y/dxdt, d^2y/d^2x etc)
    # The output component is selected by "component"
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)

    f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
    f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

    return [f_u, f_v]


# Periodic Boundary conditions
bc_u_0 = dde.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0
)
bc_u_1 = dde.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0
)
bc_v_0 = dde.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1
)
bc_v_1 = dde.PeriodicBC(
    geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=1
)

# Initial conditions


ic_u = dde.PointSetBC(X_init, Y_init_u, component=0)
ic_v = dde.PointSetBC(X_init, Y_init_v, component=1)

# In[5]:


data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_u_0, bc_u_1, bc_v_0, bc_v_1, ic_u, ic_v],
    num_domain=10000,
    num_boundary=20,
    num_initial=200,
    train_distribution="pseudo",
)

# Network architecture
net = dde.nn.FNN([2] + [100] * 4 + [2], "tanh", "Glorot normal")

model = dde.Model(data, net)

# Adam optimization.

# In[6]:


# To employ a GPU accelerated system is highly encouraged.

model.compile("adam", lr=1e-3, loss="MSE")
model.train(iterations=10000, display_every=1000)

# L-BFGS optimization.

# In[8]:


dde.optimizers.config.set_LBFGS_options(
    maxcor=50,
    ftol=1.0 * np.finfo(float).eps,
    gtol=1e-08,
    maxiter=10000,
    maxfun=10000,
    maxls=50,
)
model.compile("L-BFGS")
model.train()

# Make prediction
X_star = np.hstack((x, np.tile(t, (x.shape[0], 1))))  # 创建输入网格
prediction = model.predict(X_star)

u_pred = prediction[:, 0]  # 实部预测
v_pred = prediction[:, 1]  # 虚部预测

from scipy.io import savemat

savemat('NLS_data.mat', {'u': u, 'v': v})

# 插值至完整网格
X, T = np.meshgrid(np.linspace(x.min(), x.max(), 256), np.linspace(t.min(), t.max(), 100))
u_grid = griddata(X_star, u_pred, (X, T), method="cubic")
v_grid = griddata(X_star, v_pred, (X, T), method="cubic")

# 计算幅度
h = np.sqrt(u_grid ** 2 + v_grid ** 2)

# 绘制预测结果
fig, ax = plt.subplots(3, 1, figsize=(10, 12))

ax[0].imshow(
    u_grid.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
ax[0].set_title("Real part (u)")
ax[0].set_xlabel("Time (t)")
ax[0].set_ylabel("Space (x)")
ax[0].colorbar(label="u")

ax[1].imshow(
    v_grid.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
ax[1].set_title("Imaginary part (v)")
ax[1].set_xlabel("Time (t)")
ax[1].set_ylabel("Space (x)")
ax[1].colorbar(label="v")

ax[2].imshow(
    h.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
ax[2].set_title("Amplitude (|u| + |v|)")
ax[2].set_xlabel("Time (t)")
ax[2].set_ylabel("Space (x)")
ax[2].colorbar(label="Amplitude")

plt.tight_layout()
plt.show()
