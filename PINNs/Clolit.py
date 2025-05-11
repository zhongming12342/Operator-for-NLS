# Data-driven inference of the Peregrine Soliton using Neural Networks (TF2 version)

import time
import scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pyDOE import lhs
import scipy.io
import numpy as np
from pyDOE import lhs  # 拉丁超立方采样
import tensorflow as tf
from scipy.optimize import minimize

# from tensorflow.python.eager import tape
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.run_functions_eagerly(True)
np.random.seed(1234)
tf.random.set_seed(1234)

# 配置 GPU 使用
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 限制只使用第 0 个 GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


class PINN(tf.keras.Model):
    def __init__(self, layers, lb, ub):
        super(PINN, self).__init__()
        self.lb = tf.convert_to_tensor(lb, dtype=tf.float32)
        self.ub = tf.convert_to_tensor(ub, dtype=tf.float32)
        self.hidden = [tf.keras.layers.Dense(l, activation=tf.nn.tanh) for l in layers[1:-1]]
        self.out = tf.keras.layers.Dense(layers[-1], activation=None)

    def call(self, X):
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for layer in self.hidden:
            H = layer(H)
        return self.out(H)


class PhysicsInformedNN:
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        self.lb = tf.constant(lb, dtype=tf.float32)
        self.ub = tf.constant(ub, dtype=tf.float32)

        self.x0 = tf.convert_to_tensor(x0, dtype=tf.float32)
        self.t0 = tf.zeros_like(self.x0)
        self.u0 = tf.convert_to_tensor(u0, dtype=tf.float32)
        self.v0 = tf.convert_to_tensor(v0, dtype=tf.float32)

        self.x_lb = tf.zeros_like(tb) + lb[0]
        self.t_lb = tf.convert_to_tensor(tb, dtype=tf.float32)
        self.x_ub = tf.zeros_like(tb) + ub[0]
        self.t_ub = tf.convert_to_tensor(tb, dtype=tf.float32)

        self.x_f = tf.convert_to_tensor(X_f[:, 0:1], dtype=tf.float32)
        self.t_f = tf.convert_to_tensor(X_f[:, 1:2], dtype=tf.float32)

        self.model = PINN(layers, lb, ub)

        # Adam优化器与学习率递减
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)  # Adam with learning rate decay

        self.error_history = []

    def net_uv(self, x, t):
        x = tf.cast(x, dtype=tf.float32)  # 强制转换为 float32
        t = tf.cast(t, dtype=tf.float32)  # 强制转换为 float32

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            X = tf.concat([x, t], axis=1)  # 现在 x 和 t 都是 float32
            uv = self.model(X)
            u = uv[:, 0:1]
            v = uv[:, 1:2]

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)
        del tape  # 释放持久化磁带

        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            u, v, u_x, v_x = self.net_uv(x, t)  # 确保 u_x/v_x 非 None
            u_t = tape.gradient(u, t)
            v_t = tape.gradient(v, t)

        # 计算二阶导数
        u_xx = tape.gradient(u_x, x)
        v_xx = tape.gradient(v_x, x)
        del tape  # 手动释放资源

        # NLS 方程残差
        f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u
        return f_u, f_v

    def loss_fn(self):
        u0_pred, v0_pred, _, _ = self.net_uv(self.x0, self.t0)
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.net_uv(self.x_lb, self.t_lb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.net_uv(self.x_ub, self.t_ub)
        f_u_pred, f_v_pred = self.net_f_uv(self.x_f, self.t_f)

        loss = tf.reduce_mean((self.u0 - u0_pred) ** 2) + tf.reduce_mean((self.v0 - v0_pred) ** 2)
        loss += tf.reduce_mean((u_lb_pred - u_ub_pred) ** 2) + tf.reduce_mean((v_lb_pred - v_ub_pred) ** 2)
        loss += tf.reduce_mean((u_x_lb_pred - u_x_ub_pred) ** 2) + tf.reduce_mean((v_x_lb_pred - v_x_ub_pred) ** 2)
        loss += tf.reduce_mean(f_u_pred ** 2) + tf.reduce_mean(f_v_pred ** 2)
        return loss

    def train(self, nIter, use_lbfgs=True):
        for it in range(nIter):
            # Adam 优化步骤
            with tf.GradientTape() as tape:
                loss = self.loss_fn()
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # 日志打印
            if it % 10 == 0:
                u_pred, v_pred, _, _ = self.predict(X_star)
                h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
                error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
                error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
                error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
                print(
                    f"Iter {it}, Loss: {loss.numpy():.3e}, Error u: {error_u:.3e}, Error v: {error_v:.3e} , Error u: {error_h:.3e}")

            # 条件判断修正：>= 代替 ==
            if use_lbfgs and it >= nIter // 2:  # 确保触发
                print(f"Switching to L-BFGS at iter {it}...")
                self.switch_to_lbfgs(X_star, u_star, v_star, h_star)
                break

    def switch_to_lbfgs(self, X_star, u_star, v_star, h_star):
        # 初始化参数（保持原有代码）
        init_params = np.concatenate([v.numpy().flatten() for v in self.model.trainable_variables])

        # 定义回调函数
        iteration = [0]  # 使用列表以在闭包中修改

        def callback(xk):
            iteration[0] += 1
            current_loss, _ = loss_and_grad(xk)  # 计算当前损失

            # 每10次迭代计算测试集误差
            if iteration[0] % 10 == 0 or iteration[0] == 1:
                # 预测测试集结果
                u_pred, v_pred, _, _ = self.predict(X_star)
                h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

                # 计算相对L2误差
                error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
                error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
                error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)

                print(
                    f"Iter {iteration[0]:3d} | "
                    f"Loss: {current_loss:.3e} | "
                    f"Error u: {error_u:.3e} | "
                    f"Error v: {error_v:.3e} | "
                    f"Error h: {error_h:.3e}"
                )

        # 其余优化代码保持不变...

        # 2. 定义损失和梯度计算函数（保持不变）
        def loss_and_grad(params):
            offset = 0
            for v in self.model.trainable_variables:
                shape = v.shape
                size = tf.reduce_prod(shape).numpy()
                param_tensor = tf.reshape(
                    tf.cast(params[offset:offset + size], v.dtype),
                    shape
                )
                v.assign(param_tensor)
                offset += size

            with tf.GradientTape() as tape:
                loss = self.loss_fn()
            grads = tape.gradient(loss, self.model.trainable_variables)
            grad_flat = np.concatenate([
                tf.cast(g, tf.float64).numpy().flatten()
                for g in grads
            ])
            return float(loss.numpy()), grad_flat

        # 3. 调用L-BFGS优化器
        try:
            results = minimize(
                loss_and_grad,
                init_params,
                jac=True,
                method='L-BFGS-B',
                callback=callback,  # 使用改进后的回调函数
                options={
                    'maxiter': 100,
                    'ftol': 1e-12,
                    'gtol': 1e-12,
                    'maxfun': 50000,
                    'maxls': 50
                }
            )

            # 4. 将优化后的参数赋回模型（处理 dtype 转换）
            offset = 0
            for v in self.model.trainable_variables:
                shape = v.shape
                size = tf.reduce_prod(shape).numpy()
                # 将 float64 的结果转为模型的实际 dtype
                v.assign(
                    tf.cast(
                        tf.reshape(results.x[offset:offset + size], shape),
                        v.dtype
                    )
                )
                offset += size

            print(f"\nL-BFGS 优化完成")
            print(f"最终损失: {results.fun:.3e}")
            print(f"迭代次数: {results.nit}")
            print(f"优化消息: {results.message}")

        except Exception as e:
            print(f"\nL-BFGS 优化失败: {str(e)}")
            raise  # 重新抛出异常以便调试

    def predict(self, X_star):
        X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
        uv = self.model(X_star)

        # 这里不调用 .numpy()，而是直接返回 Tensor
        u_star = uv[:, 0:1]  # 直接使用 Tensor
        v_star = uv[:, 1:2]

        # 计算 f_u_star 和 f_v_star，保持 Tensor
        f_u_star, f_v_star = self.net_f_uv(X_star[:, 0:1], X_star[:, 1:2])

        return u_star, v_star, f_u_star, f_v_star


# ========================================
# Main logic
if __name__ == "__main__":
    layers = [2, 100, 100, 100, 100, 2]

    # 加载 .mat 数据
    data = scipy.io.loadmat('Data/NLSG_modify.mat')
    x = data['x']  # (256, 1)
    t = data['t']  # (100, 1)
    usol = data['usol1']  # (256, 100), complex

    # 转置后 meshgrid：确保 X 为 256×100 对应 usol
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

    # 初始条件采样：选择 t=0
    N0 = 100  # 初始条件点数
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]  # (N0, 1)
    u0 = Exact_u[idx_x, 0:1]  # t=0 处的实部
    v0 = Exact_v[idx_x, 0:1]  # t=0 处的虚部

    # 边界条件采样（沿 x 边界选取多个 t）
    N_b = 80
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]  # (N_b, 1)

    # 网格边界（用于拉丁超立方）
    lb = X_star.min(0)  # [x_min, t_min]
    ub = X_star.max(0)  # [x_max, t_max]

    # 方程残差采样点：拉丁超立方采样
    N_f = 20000
    X_f = lb + (ub - lb) * lhs(2, N_f)  # (N_f, 2)

    # 最终结果：
    # x0, u0, v0 — 初始条件
    # tb — 时间边界条件点
    # X_f — 方程残差点

    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)
    start_time = time.time()
    # 先使用Adam优化一定次数，然后自动切换到L-BFGS
    model.train(nIter=10, use_lbfgs=True)
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.4f}s")

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    error_H = np.linalg.norm(Exact_h[100, :] - H_pred[100, :], 2) / np.linalg.norm(Exact_h[100, :], 2)

    print(f'Error u: {error_u:e}')
    print(f'Error v: {error_v:e}')
    print(f'Error h: {error_h:e}')
    print(f'Error at time step = 0: {error_H:e}')

    # plt.imshow(H_pred, interpolation='nearest', cmap='rainbow',
    #            extent=[lb[0], ub[0], lb[1], ub[1]], origin='lower', aspect='auto')
    # plt.title('|h(x,t)|')
    # plt.xlabel('$x$')
    # plt.ylabel('$t$')
    # plt.colorbar()
    # plt.savefig("predicted_peregrine_solition.png")
    # plt.show()
    #
    # plt.plot(x, Exact_h[100, :], 'b-', linewidth=2, label='Exact')
    # plt.plot(x, H_pred[100, :], 'r--', linewidth=2, label='Prediction')
    # plt.title('$t = 0$', fontsize=10)
    # plt.xlabel('$x$')
    # plt.ylabel('$|h(x,t)|$')
    # plt.legend(frameon=False)
    # plt.savefig("predicted_vs_exact_peregrine_time_step_0.png")
    # plt.show()
