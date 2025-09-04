import numpy as np
import torch
import matplotlib.pyplot as plt
import lightning as L
import os
from anderson import anderson_accel
from fp_solvers import extrap_safe, picard, heun2, extrap


class DiffusionBridge(L.LightningModule):
    def __init__(
            self,
            n_steps,
            gamma,
            beta_start,
            beta_end,
            n_recursions,
            consistency_threshold
        ):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.beta_start = beta_start
        self.beta_end = beta_end / n_steps
        self.n_recursions = n_recursions
        self.consistency_threshold = consistency_threshold

        # Define betas
        self.betas = self._get_betas()
        
        # Mean schedule
        s = np.cumsum(self.betas)**0.5
        s_bar = np.flip(np.cumsum(self.betas))**0.5
        mu_x0, mu_y, _ = self.gaussian_product(s, s_bar)

        # Scale gamma for number of diffusion steps
        gamma = gamma * self.betas.sum()
        
        # Noise schedule
        std = gamma * s / (s**2 + s_bar**2)

        # Convert to tensors
        self.register_buffer("s", torch.tensor(s))
        self.register_buffer("mu_x0", torch.tensor(mu_x0))
        self.register_buffer("mu_y", torch.tensor(mu_y))
        self.register_buffer("std", torch.tensor(std))

    def q_sample(self, t, x0, y, return_eps: bool = False):
        """ Sample q(x_t | x_0, y). If return_eps=True, also return the used noise. """
        shape = [-1] + [1] * (x0.ndim - 1)

        mu_x0 = self.mu_x0[t].view(shape)
        mu_y = self.mu_y[t].view(shape)
        std = self.std[t].view(shape)
        eps = torch.randn_like(x0)
        x_t = mu_x0*x0 + mu_y*y + std*eps
        if return_eps:
            return x_t.detach(), eps.detach()
        return x_t.detach()

    def q_posterior(self, t, x_t, x0, y):
        """ Sample p(x_{t-1} | x_t, x0, y) """
        shape = [-1] + [1] * (x0.ndim - 1)

        std_t = self.s[t].view(shape)
        std_tm1 = self.s[t-1].view(shape)
        mu_x0_t = self.mu_x0[t].view(shape)
        mu_x0_tm1 = self.mu_x0[t-1].view(shape)
        mu_y_t = self.mu_y[t].view(shape)
        mu_y_tm1 = self.mu_y[t-1].view(shape)

        var_t = std_t**2
        var_tm1 = std_tm1**2
        var_t_tm1 = var_t - var_tm1 * (mu_x0_t / mu_x0_tm1)**2
        v = var_t_tm1 * (var_tm1 / var_t)

        x_tm1_mean = mu_x0_tm1 * x0 + mu_y_tm1 * y + \
            ((var_tm1 - v) / var_t).sqrt() * (x_t - mu_x0_t * x0 - mu_y_t * y)

        x_tm1 = x_tm1_mean + v.sqrt() * torch.randn_like(x_t)
        return x_tm1

    def q_posterior_sample_shared(self, t, x_t, x0, y, xi):
        """Sample x_{t-1} using provided standard normal xi (shared noise for GT/pred)."""
        shape = [-1] + [1] * (x0.ndim - 1)

        std_t = self.s[t].view(shape)
        std_tm1 = self.s[t-1].view(shape)
        mu_x0_t = self.mu_x0[t].view(shape)
        mu_x0_tm1 = self.mu_x0[t-1].view(shape)
        mu_y_t = self.mu_y[t].view(shape)
        mu_y_tm1 = self.mu_y[t-1].view(shape)

        var_t = std_t**2
        var_tm1 = std_tm1**2
        var_t_tm1 = var_t - var_tm1 * (mu_x0_t / mu_x0_tm1)**2
        v = var_t_tm1 * (var_tm1 / var_t)

        x_tm1_mean = mu_x0_tm1 * x0 + mu_y_tm1 * y + \
            ((var_tm1 - v) / var_t).sqrt() * (x_t - mu_x0_t * x0 - mu_y_t * y)
        return x_tm1_mean + v.sqrt() * xi

    def q_posterior_mean(self, t, x_t, x0, y):
        """Return E[x_{t-1} | x_t, x0, y] (no sampling)."""
        shape = [-1] + [1] * (x0.ndim - 1)

        std_t = self.s[t].view(shape)
        std_tm1 = self.s[t-1].view(shape)
        mu_x0_t = self.mu_x0[t].view(shape)
        mu_x0_tm1 = self.mu_x0[t-1].view(shape)
        mu_y_t = self.mu_y[t].view(shape)
        mu_y_tm1 = self.mu_y[t-1].view(shape)

        var_t = std_t**2
        var_tm1 = std_tm1**2
        var_t_tm1 = var_t - var_tm1 * (mu_x0_t / mu_x0_tm1)**2
        v = var_t_tm1 * (var_tm1 / var_t)

        x_tm1_mean = mu_x0_tm1 * x0 + mu_y_tm1 * y + \
            ((var_tm1 - v) / var_t).sqrt() * (x_t - mu_x0_t * x0 - mu_y_t * y)
        return x_tm1_mean

    def posterior_coeffs(self, t):
        """
        Return linear coefficients (a_t, b_t, c_t) such that
        q_posterior_mean(t, x_t, x0, y) = a_t * x0 + b_t * y + c_t * x_t

        t: LongTensor [B], values in [1..n_steps]
        Returns 1D tensors of shape [B] (per-sample scalars).
        """
        std_t = self.s[t]
        std_tm1 = self.s[t-1]
        mu_x0_t = self.mu_x0[t]
        mu_x0_tm1 = self.mu_x0[t-1]
        mu_y_t = self.mu_y[t]
        mu_y_tm1 = self.mu_y[t-1]

        var_t = std_t**2
        var_tm1 = std_tm1**2
        var_t_tm1 = var_t - var_tm1 * (mu_x0_t / mu_x0_tm1)**2
        v = var_t_tm1 * (var_tm1 / var_t)
        alpha = ((var_tm1 - v) / var_t).sqrt()

        a_t = mu_x0_tm1 - alpha * mu_x0_t
        b_t = mu_y_tm1 - alpha * mu_y_t
        c_t = alpha
        return a_t, b_t, c_t

    def q_posterior2_mean(self, t, x_t, x0_t, y):
        """
        Two-step direct mean to t-2 without re-estimating x0 at t-1:
        returns E[x_{t-2} | x_t, x0 at t, y].
        """
        x_tm1_mean = self.q_posterior_mean(t, x_t, x0_t, y)
        x_tm2_mean = self.q_posterior_mean(t-1, x_tm1_mean, x0_t, y)
        return x_tm2_mean

    def forward_residual_normed(self, t, x_t, x0_hat, y):
        """
        Normalized forward residual for NLL/MSE:
        z_t = (x_t - mu_x0[t]*x0_hat - mu_y[t]*y) / std[t]
        """
        shape = [-1] + [1] * (x_t.ndim - 1)
        mu_x0 = self.mu_x0[t].view(shape)
        mu_y = self.mu_y[t].view(shape)
        std = self.std[t].view(shape)
        return (x_t - mu_x0 * x0_hat - mu_y * y) / (std + 1e-8)

    @torch.inference_mode()
    def sample_x0(self, y, generator, anderson=None, sc_mode: str | None = None, self_guided: bool = False, deterministic_z: bool | None = None):
        """
        Sample p(x_0 | y).
        - sc_mode 决定步内策略（训练与验证/测试一致）：
            standard -> 推理侧 SC：每个时间步一次前向，x_r=上一时刻 x0
            recursion -> 步内递归（或数值解/Anderson，按 env 选择求解器/AA）
            none -> 无 SC：每个时间步一次前向，x_r=None
        """
        # 设置时间步（降序）

        timesteps = torch.arange(self.n_steps, 0, -1, device=y.device)
        timesteps = timesteps.unsqueeze(1).repeat(1, y.shape[0])  # [n_steps, B]

        # 采样 x_T
        x_t = self.q_sample(timesteps[0], torch.zeros_like(y), y)

        # 统一按 sc_mode 路由；默认 recursion
        if isinstance(sc_mode, str):
            sc_mode = sc_mode.lower().strip()
        else:
            sc_mode = "recursion"
        # 推理侧SC
        if sc_mode == "standard":
            prev_x0 = None
            # 固定整条轨迹的 z，使跨步自条件时网络保持一致性
            if generator.nz and generator.nz > 0:
                z_traj = torch.zeros(y.shape[0], generator.nz, device=y.device) if deterministic_z else torch.randn(y.shape[0], generator.nz, device=y.device)
            else:
                z_traj = None
            for t in timesteps:
                if not self_guided:
                    sc_in = None if prev_x0 is None else prev_x0
                    x0_pred = generator(torch.cat((x_t, y), dim=1), t, x_r=sc_in, z_in=z_traj)
                else:
                    # Self-guided：先以 prev_x0 作为自条件得到种子，再用该种子做一次同步 refine
                    seed_in = None if prev_x0 is None else prev_x0
                    x0_seed = generator(torch.cat((x_t, y), dim=1), t, x_r=seed_in, z_in=z_traj)
                    x0_pred = generator(torch.cat((x_t, y), dim=1), t, x_r=x0_seed.detach(), z_in=z_traj)
                x_t = self.q_posterior(t, x_t, x0_pred, y)
                prev_x0 = x0_pred.detach()
            return x0_pred

        # 关闭 SC：每个时间步只做一次前向，且 x_r=None（等参“无 SC”）
        if sc_mode == "none":
            for t in timesteps:
                if generator.nz and generator.nz > 0:
                    z_step = torch.zeros(x_t.shape[0], generator.nz, device=y.device) if deterministic_z else torch.randn(x_t.shape[0], generator.nz, device=y.device)
                else:
                    z_step = None
                x0_pred = generator(torch.cat((x_t, y), dim=1), t, x_r=None, z_in=z_step)
                x_t = self.q_posterior(t, x_t, x0_pred, y)
            return x0_pred

        # 解析求解器配置（env 优先，若未设置则回退到 Anderson 或基线递归）
        solver = os.getenv("SELFRDB_SOLVER", "").lower().strip()  # '', 'picard', 'heun2', 'extrap'

        use_solver = solver in {"picard", "heun2", "extrap", "extrap_safe"}
        m_steps = int(os.getenv("SELFRDB_SOLVER_STEPS", "1"))
        heun_theta = float(os.getenv("SELFRDB_HEUN_THETA", "0.5"))
        extr_gamma = float(os.getenv("SELFRDB_EXTRAP_GAMMA", "0.5"))
        if self.global_rank == 0 and use_solver:
            print(f"[InnerSolver] {solver} / m_steps={m_steps}")
        # 解析 Anderson 配置（YAML 优先，其次环境变量），当未指定新求解器时生效
        if isinstance(anderson, dict):
            use_aa = (not use_solver) and bool(anderson.get("enabled", False))
            aa_m = int(anderson.get("m", 3))
            aa_lam = float(anderson.get("lam", 1e-3))
            aa_damp = float(anderson.get("damping", 0.2))
            aa_tol = anderson.get("tol", None)
            aa_safe = bool(anderson.get("safeguard", True))
        else:
            use_aa = (not use_solver) and (os.getenv("SELFRDB_ANDERSON", "0") == "1")
            aa_m = int(os.getenv("SELFRDB_AA_M", "3"))
            aa_lam = float(os.getenv("SELFRDB_AA_LAM", "1e-3"))
            aa_damp = float(os.getenv("SELFRDB_AA_DAMP", "0.2"))
            aa_tol = None
            aa_safe = True

        for t in timesteps:
            # 为步内递归/数值解固定 z，使 F(x) 对给定 (t,x_t,y) 确定
            if generator.nz and generator.nz > 0:
                z_shared = torch.zeros(x_t.shape[0], generator.nz, device=y.device) if deterministic_z else torch.randn(x_t.shape[0], generator.nz, device=y.device)
            else:
                z_shared = None
            if use_solver:
                # 数值求解器：对步内固定点 x -> F(x)
                def F(x_r):
                    return generator(torch.cat((x_t, y), dim=1), t, x_r=x_r, z_in=z_shared)
                x0_init = torch.zeros_like(x_t)
                if solver == "picard":
                    x0_pred = picard(F, x0_init, K=max(m_steps, 1), centered=True)
                elif solver == "extrap":
                    x0_pred = extrap(F, x0_init, M=max(m_steps, 1), gamma=extr_gamma, centered=True)
                elif solver == "extrap_safe":
                    gamma = float(os.getenv("SELFRDB_EXTRAP_GAMMA", "0.3"))
                    maxbt = int(os.getenv("SELFRDB_EXTRAP_MAXBT", "3"))
                    # 近似脑区：y in [-1,1]，背景≈-1；>-0.95 视为“非背景”
                    mask = (y > -0.95).to(x_t.dtype)
                    x0_pred = extrap_safe(F, x0_init, gamma=gamma, centered=True,
                                        mask=mask, max_backtracks=maxbt, improve_ratio=0.05)
                else:  # 'heun2' 默认
                    x0_pred = heun2(F, x0_init, M=max(m_steps, 1), theta=heun_theta, centered=True)
            elif use_aa:
                # Anderson 加速：对步内递归 x_r -> F(x_r)
                def F(x_r):
                    return generator(torch.cat((x_t, y), dim=1), t, x_r=x_r, z_in=z_shared)
                x0_pred = anderson_accel(
                    F, torch.zeros_like(x_t),
                    m=aa_m, lam=aa_lam,
                    damping=aa_damp if aa_damp is not None else 0.2,
                    K=self.n_recursions,
                    tol=aa_tol,               # 推理期通常 None：走满 K
                    safeguard=aa_safe         # 保序护栏
                )
            else:
                # 基线：原始自一致递归 + 早停（阈值为 0 则基本走满 K）
                x0_r = torch.zeros_like(x_t)
                for _ in range(self.n_recursions):
                    x0_rp1 = generator(torch.cat((x_t, y), dim=1), t, x_r=x0_r, z_in=z_shared)
                    # 标量化 change，避免分布式/不同维度导致的比较问题
                    change = (x0_rp1 - x0_r).abs().mean().item()
                    if change < float(self.consistency_threshold):
                        x0_r = x0_rp1
                        break
                    x0_r = x0_rp1
                x0_pred = x0_r

            # 后验采样 q(x_{t-1} | x_t, y, x0_pred)
            x_tm1_pred = self.q_posterior(t, x_t, x0_pred, y)
            x_t = x_tm1_pred

        return x0_pred
    
    def _get_betas(self):
        betas_len = self.n_steps + 1
        betas = np.linspace(self.beta_start**0.5, self.beta_end**0.5, betas_len)**2
        
        # Discretization correction
        betas = np.append(0., betas).astype(np.float32)
        
        # Handle odd number of betas
        if betas_len % 2 == 1:
            betas = np.concatenate([
                betas[:betas_len//2],
                [betas[betas_len//2]],
                np.flip(betas[:betas_len//2])
            ])
        else:
            betas = np.concatenate([
                betas[:betas_len//2],
                np.flip(betas[:betas_len//2])
            ])
        return betas

    @staticmethod
    def gaussian_product(sigma1, sigma2):
        denom = sigma1**2 + sigma2**2
        mu1 = sigma2**2 / denom
        mu2 = sigma1**2 / denom
        var = (sigma1**2 * sigma2**2) / denom
        return mu1, mu2, var
    
    def vis_scheduler(self):
        plt.figure(figsize=(6, 3))
        plt.plot(self.std**2, label=r'$\sigma_t^2$')
        plt.plot(self.mu_x0, label=r'$\mu_{x_0}$')
        plt.plot(self.mu_y, label=r'$\mu_{y}$')
        plt.legend()
        plt.show()
