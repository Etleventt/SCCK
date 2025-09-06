import os
from typing import Optional
from random import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L
from lightning.pytorch.cli import LightningCLI

from diffusion import DiffusionBridge
from backbones.ncsnpp import NCSNpp
from backbones.discriminator import Discriminator_large
from datasets import DataModule
from utils import compute_metrics, compute_metrics_official, save_image_pair, save_preds, save_eval_images, center_crop
from anderson import anderson_accel
from fid import InceptionFeatureExtractor, compute_fid, compute_fid_from_numpy


class BridgeRunner(L.LightningModule):
    def __init__(
        self,
        generator_params,
        discriminator_params,
        diffusion_params,
        lr_g,
        lr_d,
        disc_grad_penalty_freq,
        disc_grad_penalty_weight,
        lambda_rec_loss,
        optim_betas,
        eval_mask,
        eval_subject,
        # --- 官方兼容开关：仅影响“有判别器”训练路径的 z 传递方式 ---
        official_compat: bool = False,
        # --- Loss weights (YAML-controlled; defaults, YAML can override) ---
        lambda_noise: float = 1.0,
        lambda_post: float = 0.25,
        anderson=None,   # ★ 新增：接受 YAML 的 model.anderson 块
        # --- SC / Drop-R 相关（统一用 sc_mode 配置；use_standard_sc 已弃用） ---
        use_standard_sc: bool = False,
        sc_mode: str = "auto",  # standard | recursion | none | auto(兼容旧配置)
        sc_prob: float = 0.5,
        sc_stop_grad: bool = True,
        drop_r_prob: float = 0.0,
        # --- CK 跨步一致性（仅在无判别器分支启用） ---
        lambda_ck: float = 0.0,
        ck_prob: float = 0.5,
        ck_time_norm: bool = True,
        ck_detach_tm1: bool = True,
        ck_shared_noise: bool = True,
        ck_warmup_steps: int = 0,
        ck_t_lo_frac: float = 0.0,
        ck_t_hi_frac: float = 1.0,
        ck_fix_sc_mask: bool = False,
        # --- 额外开关：验证/测试是否启用 self-guided ---
        eval_self_guided: bool = False,
        # --- 评测指标：是否采用官方对齐的 PSNR/SSIM 计算（裁剪到掩膜尺寸 + per-slice 归一化 + data_range=gt.max） ---
        eval_metrics_official: bool = False,
        # --- 验证期诊断指标（仅记录，不参与训练） ---
        eval_diag_metrics: bool = False,
        # --- z 确定性开关（默认开启：全局 z=0） ---
        deterministic_z: bool = True,
        # --- FID 开关与采样上限（默认关闭；可在 YAML 打开） ---
        eval_fid_val: bool = False,
        eval_fid_test: bool = False,
        fid_max_samples_val: int = 0,   # 0 表示不限制
        fid_max_samples_test: int = 0,  # 0 表示不限制
        fid_weights_path: Optional[str] = None,
        # --- 测试阶段：是否仅计算 FID（跳过 PSNR/SSIM 与示例图） ---
        test_only_fid: bool = False,
        # --- FID 细节开关 ---
        fid_use_mask: bool = False,          # 仅在掩膜区域计算 FID（背景置零）
        fid_crop_to_original: bool = False,  # 计算 FID 前裁剪回未填充尺寸
        # --- FID 调试：保存送入 Inception 的输入图（去均值/方差后 PNG + 原始张量） ---
        fid_dump_inputs: bool = False,
        fid_dump_n: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.disc_grad_penalty_freq = disc_grad_penalty_freq
        self.disc_grad_penalty_weight = disc_grad_penalty_weight
        self.lambda_rec_loss = lambda_rec_loss
        # Loss weights from config (no env override)
        self.lambda_noise = float(lambda_noise)
        self.lambda_post  = float(lambda_post)
        self.optim_betas = optim_betas
        self.eval_mask = eval_mask
        self.eval_subject = eval_subject
        self.eval_metrics_official = bool(eval_metrics_official)
        self.n_steps = diffusion_params['n_steps']
        self.n_recursions = diffusion_params['n_recursions']
        # 官方兼容（仅用于有判别器路径）：True 时不向生成器传入 z_in，使其每次前向内部采样 z
        self.official_compat = bool(official_compat)

        # Networks
        self.generator = NCSNpp(**generator_params)
        # 判别器可选
        self.use_discriminator = (discriminator_params is not None)
        self.discriminator = Discriminator_large(**discriminator_params) if self.use_discriminator else None

        # Configure diffusion
        self.diffusion = DiffusionBridge(**diffusion_params)

        # 保存 YAML 的 anderson 配置（验证/测试时传入）
        self.anderson_cfg = anderson if isinstance(anderson, dict) else None

        # 训练端是否启用 AA（保持你原先的环境变量开关）
        self.use_aa_train = os.getenv("SELFRDB_ANDERSON_TRAIN", "0") == "1"
        self.aa_m    = int(os.getenv("SELFRDB_AA_M", "3"))
        self.aa_lam  = float(os.getenv("SELFRDB_AA_LAM", "1e-3"))
        self.aa_damp = float(os.getenv("SELFRDB_AA_DAMP", "0.2"))

        # 仅使用 sc_mode 决定策略；use_standard_sc 仅作兼容映射，后续将移除
        if use_standard_sc:
            print('[DEPRECATION] model.use_standard_sc 已弃用，请改用 model.sc_mode=standard（当前将自动映射）。')
        self.sc_mode = (sc_mode or "recursion").lower() if isinstance(sc_mode, str) else "recursion"
        if self.sc_mode == "auto":
            self.sc_mode = "standard" if bool(use_standard_sc) else "recursion"
        if self.sc_mode not in {"standard","recursion","none"}:
            raise ValueError(f"Unsupported sc_mode: {self.sc_mode}")
        # 兼容旧属性
        self.use_standard_sc = (self.sc_mode == "standard")
        self.sc_prob = float(os.getenv("SELFRDB_SC_PROB", str(sc_prob)))
        self.sc_stop_grad = (os.getenv("SELFRDB_SC_STOP_GRAD", "1" if sc_stop_grad else "0") == "1")
        self.drop_r_prob = float(os.getenv("SELFRDB_DROP_R_PROB", str(drop_r_prob)))

        # CK 超参（默认关闭；仅在无判别器路径使用）
        self.lambda_ck = float(os.getenv("SELFRDB_LAMBDA_CK", str(lambda_ck)))
        self.ck_prob = float(os.getenv("SELFRDB_CK_PROB", str(ck_prob)))
        self.ck_time_norm = (os.getenv("SELFRDB_CK_TIME_NORM", "1" if ck_time_norm else "0") == "1")
        self.ck_detach_tm1 = (os.getenv("SELFRDB_CK_DETACH", "1" if ck_detach_tm1 else "0") == "1")
        self.ck_shared_noise = (os.getenv("SELFRDB_CK_SHARED_NOISE", "1" if ck_shared_noise else "0") == "1")
        self.ck_warmup_steps = int(os.getenv("SELFRDB_CK_WARMUP", str(ck_warmup_steps)))
        self.ck_t_lo_frac = float(os.getenv("SELFRDB_CK_T_LO_FRAC", str(ck_t_lo_frac)))
        self.ck_t_hi_frac = float(os.getenv("SELFRDB_CK_T_HI_FRAC", str(ck_t_hi_frac)))
        self.ck_fix_sc_mask = (os.getenv("SELFRDB_CK_FIX_SC_MASK", "1" if ck_fix_sc_mask else "0") == "1")
        # 配置项：不依赖环境变量
        self.eval_self_guided = bool(eval_self_guided)
        self.eval_diag_metrics = bool(eval_diag_metrics)
        # z 开关
        self.deterministic_z = bool(deterministic_z)
        # FID 配置
        self.eval_fid_val = bool(eval_fid_val)
        self.eval_fid_test = bool(eval_fid_test)
        self.fid_max_samples_val = int(fid_max_samples_val or 0)
        self.fid_max_samples_test = int(fid_max_samples_test or 0)
        self.fid_weights_path = fid_weights_path
        self.test_only_fid = bool(test_only_fid)
        self.fid_use_mask = bool(fid_use_mask)
        self.fid_crop_to_original = bool(fid_crop_to_original)
        self.fid_dump_inputs = bool(fid_dump_inputs)
        self.fid_dump_n = int(fid_dump_n or 0)
        # FID 运行时缓存
        self._fid_net = None
        self._fid_disabled_reason = None
        self._fid_val_real = []
        self._fid_val_fake = []
        self._fid_val_count = 0

        # 无判别器分支：自动优化
        if not self.use_discriminator:
            self.automatic_optimization = True

    def _z(self, B: int, device: torch.device):
        if getattr(self.generator, 'nz', 0) == 0:
            return None
        if self.deterministic_z:
            return torch.zeros(B, self.generator.nz, device=device)
        return torch.randn(B, self.generator.nz, device=device)

    # --------------------
    # Training
    # --------------------
    def training_step(self, batch):
        x0, y, _ = batch

        # ===== 无判别器路径：噪声监督 + 共享噪声后验（自动优化） =====
        if not self.use_discriminator:
            # 采样一个时间步并生成 x_t（前向桥）
            t = torch.randint(1, self.n_steps+1, (x0.shape[0],), device=x0.device)
            x_t, eps = self.diffusion.q_sample(t, x0, y, return_eps=True)

            # 选择步内策略：standard | recursion | none
            if self.sc_mode == "standard":
                # 标准 SC：同一 t 两次前向 + stop-grad + 逐样本掩码
                # 关键改动：两次前向共享同一个 z，避免额外随机性干扰 SC
                z_shared = self._z(x_t.shape[0], x_t.device)
                x0_hat1 = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=None, z_in=z_shared)
                sc_in = x0_hat1.detach() if self.sc_stop_grad else x0_hat1
                m = (torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) < self.sc_prob).float()
                sc_in = sc_in * m
                x0_pred = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=sc_in, z_in=z_shared)
            elif self.sc_mode == "recursion":
                # 自递归：可按需要走 Drop-R/AA（一般建议关闭以保持一致）
                local_r = self.n_recursions
                if local_r > 1 and self.drop_r_prob > 0.0 and torch.rand(()) < self.drop_r_prob:
                    local_r = max(1, local_r - 1)
                if self.use_aa_train:
                    # 步内递归：固定 z 以保证 F(x) 的确定性
                    z_shared = self._z(x_t.shape[0], x_t.device)
                    def F_step(x_r):
                        return self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x_r, z_in=z_shared)
                    x0_pred = anderson_accel(
                        F_step, torch.zeros_like(x_t),
                        m=self.aa_m, lam=self.aa_lam, damping=self.aa_damp,
                        K=local_r, tol=None
                    )
                else:
                    z_shared = self._z(x_t.shape[0], x_t.device)
                    x0_r = torch.zeros_like(x_t)
                    for _ in range(local_r):
                        x0_r = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x0_r, z_in=z_shared)
                    x0_pred = x0_r
            else:  # none
                # 关闭 SC/递归：单次前向，x_r=None；z 按 deterministic_z 选择
                x0_pred = self.generator(
                    torch.cat((x_t.detach(), y), dim=1), t, x_r=None, z_in=self._z(x_t.shape[0], x_t.device)
                )

            # 噪声监督（DDPM）：epsilon 预测 vs 真实 eps
            shape = [-1] + [1] * (x0.ndim - 1)
            mu_x0 = self.diffusion.mu_x0[t].view(shape)
            mu_y  = self.diffusion.mu_y[t].view(shape)
            std   = self.diffusion.std[t].view(shape)
            eps_hat = (x_t - mu_x0 * x0_pred - mu_y * y) / (std + 1e-8)
            noise_loss = F.mse_loss(eps_hat, eps)

            # 共享噪声的后验对齐（同一 xi）
            xi = torch.randn_like(x0)
            x_tm1_pred = self.diffusion.q_posterior_sample_shared(t, x_t, x0_pred, y, xi)
            x_tm1_gt   = self.diffusion.q_posterior_sample_shared(t, x_t, x0,      y, xi)
            post_loss  = F.l1_loss(x_tm1_pred, x_tm1_gt)

            # 直接重建（小权重）
            rec_loss   = F.l1_loss(x0_pred, x0)
            # 可选：跨步一致性 CK（仅无判别器路径）
            ck_loss = torch.tensor(0.0, device=x0.device)
            if self.lambda_ck > 0.0 and (torch.rand(()) < self.ck_prob):
                # 融合外层 standard-SC：复用 t/x_t/x0_pred/m/z_shared，避免 CK 内重复做 inner
                fused = None
                if self.sc_mode == "standard":
                    fused = {
                        "t": t,
                        "x_t": x_t,
                        "x0_t": x0_pred,
                        "m": m,
                        "z_t": z_shared,
                    }
                ck_loss = self._compute_ck_loss(x0, y, fused=fused)

            loss = self.lambda_noise*noise_loss + self.lambda_post*post_loss + self.lambda_rec_loss*rec_loss \
                   + self.lambda_ck * ck_loss

            self.log_dict(
                {"loss/noise": noise_loss, "loss/post": post_loss, "loss/rec": rec_loss,
                 "loss/ck": ck_loss, "loss/total": loss},
                on_epoch=True, prog_bar=True, sync_dist=True
            )
            return loss

        # ===== 有判别器路径：手动优化（原始 SelfRDB 逻辑） =====
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        # Part 1: Train discriminator
        self.toggle_optimizer(optimizer_d)

        # 1.a Real
        t = torch.randint(1, self.n_steps+1, (x0.shape[0],), device=x0.device)
        x_tm1 = self.diffusion.q_sample(t - 1, x0, y)
        x_t = self.diffusion.q_sample(t, x0, y)
        x_t.requires_grad = True

        disc_out = self.discriminator(x_tm1, x_t, t)
        real_loss = self.adversarial_loss(disc_out, is_real=True)
        disc_real_acc = (disc_out > 0).float().mean()

        if self.global_step % self.disc_grad_penalty_freq == 0:
            grads = torch.autograd.grad(outputs=disc_out.sum(), inputs=x_t, create_graph=True)[0]
            grad_penalty = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = grad_penalty * self.disc_grad_penalty_weight
            real_loss += grad_penalty

        # 1.b Fake
        if self.sc_mode == "standard":
            # 标准SC：同一 t 两次前向；第二次以 stop-grad 的第一次输出为 SC 通道（逐样本掩码）
            # 关键改动（默认）：两次前向共享同一个 z；若 official_compat，则不传 z_in（与官方实现保持一致）
            z_shared = None if self.official_compat else self._z(x_t.shape[0], x_t.device)
            x0_hat1 = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=None, z_in=z_shared)
            sc_in = x0_hat1.detach() if self.sc_stop_grad else x0_hat1
            m = (torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) < self.sc_prob).float()
            sc_in = sc_in * m
            x0_pred = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=sc_in, z_in=z_shared)
        elif self.sc_mode == "recursion":
            # Drop-R：以概率将步内递归从 n_recursions 减 1（至少为 1）
            local_r = self.n_recursions
            if local_r > 1 and self.drop_r_prob > 0.0 and random() < self.drop_r_prob:
                local_r = max(1, local_r - 1)
            if self.use_aa_train:
                z_shared = self._z(x_t.shape[0], x_t.device)
                def F_step(x_r):
                    return self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x_r, z_in=z_shared)
                x0_pred = anderson_accel(
                    F_step, torch.zeros_like(x_t),
                    m=self.aa_m, lam=self.aa_lam, damping=self.aa_damp,
                    K=local_r, tol=None  # 训练端不早停
                )
            else:
                # 官方兼容：递归内每次前向不传 z_in，由网络内部采样随机 z；否则共享 z 以减少随机性
                z_shared = None if self.official_compat else self._z(x_t.shape[0], x_t.device)
                x0_r = torch.zeros_like(x_t)
                for _ in range(local_r):
                    x0_r = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x0_r, z_in=z_shared)
                x0_pred = x0_r
        else:  # none
            x0_pred = self.generator(
                torch.cat((x_t.detach(), y), dim=1), t, x_r=None, z_in=self._z(x_t.shape[0], x_t.device)
            )

        x_tm1_pred = self.diffusion.q_posterior(t, x_t, x0_pred, y)
        disc_out = self.discriminator(x_tm1_pred, x_t, t)
        fake_loss = self.adversarial_loss(disc_out, is_real=False)
        disc_fake_acc = (disc_out < 0).float().mean()

        d_acc = (disc_real_acc + disc_fake_acc) / 2
        d_loss = real_loss + fake_loss

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Part 2: Train generator
        self.toggle_optimizer(optimizer_g)

        t = torch.randint(1, self.n_steps+1, (x0.shape[0],), device=x0.device)
        x_t = self.diffusion.q_sample(t, x0, y)

        if self.sc_mode == "standard":
            z_shared = None if self.official_compat else self._z(x_t.shape[0], x_t.device)
            x0_hat1 = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=None, z_in=z_shared)
            sc_in = x0_hat1.detach() if self.sc_stop_grad else x0_hat1
            m = (torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) < self.sc_prob).float()
            sc_in = sc_in * m
            x0_pred = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=sc_in, z_in=z_shared)
        elif self.sc_mode == "recursion":
            local_r = self.n_recursions
            if local_r > 1 and self.drop_r_prob > 0.0 and random() < self.drop_r_prob:
                local_r = max(1, local_r - 1)
            if self.use_aa_train:
                z_shared = self._z(x_t.shape[0], x_t.device)
                def F_step(x_r):
                    return self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x_r, z_in=z_shared)
                x0_pred = anderson_accel(
                    F_step, torch.zeros_like(x_t),
                    m=self.aa_m, lam=self.aa_lam, damping=self.aa_damp,
                    K=local_r, tol=None
                )
            else:
                z_shared = None if self.official_compat else self._z(x_t.shape[0], x_t.device)
                x0_r = torch.zeros_like(x_t)
                for _ in range(local_r):
                    x0_r = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x0_r, z_in=z_shared)
                x0_pred = x0_r
        else:
            x0_pred = self.generator(
                torch.cat((x_t.detach(), y), dim=1), t, x_r=None, z_in=self._z(x_t.shape[0], x_t.device)
            )

        x_tm1_pred = self.diffusion.q_posterior(t, x_t, x0_pred, y)

        rec_loss = F.l1_loss(x0_pred, x0, reduction="sum")
        adv_loss = self.adversarial_loss(self.discriminator(x_tm1_pred, x_t, t), is_real=True)
        g_loss = self.lambda_rec_loss*rec_loss + adv_loss

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        scheduler_g.step()
        scheduler_d.step()
        
        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/rec", rec_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/adv", adv_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("g_loss/total", g_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    # --------------------
    # Validation
    # --------------------
    def validation_step(self, batch, batch_idx):
        x0, y, idx = batch
        # 推理：把 YAML 的 anderson 配置传进去（若为 None 则回退到 env/基线）
        x0_pred = self.diffusion.sample_x0(
            y, self.generator, anderson=self.anderson_cfg, sc_mode=self.sc_mode, self_guided=self.eval_self_guided,
            deterministic_z=self.deterministic_z
        )

        loss = F.mse_loss(x0_pred, x0)
        # 可选：验证阶段按需要使用 mask 评估
        val_mask = None
        if self.eval_mask:
            try:
                # 惰性缓存到 runner，避免每步重复加载
                if not hasattr(self, "_val_mask_cache"):
                    self._val_mask_cache = self.trainer.datamodule.val_dataset._load_data('mask')  # (N_val,H,W)
                # 依据当前 batch 的样本索引切片，得到 (B,H,W) 的 mask
                if isinstance(idx, torch.Tensor):
                    index_np = idx.detach().cpu().numpy()
                else:
                    index_np = np.asarray(idx)
                val_mask = self._val_mask_cache[index_np]
            except Exception:
                val_mask = None
        metrics = compute_metrics_official(x0, x0_pred, mask=val_mask) if self.eval_metrics_official \
                  else compute_metrics(x0, x0_pred, mask=val_mask)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_psnr", metrics["psnr_mean"], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", metrics["ssim_mean"], on_epoch=True, prog_bar=True, sync_dist=True)

        # 累计 FID 特征（按批 all_gather 后仅在 rank0 聚合）
        if self.eval_fid_val:
            # 懒加载 Inception（如离线无权重则自动禁用）
            if self._fid_net is None and self._fid_disabled_reason is None:
                try:
                    self._fid_net = InceptionFeatureExtractor(weights_path=self.fid_weights_path).to(x0.device)
                    self._fid_net.eval()
                except Exception as e:
                    self._fid_disabled_reason = str(e)
            if self._fid_net is not None:
                with torch.no_grad():
                    fr = self._fid_net(x0)
                    ff = self._fid_net(x0_pred)
                # all_gather: [W,B,D] -> reshape to [W*B,D]
                all_fr = self.all_gather(fr)
                all_ff = self.all_gather(ff)
                if self.global_rank == 0:
                    fr_cpu = all_fr.reshape(-1, all_fr.shape[-1]).detach().cpu().numpy()
                    ff_cpu = all_ff.reshape(-1, all_ff.shape[-1]).detach().cpu().numpy()
                    # respect max samples if set
                    if self.fid_max_samples_val and self._fid_val_count >= self.fid_max_samples_val:
                        pass
                    else:
                        remain = None if self.fid_max_samples_val == 0 else max(0, self.fid_max_samples_val - self._fid_val_count)
                        if remain is not None and fr_cpu.shape[0] > remain:
                            fr_cpu = fr_cpu[:remain]
                            ff_cpu = ff_cpu[:remain]
                        self._fid_val_real.append(fr_cpu)
                        self._fid_val_fake.append(ff_cpu)
                        self._fid_val_count += fr_cpu.shape[0]

        # Optional diagnostics: only for standard SC and first val batch for cost control
        if self.eval_diag_metrics and self.sc_mode == "standard" and batch_idx == 0:
            with torch.no_grad():
                B = x0.shape[0]
                device = x0.device
                # sample t in [2..n]
                t = torch.randint(2, self.n_steps + 1, (B,), device=device)
                x_t = self.diffusion.q_sample(t, x0, y)

                # Step t: standard two-pass with shared z_t (full mask for determinism)
                z_t = torch.zeros(B, self.generator.nz, device=device) if self.deterministic_z else torch.randn(B, self.generator.nz, device=device)
                x0_hat1_t = self.generator(torch.cat((x_t, y), dim=1), t, x_r=None, z_in=z_t)
                sc_in_t = x0_hat1_t.detach() if self.sc_stop_grad else x0_hat1_t
                x0_t = self.generator(torch.cat((x_t, y), dim=1), t, x_r=sc_in_t, z_in=z_t)

                # Mean transition to t-1 to avoid sampling noise
                x_tm1_mean = self.diffusion.q_posterior_mean(t, x_t, x0_t, y)
                # 复用同一份 z，避免无关随机性放大跨步差异
                z_tm1 = z_t
                x_tm1_in = x_tm1_mean.detach() if self.ck_detach_tm1 else x_tm1_mean
                # x_r at t-1 uses current x0_t to emulate cross-step conditioning
                x0_tm1 = self.generator(torch.cat((x_tm1_in, y), dim=1), t-1, x_r=x0_t.detach(), z_in=z_tm1)

                # Conditioning gap at t (reuse z_t to isolate conditioning source)
                out_prev = self.generator(torch.cat((x_t, y), dim=1), t, x_r=x0_tm1.detach(), z_in=z_t)
                cond_gap = (x0_t - out_prev).abs().mean()

                # Temporal smoothness between x0_t and x0_tm1
                temp_smooth = (x0_t - x0_tm1).abs().mean()

                self.log_dict({
                    "diag/cond_gap": cond_gap,
                    "diag/temp_smooth": temp_smooth,
                }, on_epoch=True, prog_bar=False, sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0:
            path = os.path.join(self.logger.log_dir, "val_samples", f"epoch_{self.current_epoch}.png")
            save_image_pair(x0, x0_pred, path)

    def on_validation_epoch_start(self):
        # 重置 FID 累计容器
        if self.eval_fid_val:
            self._fid_val_real = []
            self._fid_val_fake = []
            self._fid_val_count = 0
        # 打印验证期指标模式（仅 rank0）
        if getattr(self, 'global_rank', 0) == 0 and self.eval_metrics_official:
            print("[Val] 官方的 PSNR/SSIM 计算模式已启用：crop-to-mask + per-slice normalization + data_range=gt.max")

    def on_validation_epoch_end(self):
        # 仅 rank0 计算并记录 FID
        if self.eval_fid_val and self.global_rank == 0 and self._fid_net is not None:
            try:
                if len(self._fid_val_real) and len(self._fid_val_fake):
                    feats_r = np.concatenate(self._fid_val_real, axis=0)
                    feats_f = np.concatenate(self._fid_val_fake, axis=0)
                    fid = compute_fid(feats_r, feats_f)
                    # 记录到日志（只在rank0，无需 sync_dist）
                    self.log("val_fid", torch.tensor(fid), on_epoch=True, prog_bar=True, sync_dist=False)
            except Exception:
                pass

    # --------------------
    # Test
    # --------------------
    def on_test_start(self):
        self.test_samples = []
        self.psnrs = []
        self.ssims = []
        self.mask = None
        self.subject_ids = None

        if self.eval_mask:
            self.mask = self.trainer.datamodule.test_dataset._load_data('mask')

        if self.eval_subject:
            self.subject_ids = self.trainer.datamodule.test_dataset.subject_ids
        # 打印测试期指标模式（仅 rank0）
        if getattr(self, 'global_rank', 0) == 0 and self.eval_metrics_official:
            print("[Eval] Using official-aligned PSNR/SSIM: crop-to-mask + per-slice normalization + data_range=gt.max")

    def test_step(self, batch, batch_idx):
        x0, y, slice_idx = batch
        x0_pred = self.diffusion.sample_x0(
            y, self.generator, anderson=self.anderson_cfg, sc_mode=self.sc_mode, self_guided=self.eval_self_guided,
            deterministic_z=self.deterministic_z
        )

        all_pred = self.all_gather(x0_pred)
        slice_indices = self.all_gather(slice_idx)
        
        if self.global_rank == 0:
            h, w = x0.shape[-2:]
            self.test_samples.extend(list(zip(
                slice_indices.flatten().tolist(),
                all_pred.reshape(-1, h, w).cpu().numpy())))

    def on_test_end(self):
        if self.global_rank == 0:
            self.test_samples.sort(key=lambda x: x[0])
            pred = np.array([x[1] for x in self.test_samples])
            slice_indices = np.array([x[0] for x in self.test_samples])

            _, locs = np.unique(slice_indices, return_index=True)
            pred = pred[locs]

            dataset = self.trainer.datamodule.test_dataset
            source = dataset.source
            target = dataset.target

            path = os.path.join(self.logger.log_dir, "test_samples", "pred.npy")
            save_preds(pred, path)

            metrics = None
            if not self.test_only_fid:
                if self.eval_metrics_official:
                    metrics = compute_metrics_official(
                        gt_images=target,
                        pred_images=pred,
                        mask=self.mask,
                        subject_ids=self.subject_ids,
                        report_path=os.path.join(self.logger.log_dir, "test_samples", "report.txt")
                    )
                else:
                    metrics = compute_metrics(
                        gt_images=target,
                        pred_images=pred,
                        mask=self.mask,
                        subject_ids=self.subject_ids,
                        report_path=os.path.join(self.logger.log_dir, "test_samples", "report.txt")
                    )

                print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
                print(f"SSIM: {metrics['ssim_mean']:.2f} ± {metrics['ssim_std']:.2f}")

            # 计算并报告 FID（可选）
            if self.eval_fid_test:
                try:
                    # respect max samples if set
                    if self.fid_max_samples_test and len(pred) > self.fid_max_samples_test:
                        idx = np.arange(len(pred))
                        # deterministic subset: take first N (consistent with datamodule default)
                        idx = idx[: self.fid_max_samples_test]
                        pred_eval = pred[idx]
                        target_eval = target[idx]
                    else:
                        pred_eval = pred
                        target_eval = target
                    # Align shapes for FID: ensure both are [N,H,W]
                    if isinstance(target_eval, np.ndarray) and target_eval.ndim == 4 and target_eval.shape[1] == 1:
                        target_eval = target_eval[:, 0, ...]
                    if isinstance(pred_eval, np.ndarray) and pred_eval.ndim == 4 and pred_eval.shape[1] == 1:
                        pred_eval = pred_eval[:, 0, ...]

                    # Optional: crop back to original (pre-padding) size to avoid background bias
                    if self.fid_crop_to_original and getattr(dataset, 'original_shape', None) is not None:
                        crop_h, crop_w = dataset.original_shape
                        target_eval = center_crop(target_eval, (int(crop_h), int(crop_w)))
                        pred_eval   = center_crop(pred_eval,   (int(crop_h), int(crop_w)))
                        if self.fid_use_mask and self.mask is not None and self.mask.shape[-2:] != (crop_h, crop_w):
                            mask_use = center_crop(self.mask, (int(crop_h), int(crop_w)))
                        else:
                            mask_use = self.mask
                    else:
                        mask_use = self.mask

                    # Optional: apply mask to focus on brain region
                    if self.fid_use_mask and mask_use is not None:
                        try:
                            target_eval = target_eval * mask_use
                            pred_eval   = pred_eval   * mask_use
                        except Exception:
                            pass

                    # Use Lightning's current device to avoid mismatches under DDP
                    device = self.device if isinstance(self.device, torch.device) else torch.device(str(self.device))
                    dump_dir = os.path.join(self.logger.log_dir, "test_samples", "fid_inputs") if self.fid_dump_inputs else None
                    fid = compute_fid_from_numpy(
                        target_eval,
                        pred_eval,
                        device=device,
                        weights_path=self.fid_weights_path,
                        dump_dir=dump_dir,
                        dump_n=self.fid_dump_n,
                    )
                    if fid is not None:
                        print(f"FID:  {fid:.4f}")
                        # append to report
                        report_path = os.path.join(self.logger.log_dir, "test_samples", "report.txt")
                        try:
                            with open(report_path, 'a') as f:
                                f.write(f"FID: {fid:.4f}\n")
                        except Exception:
                            pass
                    else:
                        print("FID skipped: Inception weights unavailable (returned None)")
                except Exception as e:
                    print(f"FID computation failed: {e}")

            if not self.test_only_fid and metrics is not None:
                indices = np.random.choice(len(dataset), 10)
                save_eval_images(
                    source_images=source[indices],
                    target_images=target[indices],
                    pred_images=pred[indices],
                    psnrs=metrics["psnrs"][indices],
                    ssims=metrics["ssims"][indices],
                    save_path=os.path.join(self.logger.log_dir, "test_samples")
                )

    # --------------------
    # Misc
    # --------------------
    def adversarial_loss(self, pred, is_real):
        loss = F.softplus(-pred) if is_real else F.softplus(pred)
        return loss.mean()
    
    def configure_optimizers(self):
        optimizer_g = Adam(self.generator.parameters(), lr=self.lr_g, betas=self.optim_betas)
        scheduler_g = CosineAnnealingLR(optimizer_g, T_max=self.trainer.max_epochs, eta_min=1e-5)
        if not self.use_discriminator:
            return [optimizer_g], [scheduler_g]

        optimizer_d = Adam(self.discriminator.parameters(), lr=self.lr_d, betas=self.optim_betas)
        scheduler_d = CosineAnnealingLR(optimizer_d, T_max=self.trainer.max_epochs, eta_min=1e-5)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def _compute_ck_loss(self, x0, y, fused: dict | None = None):
        """
        Fused CK in x0-space:
        - If `fused` is provided (standard mode), reuse outer-step results: {t, x_t, x0_t, m, z_t}.
          Compute a single forward at t-1 with temporal-SC (x_r = m * stopgrad(x0_t), z=z_t), then
          minimize || x0_tm1 - x0_t ||_1 (optionally weighted by 1/|a_{t-1}|).
        - Otherwise, fall back to independent sampling of (t, x_t) and compute the same x0-space consistency.
        """
        B = x0.shape[0]
        device = x0.device
        # Warm-up: skip CK for first N steps
        if self.ck_warmup_steps > 0 and self.global_step < self.ck_warmup_steps:
            return torch.tensor(0.0, device=device)

        # 选择 t/x_t/x0_t：若提供 fused，则复用外层；否则独立采样
        if fused is not None:
            t = fused["t"]
            x_t = fused["x_t"]
            x0_t = fused["x0_t"].detach()
            # 外层 SC 的 mask，与 temporal-SC 共享，确保信息量一致
            mask_shape = (B, 1) + (1,) * (x_t.ndim - 2)
            m = fused.get("m", torch.ones(mask_shape, device=device))
            z_ck = fused.get("z_t", self._z(B, device))
        else:
            # Sample t in a middle-noise window [lo, hi] to avoid extremes
            lo_idx = max(2, int(round(self.ck_t_lo_frac * self.n_steps)))
            hi_idx = int(round(self.ck_t_hi_frac * self.n_steps))
            lo_idx = max(2, min(self.n_steps, lo_idx))
            hi_idx = max(lo_idx, min(self.n_steps, hi_idx))
            t = torch.randint(lo_idx, hi_idx + 1, (B,), device=device)
            # sample x_t from forward bridge (no need to return eps here)
            x_t = self.diffusion.q_sample(t, x0, y)
            # One generator forward at t (target branch)
            with torch.no_grad():
                if self.sc_mode == "standard":
                    # 重要：CK 在 t 与 t-1 共享同一个 z，避免把随机性混入对比
                    z_ck = self._z(B, device)
                    x0_hat1_t = self.generator(torch.cat((x_t, y), dim=1), t, x_r=None, z_in=z_ck)
                    sc_in_t = x0_hat1_t.detach() if self.sc_stop_grad else x0_hat1_t
                    mask_shape = (B, 1) + (1,) * (x_t.ndim - 2)
                    m = (torch.rand(mask_shape, device=device) < self.sc_prob).float()
                    sc_in_t = sc_in_t * m
                    x0_t = self.generator(torch.cat((x_t, y), dim=1), t, x_r=sc_in_t, z_in=z_ck)
                elif self.sc_mode == "recursion":
                    z_ck = self._z(B, device)
                    x0_t = self.generator(torch.cat((x_t, y), dim=1), t, x_r=None, z_in=z_ck)
                    mask_shape = (B, 1) + (1,) * (x_t.ndim - 2)
                    m = torch.ones(mask_shape, device=device)
                else:  # none
                    z_ck = self._z(B, device)
                    x0_t = self.generator(torch.cat((x_t, y), dim=1), t, x_r=None, z_in=z_ck)
                    mask_shape = (B, 1) + (1,) * (x_t.ndim - 2)
                    m = torch.ones(mask_shape, device=device)

        # （已在上方根据 fused/独立分支计算 x0_t）

        # === 改为 x0 空间的一致性：直接约束 || x0_tm1 - x0_t ||（可时间归一化） ===
        if self.ck_shared_noise:
            # 构造 x_{t-1}（采样或均值）后，以 temporal-SC 得到 x0_tm1
            xi1 = torch.randn_like(x0)
            x_tm1_in = self.diffusion.q_posterior_sample_shared(t, x_t, x0_t, y, xi1)
        else:
            x_tm1_in = self.diffusion.q_posterior_mean(t, x_t, x0_t, y)

        if self.sc_mode == "standard":
            # t-1 单趟 temporal-SC，统一 mask 与 z
            xr_tm1 = (m if self.ck_fix_sc_mask else 1.0) * x0_t.detach()
            x0_tm1 = self.generator(torch.cat((x_tm1_in.detach() if self.ck_detach_tm1 else x_tm1_in, y), dim=1),
                                    t-1, x_r=xr_tm1, z_in=z_ck)
        elif self.sc_mode == "recursion":
            x0_tm1 = self.generator(torch.cat((x_tm1_in.detach() if self.ck_detach_tm1 else x_tm1_in, y), dim=1), t-1, x_r=None, z_in=z_ck)
        else:
            x0_tm1 = self.generator(torch.cat((x_tm1_in.detach() if self.ck_detach_tm1 else x_tm1_in, y), dim=1), t-1, x_r=None, z_in=z_ck)

        diff_x0 = x0_tm1 - x0_t
        if self.ck_time_norm:
            a_t_minus_1, _, _ = self.diffusion.posterior_coeffs(t-1)
            w_scalar = (a_t_minus_1.abs() + 1e-8).reciprocal().clamp(max=1e3)
            view_shape = (-1,) + (1,) * (diff_x0.ndim - 1)
            w = w_scalar.view(*view_shape)
            diff_x0 = diff_x0 * w
        return F.l1_loss(diff_x0, torch.zeros_like(diff_x0))


class _LightningCLI(LightningCLI):
    def instantiate_classes(self):
        # Log to checkpoint directory when testing
        if 'test' in self.parser.args and 'CSVLogger' in self.config.test.trainer.logger[0].class_path:
            exp_dir = os.path.dirname(os.path.dirname(self.config.test.ckpt_path))
            logger = self.config.test.trainer.logger[0]
            logger.init_args.save_dir = os.path.dirname(exp_dir)
            logger.init_args.name = os.path.basename(exp_dir)
            logger.init_args.version = "test"
        super().instantiate_classes()


def cli_main():
    cli = _LightningCLI(
        BridgeRunner,
        DataModule,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"}
    )


if __name__ == "__main__":
    cli_main()
