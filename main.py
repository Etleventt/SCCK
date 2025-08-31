import os
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
from utils import compute_metrics, save_image_pair, save_preds, save_eval_images
from anderson import anderson_accel


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
        anderson=None,   # ★ 新增：接受 YAML 的 model.anderson 块
        # --- SC / Drop-R 相关（可选，保持向后兼容） ---
        use_standard_sc: bool = False,
        sc_prob: float = 0.5,
        sc_stop_grad: bool = True,
        drop_r_prob: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.disc_grad_penalty_freq = disc_grad_penalty_freq
        self.disc_grad_penalty_weight = disc_grad_penalty_weight
        self.lambda_rec_loss = lambda_rec_loss
        self.optim_betas = optim_betas
        self.eval_mask = eval_mask
        self.eval_subject = eval_subject
        self.n_steps = diffusion_params['n_steps']
        self.n_recursions = diffusion_params['n_recursions']

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

        # 标准 SC 与 Drop-R（环境变量可覆盖传参，默认不启用）
        env_use_sc = os.getenv("SELFRDB_USE_STANDARD_SC")
        self.use_standard_sc = (env_use_sc == "1") if env_use_sc is not None else bool(use_standard_sc)
        self.sc_prob = float(os.getenv("SELFRDB_SC_PROB", str(sc_prob)))
        self.sc_stop_grad = (os.getenv("SELFRDB_SC_STOP_GRAD", "1" if sc_stop_grad else "0") == "1")
        self.drop_r_prob = float(os.getenv("SELFRDB_DROP_R_PROB", str(drop_r_prob)))

        # 无判别器分支：改为自动优化 + 增加无对抗损失权重
        if not self.use_discriminator:
            self.automatic_optimization = True
        # 无判别器权重（噪声监督 + 共享噪声后验）
        self.lambda_noise = float(os.getenv("LAMBDA_NOISE", "1.0"))
        self.lambda_post  = float(os.getenv("LAMBDA_POST",  "0.25"))

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

            # 标准 SC：同一 t 两次前向 + stop-grad + 逐样本掩码
            if self.use_standard_sc:
                x0_hat1 = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=None)
                sc_in = x0_hat1.detach() if self.sc_stop_grad else x0_hat1
                m = (torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) < self.sc_prob).float()
                sc_in = sc_in * m
                x0_pred = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=sc_in)
            else:
                # 非 SC：可按需要走 Drop-R/AA（一般建议关闭以保持一致）
                local_r = self.n_recursions
                if local_r > 1 and self.drop_r_prob > 0.0 and torch.rand(()) < self.drop_r_prob:
                    local_r = max(1, local_r - 1)
                if self.use_aa_train:
                    def F_step(x_r):
                        return self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x_r)
                    x0_pred = anderson_accel(
                        F_step, torch.zeros_like(x_t),
                        m=self.aa_m, lam=self.aa_lam, damping=self.aa_damp,
                        K=local_r, tol=None
                    )
                else:
                    x0_r = torch.zeros_like(x_t)
                    for _ in range(local_r):
                        x0_r = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x0_r)
                    x0_pred = x0_r

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
            loss = self.lambda_noise*noise_loss + self.lambda_post*post_loss + self.lambda_rec_loss*rec_loss

            self.log_dict(
                {"loss/noise": noise_loss, "loss/post": post_loss, "loss/rec": rec_loss, "loss/total": loss},
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
        if self.use_standard_sc:
            # 标准SC：同一 t 两次前向；第二次以 stop-grad 的第一次输出为 SC 通道（逐样本掩码）
            x0_hat1 = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=None)
            sc_in = x0_hat1.detach() if self.sc_stop_grad else x0_hat1
            m = (torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) < self.sc_prob).float()
            sc_in = sc_in * m
            x0_pred = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=sc_in)
        else:
            # Drop-R：以概率将步内递归从 n_recursions 减 1（至少为 1）
            local_r = self.n_recursions
            if local_r > 1 and self.drop_r_prob > 0.0 and random() < self.drop_r_prob:
                local_r = max(1, local_r - 1)
            if self.use_aa_train:
                def F_step(x_r):
                    return self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x_r)
                x0_pred = anderson_accel(
                    F_step, torch.zeros_like(x_t),
                    m=self.aa_m, lam=self.aa_lam, damping=self.aa_damp,
                    K=local_r, tol=None  # 训练端不早停
                )
            else:
                x0_r = torch.zeros_like(x_t)
                for _ in range(local_r):
                    x0_r = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x0_r)
                x0_pred = x0_r

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

        if self.use_standard_sc:
            x0_hat1 = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=None)
            sc_in = x0_hat1.detach() if self.sc_stop_grad else x0_hat1
            m = (torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) < self.sc_prob).float()
            sc_in = sc_in * m
            x0_pred = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=sc_in)
        else:
            local_r = self.n_recursions
            if local_r > 1 and self.drop_r_prob > 0.0 and random() < self.drop_r_prob:
                local_r = max(1, local_r - 1)
            if self.use_aa_train:
                def F_step(x_r):
                    return self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x_r)
                x0_pred = anderson_accel(
                    F_step, torch.zeros_like(x_t),
                    m=self.aa_m, lam=self.aa_lam, damping=self.aa_damp,
                    K=local_r, tol=None
                )
            else:
                x0_r = torch.zeros_like(x_t)
                for _ in range(local_r):
                    x0_r = self.generator(torch.cat((x_t.detach(), y), dim=1), t, x_r=x0_r)
                x0_pred = x0_r

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
        x0, y, _ = batch
        # 推理：把 YAML 的 anderson 配置传进去（若为 None 则回退到 env/基线）
        x0_pred = self.diffusion.sample_x0(y, self.generator, anderson=self.anderson_cfg)

        loss = F.mse_loss(x0_pred, x0)
        metrics = compute_metrics(x0, x0_pred)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_psnr", metrics["psnr_mean"], on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssim", metrics["ssim_mean"], on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.global_rank == 0:
            path = os.path.join(self.logger.log_dir, "val_samples", f"epoch_{self.current_epoch}.png")
            save_image_pair(x0, x0_pred, path)

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

    def test_step(self, batch, batch_idx):
        x0, y, slice_idx = batch
        x0_pred = self.diffusion.sample_x0(y, self.generator, anderson=self.anderson_cfg)

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

            metrics = compute_metrics(
                gt_images=target,
                pred_images=pred,
                mask=self.mask,
                subject_ids=self.subject_ids,
                report_path=os.path.join(self.logger.log_dir, "test_samples", "report.txt")
            )

            print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
            print(f"SSIM: {metrics['ssim_mean']:.2f} ± {metrics['ssim_std']:.2f}")

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
