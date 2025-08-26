# anderson.py
import torch

@torch.no_grad()
def anderson_accel(
    F,
    x0,
    m: int = 3,
    lam: float = 1e-3,
    damping: float = 0.2,
    K: int = 3,
    tol: float | None = None,
    safeguard: bool = True,
):
    """
    Anderson (Type-I) with safeguard for fixed-point iteration x = F(x).
    - F: mapping (B,C,H,W) -> (B,C,H,W), no grad
    - x0: initial iterate
    - m: history size (2~5)
    - lam: Tikhonov regularization
    - damping: mix between accelerated step and plain step (0~1)
    - K: max inner iterations
    - tol: optional early-stop on ||F(x)-x||_1 (batch-mean)
    - safeguard: if new residual not smaller, fallback to plain step
    """
    xk = x0.clone()
    X_hist = []   # x_k
    F_hist = []   # f_k = F(x_k)
    B = xk.size(0)

    eye_cache = {}

    def flat(v):   # (B,C,H,W) -> (B, D)
        return v.reshape(B, -1)

    def combine_cols(coeff, cols):  # coeff: (B,M), cols: list of M tensors (B,C,H,W)
        M = len(cols)
        stack = torch.stack(cols, dim=1)          # (B,M,C,H,W)
        w = coeff.view(B, M, 1, 1, 1)
        return (w * stack).sum(dim=1)             # (B,C,H,W)

    for k in range(K):
        fk = F(xk)                                # plain one step
        gk = fk - xk                              # residual

        # early stop (eval only)
        if tol is not None:
            if bool(flat(gk).abs().mean(dim=1).lt(tol).all()):
                return fk

        # push history (cap m)
        if len(X_hist) == m:
            X_hist.pop(0); F_hist.pop(0)
        X_hist.append(xk)
        F_hist.append(fk)

        if len(X_hist) == 1:
            x_next = fk
        else:
            # Type-I Anderson:
            # Build ΔF = [f_i - f_{i-1}]_{i=1..M-1}, ΔX = [x_i - x_{i-1}]_{i=1..M-1}
            M = len(X_hist)
            dF_cols, dX_cols = [], []
            for i in range(1, M):
                dF_cols.append((F_hist[i] - F_hist[i-1]).reshape(B, -1))
                dX_cols.append( F_hist[i] -  X_hist[i])   # will combine later in tensor form
            # ΔF: (B, D, M-1)
            dF = torch.stack(dF_cols, dim=2)
            # Solve per-sample:  (ΔF ΔF^T + lam I) γ = g_k
            # γ shape: (B, M-1, 1)
            BT = dF.transpose(1, 2)               # (B, M-1, D)
            A  = BT @ dF                           # (B, M-1, M-1)
            rhs = BT @ flat(gk).unsqueeze(-1)      # (B, M-1, 1)
            gammas = []
            for b in range(B):
                Mb1 = A[b].size(0)
                if Mb1 not in eye_cache:
                    eye_cache[Mb1] = torch.eye(Mb1, device=A.device, dtype=A.dtype)
                Ab = A[b] + lam * eye_cache[Mb1]
                gb = torch.linalg.solve(Ab, rhs[b])  # (M-1,1)
                gammas.append(gb.squeeze(-1))
            gamma = torch.stack(gammas, dim=0)     # (B, M-1)

            # accelerated candidate: x_acc = f_k - ΔX γ
            # build ΔX as list (M-1) of (B,C,H,W), then weighted sum by gamma
            dX_list = []
            for i in range(1, M):
                dX_list.append(F_hist[i] - X_hist[i])  # (B,C,H,W)
            x_acc = fk - combine_cols(gamma, dX_list)

            # damping between accelerated and plain step
            x_next = (1.0 - damping) * fk + damping * x_acc

            # safeguard: ensure residual decrease, else fallback to fk
            if safeguard:
                r_new = flat(F(x_next) - x_next).abs().mean(dim=1)
                r_old = flat(gk).abs().mean(dim=1)
                # if any sample got worse, fallback that sample to fk
                mask_worse = (r_new > r_old).view(B, *([1]*(xk.ndim-1)))
                x_next = torch.where(mask_worse, fk, x_next)

        xk = x_next

    return xk
