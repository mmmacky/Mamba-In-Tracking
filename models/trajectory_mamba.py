# copied from exp1/models/trajectory_mamba.py with online step improvements
import os, sys
import torch
import torch.nn as nn
try:
    from models.mamba.mamba import Mamba, MambaConfig
except Exception:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TEST2 = os.path.join(ROOT, 'test2')
    if os.path.isdir(TEST2):
        sys.path.insert(0, TEST2)
    else:
        sys.path.insert(0, ROOT)
    from models.mamba.mamba import Mamba, MambaConfig

class TrajectoryMamba(nn.Module):
    def __init__(self, d_model=64, n_layers=4, seq_len=20):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # project 4D measurements (r, az_deg, el_deg, vr) to d_model
        self.in_proj = nn.Linear(4, d_model)

        # optional small positional embeddings - not required for SSM but harmless
        self.pos_emb = nn.Embedding(seq_len, d_model)

        mconf = MambaConfig(d_model=d_model, n_layers=n_layers, use_cuda=False)
        self.core = Mamba(mconf)

        # output head: map last time-step representation to next position (x,y,z) and vr
        self.out_head = nn.Linear(d_model, 4)

    def forward(self, meas):
        # meas: (B, L, 4)
        B, L, _ = meas.size()
        x = self.in_proj(meas)  # (B, L, d_model)
        pos_ids = torch.arange(L, device=meas.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos_ids)  # (B, L, d_model)

        out = self.core(x)  # (B, L, d_model)
        last = out[:, -1, :]
        pred = self.out_head(last)  # (B, 4)
        return pred

    def init_caches(self, batch_size, device):
        """
        Initialize caches compatible with Mamba.step: a list of (h, inputs) tuples,
        one per layer. h is set to None, inputs is zeros with shape (B, ED, d_conv-1).
        """
        mconf = self.core.config
        n_layers = mconf.n_layers
        ED = mconf.d_inner
        d_conv = mconf.d_conv
        inputs_len = max(0, d_conv - 1)
        caches = []
        for _ in range(n_layers):
            h = None
            # inputs shape: (B, ED, d_conv-1)
            inputs = torch.zeros((batch_size, ED, inputs_len), device=device)
            caches.append((h, inputs))
        return caches

    def step(self, x_step, caches, step_id=0):
        """
        Online single-step inference wrapper. Tries to call self.core.step for
        efficiency; if that fails (incompatible cache format) it falls back to
        accumulating inputs in caches['history'] and running the full forward
        on the accumulated prefix.

        Inputs:
          x_step: (B, 1, 3)
          caches: whatever was returned by init_caches (dict) or previous step
          step_id: integer time index (used for pos embedding)

        Returns:
          pred: (B,4), updated_caches
        """
        B = x_step.size(0)
        device = x_step.device
        core_in = self.in_proj(x_step).squeeze(1)  # (B, d_model)
        # position id for this step
        pos_id = torch.tensor([step_id % self.seq_len], dtype=torch.long, device=device).unsqueeze(0).expand(B,1)
        core_in = core_in + self.pos_emb(pos_id).squeeze(1)  # (B, d_model)

        # First try the efficient core.step if available
        try:
            out, new_caches = self.core.step(core_in, caches)
            # normalize output shape to (B, d_model)
            if out.dim() == 3 and out.size(1) == 1:
                out = out.squeeze(1)
            pred = self.out_head(out)
            return pred, new_caches
        except Exception:
            # fallback: use history-based cache
            if caches is None or not isinstance(caches, dict) or 'history' not in caches:
                caches = {'history': torch.zeros((B, 0, self.d_model), device=device)}

            hist = caches['history']  # (B, T, d_model)已有历史序列
            new_step = core_in.unsqueeze(1)  # (B,1,d_model)当前步输入
            hist = torch.cat([hist, new_step], dim=1)# 拼接历史与当前步，更新历史
            caches['history'] = hist

            # run full forward on accumulated history and return last output
            out_full = self.core(hist)  # (B, T+1, d_model)
            last = out_full[:, -1, :]#取最后一步输出用于预测下一步
            pred = self.out_head(last)
            return pred, caches
