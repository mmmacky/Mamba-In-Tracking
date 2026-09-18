import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.traj import generate_dataset
from datasets.td import TrajectoryDataset
from models.trajectory_mamba import TrajectoryMamba


def run_demo():
    # reproducible
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cpu')
    print('Running demo on device:', device)

    # small dataset for quick run
    N = 64
    T = 100
    seq_len = 20

    print('Generating dataset...')
    dataset = generate_dataset(N=N, T=T, trajectory_distribution=None,
                               dt=1.0, noise_std_r=5.0, noise_std_theta=1.0,
                               noise_std_vr=0.1, radar_pos=(0.0, 0.0))

    ds = TrajectoryDataset(dataset, seq_len=seq_len, radar_pos=(0.0, 0.0))
    batch_size = 8
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f'Dataset samples: {len(ds)}, batches: {len(loader)}')

    # small model
    model = TrajectoryMamba(d_model=32, n_layers=2, seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    pos_weight = 1.0
    vr_weight = 0.1

    # one epoch
    model.train()
    total_loss = 0.0
    total_pos = 0.0
    total_vr = 0.0
    total_samples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        pred_pos = pred[:, :3]
        pred_vr = pred[:, 3:4]
        yb_pos = yb[:, :3]
        yb_vr = yb[:, 3:4]

        loss_pos = torch.nn.functional.mse_loss(pred_pos, yb_pos)
        loss_vr = torch.nn.functional.mse_loss(pred_vr, yb_vr)
        loss = pos_weight * loss_pos + vr_weight * loss_vr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_n = xb.size(0)
        total_loss += loss.item() * batch_n
        total_pos += loss_pos.item() * batch_n
        total_vr += loss_vr.item() * batch_n
        total_samples += batch_n

    avg_loss = total_loss / total_samples
    avg_pos_rmse = np.sqrt(total_pos / total_samples)
    avg_vr_rmse = np.sqrt(total_vr / total_samples)

    print(f'One-epoch training finished. Avg loss: {avg_loss:.6f}, pos RMSE: {avg_pos_rmse:.3f} m, vr RMSE: {avg_vr_rmse:.3f} m/s')

    # show a batch of predictions
    model.eval()
    xb, yb = next(iter(loader))
    with torch.no_grad():
        pred = model(xb)
    print('Example predictions (first 4):')
    for i in range(min(4, xb.size(0))):
        p = pred[i].cpu().numpy()
        t = yb[i].cpu().numpy()
        print(f' pred_pos: {p[:3].round(3)} pred_vr: {p[3]:.3f} | true_pos: {t[:3].round(3)} true_vr: {t[3]:.3f}')

    # save small checkpoint
    torch.save(model.state_dict(), 'demo_model_cpu.pt')
    print('Saved demo_model_cpu.pt')


if __name__ == "__main__":
    run_demo()
