
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMB_DIR   = Path("results/embedding")
META_DIR  = Path("results/meta")
L1000_DIR = Path("results/l1000")
REV_DIR   = Path("results/reversal")
EMB_DIR.mkdir(exist_ok=True)

LATENT_DIM  = int(sys.argv[sys.argv.index("--latent-dim") + 1]) if "--latent-dim" in sys.argv else 128
N_EPOCHS    = int(sys.argv[sys.argv.index("--epochs") + 1])     if "--epochs"    in sys.argv else 50
BATCH_SIZE  = 512
LR          = 1e-3
BETA        = 0.5
LAMBDA_CON  = 0.1
TEMPERATURE = 0.1

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"Device: {DEVICE}")


class L1000Dataset(Dataset):
    def __init__(self, mat: np.ndarray, drug_labels: np.ndarray):
        self.X = torch.tensor(mat, dtype=torch.float32)
        self.y = torch.tensor(drug_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class VAE(nn.Module):
    def __init__(self, n_genes: int = 978, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Linear(n_genes, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256),     nn.LayerNorm(256), nn.GELU(),
        )
        self.mu_head     = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 512),        nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, n_genes),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = BETA) -> tuple[torch.Tensor, dict]:
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, {"recon": recon_loss.item(), "kl": kl_loss.item()}


def nt_xent_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                 temperature: float = TEMPERATURE) -> torch.Tensor:
    n = embeddings.size(0)
    if n < 2:
        return torch.tensor(0.0, device=embeddings.device)

    z = F.normalize(embeddings, dim=1)
    sim = torch.mm(z, z.T) / temperature
    sim.fill_diagonal_(-1e9)

    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    label_eq.fill_diagonal_(False)

    if not label_eq.any():
        return torch.tensor(0.0, device=embeddings.device)

    loss = 0.0
    count = 0
    for i in range(n):
        pos_mask = label_eq[i]
        if not pos_mask.any():
            continue
        logits = sim[i]
        log_probs = logits - torch.logsumexp(logits, dim=0)
        loss -= log_probs[pos_mask].mean()
        count += 1

    return loss / max(count, 1)


def train(model: VAE, loader: DataLoader, optimizer: torch.optim.Optimizer,
          epoch: int) -> dict:
    model.train()
    total_loss = recon_sum = kl_sum = con_sum = 0.0
    n_batches = 0

    for x, drug_labels in loader:
        x, drug_labels = x.to(DEVICE), drug_labels.to(DEVICE)
        optimizer.zero_grad()

        recon, mu, logvar = model(x)
        loss_vae, info = vae_loss(recon, x, mu, logvar)
        loss_con = nt_xent_loss(mu, drug_labels) * LAMBDA_CON

        loss = loss_vae + loss_con
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        recon_sum  += info["recon"]
        kl_sum     += info["kl"]
        con_sum    += loss_con.item()
        n_batches  += 1

    return {
        "total": total_loss / n_batches,
        "recon": recon_sum  / n_batches,
        "kl":    kl_sum     / n_batches,
        "con":   con_sum    / n_batches,
    }


@torch.no_grad()
def encode_matrix(model: VAE, mat: np.ndarray,
                  batch_size: int = 1024) -> np.ndarray:
    model.eval()
    X = torch.tensor(mat.T, dtype=torch.float32)
    zs = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size].to(DEVICE)
        mu, _ = model.encode(batch)
        zs.append(mu.cpu().numpy())
    return np.vstack(zs)


def main() -> None:
    print("Loading L1000 signatures...")
    parquet_path = Path("data/raw/l1000/all_signatures_landmark.parquet")
    mat_df = pd.read_parquet(parquet_path)
    sig_info = pd.read_csv(L1000_DIR / "sm_sig_info.csv", low_memory=False)
    sig_info = sig_info.set_index("sig_id")

    n_genes, n_sigs = mat_df.shape
    print(f"  {n_sigs:,} signatures × {n_genes} genes")

    drug_names   = sig_info.loc[mat_df.columns, "pert_iname"].values
    unique_drugs = {d: i for i, d in enumerate(sorted(set(drug_names)))}
    drug_labels  = np.array([unique_drugs[d] for d in drug_names])

    mat = mat_df.values.T.astype(np.float32)
    gene_mean = mat.mean(axis=0)
    gene_std  = mat.std(axis=0)
    gene_std[gene_std == 0] = 1.0
    mat = (mat - gene_mean) / gene_std

    dataset  = L1000Dataset(mat, drug_labels)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE.type != "mps"))

    model     = VAE(n_genes=n_genes, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} parameters  latent_dim={LATENT_DIM}")
    print(f"  Training for {N_EPOCHS} epochs on {DEVICE}...")

    history = []
    best_loss = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        metrics = train(model, loader, optimizer, epoch)
        scheduler.step()
        history.append(metrics)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
                  f"total={metrics['total']:.4f}  "
                  f"recon={metrics['recon']:.4f}  "
                  f"kl={metrics['kl']:.4f}  "
                  f"con={metrics['con']:.4f}")

        if metrics["total"] < best_loss:
            best_loss = metrics["total"]
            torch.save({
                "model_state": model.state_dict(),
                "gene_mean":   gene_mean,
                "gene_std":    gene_std,
                "latent_dim":  LATENT_DIM,
                "n_genes":     n_genes,
                "gene_ids":    mat_df.index.tolist(),
            }, EMB_DIR / "vae_model.pt")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, label in [
        (axes[0], "total", "Total loss"),
        (axes[1], "recon", "Reconstruction loss"),
    ]:
        ax.plot([h[key] for h in history])
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.set_title(label)
    fig.tight_layout()
    fig.savefig(EMB_DIR / "vae_training_curve.png", dpi=150)
    plt.close(fig)

    print("\nEncoding all drug signatures into VAE latent space...")
    checkpoint = torch.load(EMB_DIR / "vae_model.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    drug_sig_mat = pd.read_csv(L1000_DIR / "drug_signatures_landmark.csv.gz", index_col=0)
    drug_sig_mat.index = drug_sig_mat.index.astype(str)

    gene_ids_train = checkpoint["gene_ids"]
    drug_mat_ordered = drug_sig_mat.reindex(gene_ids_train).fillna(0).values.T
    drug_mat_std = (drug_mat_ordered - gene_mean) / gene_std

    drug_embeddings = encode_matrix(model, drug_mat_std.T)
    drug_emb_df = pd.DataFrame(
        drug_embeddings,
        index=drug_sig_mat.columns,
        columns=[f"z{i}" for i in range(LATENT_DIM)],
    )
    drug_emb_df.to_csv(EMB_DIR / "vae_drug_embeddings.csv")
    print(f"  Drug embeddings: {drug_emb_df.shape}")

    print("Encoding consensus IPF signature...")
    consensus = pd.read_csv(META_DIR / "consensus_signature.csv", index_col=0)
    consensus.index = consensus.index.astype(str)

    ipf_vec = pd.Series(0.0, index=gene_ids_train)
    overlap = consensus.index.intersection(pd.Index(gene_ids_train))
    ipf_vec[overlap] = consensus.loc[overlap, "meta_log2FC"]
    ipf_vec_std = ((ipf_vec.values - gene_mean) / gene_std).astype(np.float32)

    model.eval()
    with torch.no_grad():
        ipf_tensor = torch.tensor(ipf_vec_std).unsqueeze(0).to(DEVICE)
        ipf_mu, _ = model.encode(ipf_tensor)
        ipf_emb = ipf_mu.cpu().numpy()[0]

    np.save(EMB_DIR / "vae_ipf_embedding.npy", ipf_emb)
    print(f"  IPF latent vector: shape={ipf_emb.shape}  norm={np.linalg.norm(ipf_emb):.4f}")

    print("Computing VAE-TRACE reversal scores...")
    drug_vecs = drug_emb_df.values
    ipf_norm  = np.linalg.norm(ipf_emb)
    drug_norms = np.linalg.norm(drug_vecs, axis=1)
    drug_norms[drug_norms == 0] = 1e-10

    vae_scores = -(drug_vecs @ ipf_emb) / (drug_norms * ipf_norm)

    vae_df = pd.DataFrame({
        "drug":       drug_emb_df.index,
        "vae_score":  vae_scores,
    }).sort_values("vae_score", ascending=False)
    vae_df["vae_rank"] = range(1, len(vae_df) + 1)
    vae_df.to_csv(EMB_DIR / "vae_trace_scores.csv", index=False)

    n = len(vae_df)
    print(f"\n=== Positive control ranks (VAE latent space) ===")
    for pc in ["pirfenidone", "nintedanib"]:
        rows = vae_df[vae_df["drug"].str.lower().str.contains(pc.lower(), na=False)]
        if not rows.empty:
            r = rows.iloc[0]
            print(f"  {pc:15}  VAE rank {int(r['vae_rank']):4}/{n} "
                  f"({r['vae_rank']/n*100:.1f}th pct)  score={r['vae_score']:.4f}")

    print(f"\n=== Top 20 VAE-TRACE candidates ===")
    print(f"{'Rank':>4}  {'Drug':30}  {'VAE score':>10}")
    for _, r in vae_df.head(20).iterrows():
        print(f"  {int(r['vae_rank']):>4}  {r['drug']:30}  {r['vae_score']:>10.4f}")

    print(f"\nOutputs saved to results/embedding/")
    print("Next: GTEx/CCLE cell-line weighting, then ablation vs. network propagation.")


if __name__ == "__main__":
    main()
