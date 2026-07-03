
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIR   = Path("results/embedding")
META_DIR  = Path("results/meta")
L1000_DIR = Path("results/l1000")
REV_DIR   = Path("results/reversal")
CKPT_PATH = EMB_DIR / "vae_model.pt"

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)


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

    def reparameterise(self, mu, logvar):
        return mu

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


@torch.no_grad()
def encode_batch(model: VAE, mat: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    X = torch.tensor(mat.T, dtype=torch.float32)
    zs = []
    for i in range(0, len(X), batch_size):
        batch = X[i : i + batch_size].to(DEVICE)
        mu, _ = model.encode(batch)
        zs.append(mu.cpu().numpy())
    return np.vstack(zs)


def main() -> None:
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"{CKPT_PATH} not found — run 17_vae_embedding.py first")

    print(f"Loading checkpoint from {CKPT_PATH}  device={DEVICE}")
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    latent_dim = ck["latent_dim"]
    n_genes    = ck["n_genes"]
    gene_ids   = ck["gene_ids"]
    gene_mean  = np.array(ck["gene_mean"], dtype=np.float32)
    gene_std   = np.array(ck["gene_std"],  dtype=np.float32)
    gene_std[gene_std == 0] = 1.0

    model = VAE(n_genes=n_genes, latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"  VAE: {n_genes} genes → {latent_dim}-dim latent")

    print("\nEncoding drug signatures...")
    drug_sig_path = L1000_DIR / "drug_signatures_weighted.csv.gz"
    if not drug_sig_path.exists():
        drug_sig_path = L1000_DIR / "drug_signatures_landmark.csv.gz"
    drug_sig = pd.read_csv(drug_sig_path, index_col=0)
    drug_sig.index = drug_sig.index.astype(str)

    drug_mat = drug_sig.reindex(gene_ids).fillna(0).values.astype(np.float32)
    drug_mat_std = (drug_mat.T - gene_mean) / gene_std

    drug_emb = encode_batch(model, drug_mat_std.T)
    drug_emb_df = pd.DataFrame(
        drug_emb,
        index=drug_sig.columns,
        columns=[f"z{i}" for i in range(latent_dim)],
    )
    out_emb = EMB_DIR / "vae_drug_embeddings.csv"
    drug_emb_df.to_csv(out_emb)
    print(f"  Drug embeddings: {drug_emb_df.shape}  → {out_emb.name}")

    print("\nEncoding consensus IPF signature...")
    consensus = pd.read_csv(META_DIR / "consensus_signature.csv", index_col=0)
    consensus.index = consensus.index.astype(str)

    ipf_vec = pd.Series(0.0, index=gene_ids, dtype=np.float32)
    overlap  = consensus.index.intersection(pd.Index(gene_ids))
    ipf_vec[overlap] = consensus.loc[overlap, "meta_log2FC"].astype(np.float32)
    ipf_std = (ipf_vec.values - gene_mean) / gene_std

    with torch.no_grad():
        t = torch.tensor(ipf_std).unsqueeze(0).to(DEVICE)
        ipf_mu, _ = model.encode(t)
        ipf_emb = ipf_mu.cpu().numpy()[0]

    out_ipf = EMB_DIR / "vae_ipf_embedding.npy"
    np.save(out_ipf, ipf_emb)
    print(f"  IPF latent vector: shape={ipf_emb.shape}  norm={np.linalg.norm(ipf_emb):.4f}")
    print(f"  Consensus genes in VAE space: {len(overlap)}/{len(gene_ids)}")

    print("\nComputing VAE-TRACE reversal scores...")
    drug_vecs  = drug_emb_df.values
    ipf_norm   = np.linalg.norm(ipf_emb) + 1e-10
    drug_norms = np.linalg.norm(drug_vecs, axis=1)
    drug_norms[drug_norms == 0] = 1e-10

    vae_scores = -(drug_vecs @ ipf_emb) / (drug_norms * ipf_norm)

    vae_df = pd.DataFrame({
        "drug":      drug_emb_df.index,
        "vae_score": vae_scores,
    }).sort_values("vae_score", ascending=False).reset_index(drop=True)
    vae_df["vae_rank"] = range(1, len(vae_df) + 1)

    out_scores = EMB_DIR / "vae_trace_scores.csv"
    vae_df.to_csv(out_scores, index=False)
    print(f"  VAE-TRACE scores: {len(vae_df)} drugs  → {out_scores.name}")

    n = len(vae_df)
    print(f"\n=== Positive control ranks (VAE latent space) ===")
    for pc in ["pirfenidone", "nintedanib"]:
        rows = vae_df[vae_df["drug"].str.lower().str.contains(pc, na=False)]
        if not rows.empty:
            r = rows.iloc[0]
            print(f"  {pc:15}  VAE rank {int(r.vae_rank):4}/{n} "
                  f"({r.vae_rank/n*100:.1f}th pct)  score={r.vae_score:.4f}")
        else:
            print(f"  {pc:15}  not found in L1000")

    print(f"\n=== Top 20 VAE-TRACE candidates ===")
    print(f"{'Rank':>4}  {'Drug':30}  {'Score':>8}")
    for _, r in vae_df.head(20).iterrows():
        print(f"  {int(r.vae_rank):>4}  {r.drug:30}  {r.vae_score:>8.4f}")

    lines = ["=== VAE-TRACE encoding summary ===\n"]
    for pc in ["pirfenidone", "nintedanib"]:
        rows = vae_df[vae_df["drug"].str.lower().str.contains(pc, na=False)]
        if not rows.empty:
            r = rows.iloc[0]
            lines.append(f"  {pc}: rank {int(r.vae_rank)}/{n} ({r.vae_rank/n*100:.1f}th pct)\n")
    lines.append(f"\nTop 20:\n{vae_df.head(20).to_string(index=False)}\n")
    (EMB_DIR / "vae_training_summary.txt").write_text("".join(lines))
    print(f"\nSaved to {EMB_DIR}/")


if __name__ == "__main__":
    main()
