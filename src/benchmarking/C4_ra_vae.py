
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

ROOT   = Path(__file__).resolve().parents[2]
EMB    = ROOT / "results/embedding"
VAL    = ROOT / "results/validation"
L1K    = ROOT / "results/l1000"
BENCH  = ROOT / "results/benchmarking"
ACT    = ROOT / "data/known_actives"
BENCH.mkdir(parents=True, exist_ok=True)


class BVAE(nn.Module):
    def __init__(self, gene_dim: int, latent_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
        )
        self.mu_head  = nn.Linear(hidden_dim // 2, latent_dim)
        self.logv_head = nn.Linear(hidden_dim // 2, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, gene_dim),
        )

    def encode(self, x):
        h = self.enc(x); return self.mu_head(h), self.logv_head(h)

    def reparameterize(self, mu, logv):
        std = torch.exp(0.5 * logv.clamp(-10, 10))
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparameterize(mu, logv)
        return self.dec(z), mu, logv


def load_state_dict_compatible(model: BVAE, path: Path) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    model_sd = model.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in model_sd and model_sd[k].shape == v.shape}
    model_sd.update(matched)
    model.load_state_dict(model_sd)
    print(f"  Loaded {len(matched)}/{len(sd)} checkpoint tensors from {path.name}")


def ra_contrastive_loss(drug_latents: torch.Tensor, disease_latent: torch.Tensor,
                        is_active: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    sim = F.cosine_similarity(drug_latents, disease_latent.unsqueeze(0).expand_as(drug_latents))
    pos_loss = (1 - sim[is_active.bool()]).clamp(min=0).mean() if is_active.any() else torch.tensor(0.0)
    neg_loss = (sim[~is_active.bool()] - (1 - margin)).clamp(min=0).mean() if (~is_active.bool()).any() else torch.tensor(0.0)
    return pos_loss + neg_loss


def elbo(x_hat, x, mu, logv, beta: float = 1.0):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kld   = -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()
    return recon + beta * kld, recon, kld


def load_l1000_matrix() -> tuple[pd.DataFrame, np.ndarray]:
    df   = pd.read_csv(L1K / "drug_signatures_landmark.csv.gz", index_col=0)
    mat  = df.values.astype(np.float32)
    mean = mat.mean(axis=1, keepdims=True); std = mat.std(axis=1, keepdims=True) + 1e-8
    mat  = (mat - mean) / std
    return df, mat


def compute_ra_drug_scores(model: BVAE, gene_idx: np.ndarray,
                           drug_mat: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        drug_t = torch.tensor(drug_mat[gene_idx].T)
        mu_d, _= model.encode(drug_t)
        return -F.cosine_similarity(mu_d, mu_d.mean(dim=0, keepdim=True)).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain even if vae_ra_model.pt exists")
    args = parser.parse_args()

    ra_model_path = EMB / "vae_ra_model.pt"
    ipf_model_path = EMB / "vae_model.pt"

    if not ipf_model_path.exists():
        print(f"ERROR: Pre-trained IPF VAE not found at {ipf_model_path}. Run Step 5 first.")
        return

    print("Loading L1000 drug matrix ...")
    drug_df, drug_mat = load_l1000_matrix()
    n_genes, n_drugs = drug_mat.shape
    print(f"  {n_genes} landmark genes × {n_drugs} drugs")

    ra_sig_path = VAL / "ra_consensus_signature.csv"
    if not ra_sig_path.exists():
        print(f"  RA signature not found at {ra_sig_path}; using RA z-score proxy ...")
        ra_trace = pd.read_csv(VAL / "ra_trace_scores.csv", index_col="drug")
        best_drug = ra_trace["net_trace"].idxmax()
        drug_idx  = list(drug_df.columns).index(best_drug) if best_drug in drug_df.columns else 0
        ra_sig_vec = drug_mat[:, drug_idx]
    else:
        ra_sig = pd.read_csv(ra_sig_path, index_col=0)
        gene_ids = [str(g) for g in drug_df.index]
        lfc_series = ra_sig["meta_log2FC"].astype(float)
        ra_sig_vec = np.array([float(lfc_series.get(g, 0)) for g in gene_ids], dtype=np.float32)
        ra_sig_vec = (ra_sig_vec - ra_sig_vec.mean()) / (ra_sig_vec.std() + 1e-8)

    act_path = ACT / "ra_actives.txt"
    ra_actives = set()
    if act_path.exists():
        ra_actives = {l.strip().lower() for l in act_path.read_text().splitlines() if l.strip()}
    drug_names = [d.lower() for d in drug_df.columns]
    is_active  = np.array([1 if d in ra_actives else 0 for d in drug_names], dtype=np.float32)
    print(f"  RA actives in L1000 matrix: {int(is_active.sum())}")

    ck = torch.load(ipf_model_path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model_state" in ck:
        gene_dim_ck = int(ck.get("n_genes", n_genes))
        latent_dim  = int(ck.get("latent_dim", 64))
        sd = ck["model_state"]
    else:
        sd = ck
        gene_dim_ck = None
        for k, v in sd.items():
            if "enc.0.weight" in k: gene_dim_ck = v.shape[1]; break
        if gene_dim_ck is None: gene_dim_ck = n_genes
        latent_dim = sd.get("mu_head.bias", torch.zeros(64)).shape[0] if "mu_head.bias" in sd else 64
    print(f"  Checkpoint architecture: gene_dim={gene_dim_ck}, latent_dim={latent_dim}")

    if gene_dim_ck != n_genes:
        drug_mat_in = drug_mat[:gene_dim_ck]
        ra_sig_vec  = ra_sig_vec[:gene_dim_ck]
    else:
        drug_mat_in = drug_mat

    model = BVAE(gene_dim=gene_dim_ck, latent_dim=latent_dim)

    if ra_model_path.exists() and not args.retrain:
        print(f"\n[skip] {ra_model_path.name} exists; loading for scoring. Pass --retrain to redo.")
        load_state_dict_compatible(model, ra_model_path)
    else:
        load_state_dict_compatible(model, ipf_model_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        X = torch.tensor(drug_mat_in.T, dtype=torch.float32)
        ra_t = torch.tensor(ra_sig_vec, dtype=torch.float32).unsqueeze(0)
        act_t = torch.tensor(is_active, dtype=torch.float32)
        ds = TensorDataset(X, act_t); loader = DataLoader(ds, batch_size=256, shuffle=True)

        print("\nFine-tuning β-VAE for RA (10 epochs) ...")
        LAMBDA = 0.1
        for epoch in range(1, 11):
            model.train(); tot_loss = 0.0; n_batch = 0
            for x_batch, act_batch in loader:
                x_hat, mu, logv = model(x_batch)
                e, recon, kld = elbo(x_hat, x_batch, mu, logv, beta=0.5)

                with torch.no_grad():
                    ra_mu, _ = model.encode(ra_t)
                cont = ra_contrastive_loss(mu, ra_mu.squeeze(0), act_batch)
                loss = e + LAMBDA * cont

                optimizer.zero_grad(); loss.backward(); optimizer.step()
                tot_loss += loss.item(); n_batch += 1
            scheduler.step()
            print(f"  Epoch {epoch:2d}: loss={tot_loss/n_batch:.4f}")

        torch.save(model.state_dict(), ra_model_path)
        print(f"  Saved fine-tuned model → {ra_model_path.name}")

    print("\nScoring drugs in RA-fine-tuned latent space ...")
    model.eval()
    with torch.no_grad():
        X_all  = torch.tensor(drug_mat_in.T, dtype=torch.float32)
        ra_t   = torch.tensor(ra_sig_vec, dtype=torch.float32).unsqueeze(0)
        mu_all, _ = model.encode(X_all)
        ra_mu, _  = model.encode(ra_t)
        sim = F.cosine_similarity(mu_all, ra_mu.expand_as(mu_all)).cpu().numpy()
        vae_ra_scores = -sim

    score_df = pd.DataFrame({"drug": drug_df.columns, "vae_ra_score": vae_ra_scores})
    score_df["vae_ra_rank"] = score_df["vae_ra_score"].rank(ascending=False).astype(int)
    score_df = score_df.sort_values("vae_ra_rank")
    score_df.to_csv(EMB / "vae_ra_trace_scores.csv", index=False)

    n_act = int(is_active.sum())
    if n_act >= 2:
        auroc = float(roc_auc_score(is_active, vae_ra_scores))
        ap    = float(average_precision_score(is_active, vae_ra_scores))
        rand  = n_act / n_drugs
    else:
        auroc = ap = rand = float("nan")
        print(f"  Only {n_act} RA actives in L1000 — cannot compute AUROC/AUPRC")

    abl_rows = []
    b5_csv = BENCH / "auroc_summary.csv"
    if b5_csv.exists():
        b5 = pd.read_csv(b5_csv)
        ra_rows = b5[b5["disease"] == "RA"]
        for _, r in ra_rows.iterrows():
            abl_rows.append({
                "arm": r["arm"], "auroc": r["auroc"], "auprc": r.get("auprc", float("nan")),
                "auprc_fold_over_random": r.get("auprc_fold_over_random", float("nan")),
            })
    abl_rows.append({
        "arm": "VAE-RA (fine-tuned)",
        "auroc": round(auroc, 4) if not np.isnan(auroc) else np.nan,
        "auprc": round(ap, 6) if not np.isnan(ap) else np.nan,
        "auprc_fold_over_random": round(ap / rand, 2) if (not np.isnan(ap) and rand > 0) else np.nan,
    })
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(BENCH / "ablation_ra_with_vae.csv", index=False)

    print("\n=== RA Ablation — Existing vs Fine-Tuned VAE ===")
    for _, r in abl_df.iterrows():
        print(f"  {r['arm']:35} AUROC={r['auroc']:.4f}  "
              f"AUPRC={r['auprc']:.5f} ({r['auprc_fold_over_random']:.1f}× random)")

    priority = ["tofacitinib", "baricitinib", "leflunomide", "methotrexate"]
    top200 = set(score_df.head(200)["drug"].str.lower())
    print("\nRA priority drugs in VAE-RA top-200:")
    for d in priority:
        rank = score_df[score_df["drug"].str.lower() == d]["vae_ra_rank"].values
        print(f"  {d}: rank {int(rank[0])}/{n_drugs}" if len(rank) else f"  {d}: not in matrix")

    print(f"\nC4 complete. vae_ra_trace_scores.csv and ablation_ra_with_vae.csv written.")


if __name__ == "__main__":
    main()
