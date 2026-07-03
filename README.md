# TRACE — Transcriptomic Reversal and Convergence Engine

TRACE is a computational drug-repurposing pipeline that identifies candidates whose transcriptomic profiles reverse a target disease signature. It was developed for idiopathic pulmonary fibrosis (IPF) but is designed to be applied to any disease with bulk RNA-seq or microarray data.

---

## How it works

```
Disease RNA-seq data
        │
        ▼
  Meta-analysis (consensus DE signature)
        │
        ▼
  Network propagation (STRING RWR)  ←── expands signal through PPI network
        │
        ▼
  L1000 reversal scoring  ←── Pearson / cosine similarity against 1,768 drug signatures
        │
        ▼
  β-VAE latent embedding  ←── unsupervised drug landscape (978 landmark genes → 128-d z)
        │
        ▼
  Ranked candidates + validation (held-out actives, FAERS, clinical trials)
```

The key insight is the **reversal criterion**: a drug scores highly when its L1000 gene-expression signature is maximally anti-correlated with the disease signature, meaning it pushes the transcriptome back toward a healthy state.

---

## Adapting TRACE to a new disease

### 1 — Collect disease RNA-seq / microarray datasets

Put raw GEO accessions in `data/raw/geo_accessions.txt`. The acquisition scripts download and parse them automatically:

```bash
python src/acquisition/01_geo_search.py        # search GEO for relevant studies
python src/acquisition/02_download_datasets.py  # download selected GEO series
python src/acquisition/03_parse_datasets.py     # harmonise to a count / expression matrix
```

### 2 — Run differential expression and meta-analysis

```bash
python src/differential_expression/05_differential_expression.py
python src/meta_analysis/06_meta_analysis.py
```

This produces `results/meta/consensus_signature.csv` — a table of genes with a meta-analysed log₂FC and adjusted p-value. This file is the central input to every downstream step.

### 3 — (Optional) Network propagation

Network propagation smooths the DE signal over the STRING protein–protein interaction network, recovering genes that are biologically connected to the disease even if they are not individually differentially expressed.

```bash
python src/embedding/07_network_propagation.py
```

### 4 — Score L1000 drug signatures

```bash
python src/aim2_reversal/08_l1000_setup.py        # download / prepare L1000 landmark matrix
python src/aim2_reversal/09_l1000_signatures.py   # build per-drug expression profiles
python src/aim2_reversal/10_reversal_scoring.py   # Pearson + cosine reversal scores
python src/aim2_reversal/11_empirical_null.py     # permutation null distribution
python src/aim2_reversal/12_prioritization_model.py  # combine scores into ranked list
```

Output: `results/benchmarking/dual_disease_scores.csv`

### 5 — Train the β-VAE embedding

The VAE maps 978-gene drug signatures to a 128-dimensional latent space. Candidates are re-ranked by cosine anti-similarity in this space (reversal in latent representation).

```bash
python src/embedding/17_vae_embedding.py   # train β-VAE on L1000 matrix
python src/embedding/19_vae_encode.py      # encode all drugs into latent z
```

Checkpoint saved to `results/embedding/vae_model.pt`. Architecture: 978 → 512 → 256 → 128 (μ/σ) → 128 (z) → 256 → 512 → 978, SiLU activations, LayerNorm.

### 6 — Validate candidates

```bash
python src/aim3_validation/15_heldout_validation.py   # held-out known actives (AUROC/AUPRC)
python src/aim3_validation/16_faers_validation.py     # pharmacovigilance signal (FAERS)
python src/aim3_validation/14_clinicaltrials.py       # cross-reference ClinicalTrials.gov
python src/aim3_validation/17_literature_corroboration.py  # PubMed evidence
```

### 7 — Visualise

```bash
python src/visualization/umap_l1000.py          # UMAP of 978-gene drug landscape
python src/visualization/trace_architecture.py  # β-VAE architecture diagram
python src/visualization/l1000_input_heatmap.py # gene-expression input heatmap
python src/figures/make_figures.py              # all paper figures
```

---

## Applying to a second disease (benchmarking example)

The `src/benchmarking/` directory shows how TRACE was benchmarked on rheumatoid arthritis (RA) as an independent held-out disease. The pattern generalises to any disease:

| Script | Purpose |
|---|---|
| `B1_ra_geo_de.py` | DE analysis on disease-specific GEO datasets |
| `B2_ra_network_propagation.py` | STRING network propagation of disease signature |
| `B3_dual_disease_scoring.py` | Simultaneous Pearson scoring for both diseases |
| `B4_ablation_dual.py` | Ablation: which pipeline components contribute most |
| `B5_benchmark_auroc.py` | AUROC / AUPRC against known actives |
| `B6_permutation_test.py` | Gene-label permutation null |
| `C1_scrna_signature.py` | scRNA-seq cell-type signature extraction |
| `C2_l2s2_query.py` | L2S2 / iLINCS cross-platform validation |
| `C3_crispr_targets.py` | CRISPR KO genetic target identification (Enrichr) |
| `C4_ra_vae.py` | Fine-tune pre-trained VAE on a second-disease signature |

To repurpose for a new disease, replace the RA inputs with your own consensus DE signature in each script. The only required file is `results/meta/consensus_signature.csv` with columns `meta_log2FC` and `meta_padj`.

---

## Requirements

```bash
conda env create -f environment.yml
conda activate trace-ipf
```

Additional packages (not in `environment.yml`):

```bash
pip install umap-learn adjustText
```

R packages for DE meta-analysis: see `r_requirements.R`.

---

## Repository layout

```
src/
├── acquisition/          GEO download and parsing
├── differential_expression/  limma / DESeq2 wrappers
├── meta_analysis/        cross-study meta-analysis
├── embedding/            STRING propagation + β-VAE
├── aim2_reversal/        L1000 reversal scoring pipeline
├── aim3_validation/      held-out, FAERS, clinical, literature
├── benchmarking/         RA second-disease benchmark
├── reversal/             sensitivity analyses
├── validation/           additional validation scripts
├── figures/              publication figure generation
└── visualization/        UMAP, VAE architecture, heatmap

results/
├── meta/                 consensus_signature.csv
├── l1000/                drug_signatures_landmark.csv.gz
├── benchmarking/         AUROC tables, ablation
├── embedding/            vae_model.pt, encoded drugs
└── figures/              all output figures (PNG + SVG)
```

---

## Citation

If you use TRACE, please cite:

> Oza N. *Transcriptomic Reversal and Convergence Engine (TRACE): a continuous-membership drug repurposing framework applied to idiopathic pulmonary fibrosis.* 2025.
