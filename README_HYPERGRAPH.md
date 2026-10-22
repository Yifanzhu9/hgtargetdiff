# TargetDiff + Dynamic Hypergraph (v1)

This repository is a research snapshot of an early TargetDiff variant that
replaces the pairwise kNN message passing with dynamic hypergraph message
passing.  It is **not** the final/best model; it is kept for reproducibility of
the v1 experiments.

## Hyperedge construction

Three hyperedge families are built at every diffusion step:

1. **Ligand adaptive hyperedges (HyperACE-style)**
   - `K = 8` learnable prototypes per molecule.
   - Prototypes are adapted from the global context (mean/max of node
     features + time embedding).
   - Soft incidence `A` is obtained by cosine/dot-product similarity followed
     by a softmax over nodes within each molecule.
   - Rebuilt at every step/layer.

2. **Protein residue hyperedges (PAMol-style)**
   - One hyperedge per residue.
   - Residue `j` belongs to hyperedge `i` iff the C-alpha distance is
     `< 5 Å` (self included).
   - Protein coordinates are fixed during diffusion, so the protein-protein
     part can be cached per sample.

3. **Protein-ligand cross-modal hyperedges**
   - For every ligand atom, its `k = 4` nearest protein atoms are selected
     (no distance cutoff).
   - The ligand atom is added to the residue hyperedges containing those
     protein atoms.
   - Rebuilt every step because the ligand moves.

All membership decisions and scalar weights depend only on invariant
quantities (distances / invariant features).  Coordinates are updated through
equivariant relative vectors (hyperedge centroid minus atom position), so the
module is SE(3)-equivariant.

## Message passing

The v1 code uses one shared equivariant hyperedge message per hyperedge
family (HyperACE-style soft assignment, weighted centroid, invariant gating).
No explicit bond prediction is used: atom types and coordinates are diffused,
and bonds are reconstructed after sampling.

## Files

- `models/hypergraph.py` — hyperedge construction + equivariant message
  passing + `HypergraphBackbone`.
- `models/molopt_score_model.py` — TargetDiff score model with
  `model_type: hypergraph`.
- `utils/transforms.py` — `FeaturizeProteinResidue` (derives residue ids and
  C-alpha flags from PDB atom names).
- `configs/training_hypergraph.yml` — v1 training config (`k_cross: 4`,
  `num_ligand_hyperedges: 8`, `residue_radius: 5`).
- `scripts/train_diffusion.py`, `scripts/sample_diffusion.py` — train/sample
  entry points.

## Environment

Python 3.10, PyTorch 2.2.2 + CUDA 12.1, PyG 2.5.3 and the matching
`torch_scatter` / `torch_cluster` wheels:

```bash
pip install -r requirements.txt
```

Evaluation dependencies (Vina / meeko / pdb2pqr) are in
`requirements_eval.txt`.

## Data

The processed CrossDocked data is not included.  Expected files:

```
data/
├── crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
└── crossdocked_pocket10_pose_split.pt
```

Both can be obtained from the TargetDiff / DiffGui data folders.  The model
config expects them under `./data/`.

## Training

```bash
python scripts/train_diffusion.py configs/training_hypergraph.yml
```

The v1 run used:

- batch size 4
- 200,000 iterations
- Adam, lr 5e-4, gradient clipping 8
- validation every 1,000 iterations
- best checkpoint: val loss **0.992** at 135k

## Sampling

```bash
python scripts/sample_diffusion.py configs/sampling.yml -i 0 \
    --result_path ./sampling_results
```

Set `model.checkpoint` in `configs/sampling.yml` to your trained checkpoint.

## v1 results (for reference)

Validation on the 100-complex CrossDocked test split:

| model | best val loss | atom AUC |
|---|---|---|
| TargetDiff baseline | 0.744 | ~0.947 |
| v1 hypergraph (this repo) | 0.992 | ~0.85 |

The pure hypergraph replacement **underperforms** the original TargetDiff
baseline.  Later experiments (k = 4/8/16, static-vs-dynamic hyperedges,
inference-time hybrid sampling) did not close the gap.  This snapshot is
therefore mainly useful as a record of the negative/ablation results.

## License

The base code follows the original TargetDiff license (`LICIENCE`).  The
hypergraph additions in this snapshot are provided for research use.
