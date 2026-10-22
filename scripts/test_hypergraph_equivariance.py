"""Smoke tests for the dynamic hypergraph backbone.

Checks
------
1. Forward/backward run on a variable-size batch.
2. SE(3) equivariance:
   - invariant features h are unchanged under rotation/translation;
   - ligand coordinates transform as R x + t;
   - protein coordinates are left untouched.

Run with:
    PYTHONPATH=. python scripts/test_hypergraph_equivariance.py
"""

import torch

from models.hypergraph import HypergraphBackbone


def make_batch(dtype=torch.float64):
    torch.manual_seed(0)
    n_residues = [6, 5]
    n_ligand = [15, 12]
    atoms_per_residue = 4

    protein_atoms, residue_ids, is_ca, batch = [], [], [], []
    offset = 0
    for b, n_res in enumerate(n_residues):
        for r in range(n_res):
            for a in range(atoms_per_residue):
                protein_atoms.append(offset + r * atoms_per_residue + a)
                residue_ids.append(r)
                is_ca.append(a == 0)
                batch.append(b)
    n_protein = len(protein_atoms)

    lig_ids, lig_batch = [], []
    for b, n_lig in enumerate(n_ligand):
        for i in range(n_lig):
            lig_ids.append(n_protein + sum(n_ligand[:b]) + i)
            lig_batch.append(b)

    n_total = n_protein + sum(n_ligand)
    mask_ligand = torch.zeros(n_total, dtype=torch.bool)
    mask_ligand[torch.tensor(lig_ids)] = True
    batch_all = torch.tensor(batch + lig_batch, dtype=torch.long)

    x = torch.randn(n_total, 3, dtype=dtype)
    h = torch.randn(n_total, 32, dtype=dtype)
    protein_residue_id = torch.tensor(residue_ids, dtype=torch.long)
    protein_is_ca = torch.tensor(is_ca, dtype=torch.bool)
    time_step = torch.tensor([100, 500], dtype=torch.long)
    return h, x, mask_ligand, batch_all, protein_residue_id, protein_is_ca, time_step


def random_rotation(dtype):
    a = torch.randn(3, 3, dtype=dtype)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def main():
    dtype = torch.float64
    h, x, mask_ligand, batch, resid, is_ca, t = make_batch(dtype)

    model = HypergraphBackbone(
        hidden_dim=32, num_layers=2, num_ligand_hyperedges=4,
        k_cross=3, residue_radius=5.0, use_distance_weight=True,
        distance_sigma=5.0, num_gaussians=8, r_max=10.0,
    ).to(dtype)

    ligand_v = torch.randint(0, 13, (int(mask_ligand.sum()),), dtype=torch.long)
    out = model(h, x, mask_ligand, batch,
                protein_residue_id=resid, protein_is_ca=is_ca,
                time_step=t, return_all=True, ligand_v=ligand_v)
    assert out['x'].shape == x.shape
    assert out['h'].shape == h.shape
    assert len(out['all_x']) == 3 and len(out['all_h']) == 3
    print('[ok] forward shapes:', out['x'].shape, out['h'].shape)

    # --- SE(3) equivariance ---
    q = random_rotation(dtype)
    trans = torch.randn(3, dtype=dtype)
    x_rot = x @ q.transpose(0, 1) + trans

    out_rot = model(h, x_rot, mask_ligand, batch,
                    protein_residue_id=resid, protein_is_ca=is_ca,
                    time_step=t, ligand_v=ligand_v)

    # Invariant features must not change.
    h_err = (out['h'] - out_rot['h']).abs().max().item()
    # Protein coordinates must stay fixed (compare each output with its own input).
    prot_err = max(
        (out['x'][~mask_ligand] - x[~mask_ligand]).abs().max().item(),
        (out_rot['x'][~mask_ligand] - x_rot[~mask_ligand]).abs().max().item(),
    )
    # Ligand coordinates must transform equivariantly.
    expected_lig = out['x'][mask_ligand] @ q.transpose(0, 1) + trans
    lig_err = (expected_lig - out_rot['x'][mask_ligand]).abs().max().item()

    print(f'[equiv] max h error     : {h_err:.3e}')
    print(f'[equiv] max protein error: {prot_err:.3e}')
    print(f'[equiv] max ligand error : {lig_err:.3e}')
    assert h_err < 1e-8, 'invariant features are not invariant'
    assert prot_err < 1e-10, 'protein coordinates moved'
    assert lig_err < 1e-8, 'ligand coordinates are not equivariant'

    # --- backward ---
    loss = out['h'].sum() + out['x'].sum()
    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    print(f'[ok] backward: {n_grad}/{sum(1 for _ in model.parameters())} params have grads')
    assert n_grad > 0

    print('ALL_HYPERGRAPH_TESTS_PASSED')


if __name__ == '__main__':
    main()
