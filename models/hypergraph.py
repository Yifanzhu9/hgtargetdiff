"""Equivariant dynamic-hypergraph message passing for TargetDiff.

Hyperedge families
------------------
1. Ligand adaptive hyperedges (HyperACE-style):
   K learnable prototypes are adapted per molecule from the global context
   (mean/max of node features + time embedding).  A soft incidence matrix
   A (normalised over nodes within each molecule) assigns ligand atoms to
   prototypes.  Rebuilt at every diffusion step.

2. Protein residue hyperedges (PAMol-style):
   One hyperedge per residue.  A residue j belongs to hyperedge i iff the
   distance between their C-alpha atoms is below ``residue_radius`` (5 A by
   default).  Protein coordinates are fixed during diffusion, so the
   protein-protein part is built once per forward pass.

3. Protein-ligand cross-modal hyperedges:
   For every ligand atom, the k nearest protein atoms (no distance cutoff)
   are selected.  The ligand atom is added to the residue hyperedges those
   protein atoms belong to.  Rebuilt at every diffusion step because the
   ligand moves.

All membership decisions and scalar weights depend only on invariant
quantities (distances, invariant node features).  Coordinates are updated
only through equivariant relative vectors (hyperedge centroid minus atom
position), so the whole module is SE(3)-equivariant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_softmax
except ImportError:  # CPU-only environments without torch_scatter (e.g. local tests)
    def _dim_size(index, dim_size):
        return int(index.max().item()) + 1 if dim_size is None else dim_size

    def scatter_add(src, index, dim=0, dim_size=None):
        dim_size = _dim_size(index, dim_size)
        out = src.new_zeros(dim_size, *src.shape[1:])
        return out.index_add(dim, index, src)

    def scatter_max(src, index, dim=0, dim_size=None):
        dim_size = _dim_size(index, dim_size)
        fill = float('-inf') if src.is_floating_point() else torch.iinfo(src.dtype).min
        out = src.new_full((dim_size, *src.shape[1:]), fill)
        if src.dim() > index.dim():
            index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        out = out.scatter_reduce(dim, index, src, reduce='amax', include_self=True)
        return out, None

    def scatter_mean(src, index, dim=0, dim_size=None):
        dim_size = _dim_size(index, dim_size)
        out = scatter_add(src, index, dim=dim, dim_size=dim_size)
        cnt = scatter_add(torch.ones_like(index, dtype=src.dtype), index,
                          dim=dim, dim_size=dim_size)
        shape = (dim_size,) + (1,) * (src.dim() - 1)
        return out / cnt.view(shape).clamp(min=1)

    def scatter_softmax(src, index, dim=0, dim_size=None):
        dim_size = _dim_size(index, dim_size)
        max_per = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
        out = (src - max_per[index]).exp()
        denom = scatter_add(out, index, dim=dim, dim_size=dim_size)
        return out / denom[index].clamp(min=1e-12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def segment_residues_from_atom_names(atom_names, ca_name='CA'):
    """Heuristic residue segmentation from PDB atom names.

    PDB files list atoms grouped by residue and every standard residue starts
    with its backbone nitrogen (``N``).  A new residue is therefore started
    whenever an ``N`` atom follows a non-``N`` atom.  The first atom always
    starts residue 0.

    Returns
    -------
    residue_id : LongTensor (N,)
    is_ca : BoolTensor (N,)
    """
    residue_id = torch.zeros(len(atom_names), dtype=torch.long)
    is_ca = torch.zeros(len(atom_names), dtype=torch.bool)
    cur = -1
    prev = None
    for i, name in enumerate(atom_names):
        if isinstance(name, bytes):
            name = name.decode()
        if i == 0:
            cur = 0
        elif name == 'N' and prev != 'N':
            cur += 1
        residue_id[i] = cur
        is_ca[i] = (name == ca_name)
        prev = name
    return residue_id, is_ca


class RBF(nn.Module):
    """Gaussian radial basis expansion of an invariant distance."""

    def __init__(self, num_gaussians=16, r_max=10.0):
        super().__init__()
        centers = torch.linspace(0.0, r_max, num_gaussians)
        self.register_buffer('centers', centers)
        self.register_buffer('width', torch.tensor(r_max / max(num_gaussians - 1, 1)))

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return torch.exp(-((dist - self.centers) ** 2) / (2.0 * self.width ** 2))


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t, dtype=None):
        device = t.device
        if dtype is None:
            dtype = t.dtype if t.is_floating_point() else torch.float32
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = t.to(dtype)[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Adaptive ligand hyperedges (HyperACE)
# ---------------------------------------------------------------------------

class AdaptiveLigandHyperedgeGen(nn.Module):
    """Generate K soft ligand hyperedges per molecule.

    The prototypes are ``prototype_base + MLP(context)`` where the context is
    the mean/max of the node features concatenated with a time embedding.
    Inputs to the node features include the hidden invariant features and
    invariant geometric descriptors (distance to the ligand centroid, distance
    to the closest protein atom).  This is how coordinates are injected while
    keeping the construction invariant.
    """

    def __init__(self, hidden_dim, num_hyperedges=8, num_gaussians=16,
                 time_dim=32, r_max=10.0):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.rbf = RBF(num_gaussians, r_max)
        self.geo_proj = nn.Sequential(
            nn.Linear(2 * num_gaussians + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feat_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.prototype_base = nn.Parameter(torch.randn(num_hyperedges, hidden_dim) * 0.02)
        self.time_emb = SinusoidalTimeEmb(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.context_net = nn.Sequential(
            nn.Linear(2 * 2 * hidden_dim + time_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, num_hyperedges * hidden_dim),
        )
        self.scale = hidden_dim ** 0.5

    def forward(self, h, x, batch, num_graphs, time_step=None, min_prot_dist=None,
                r_max=10.0):
        """
        h : (N_lig, D) invariant ligand features
        x : (N_lig, 3) ligand coordinates
        batch : (N_lig,) molecule index
        min_prot_dist : (N_lig,) distance to the nearest protein atom
        Returns A : (N_lig, K), softmax-normalised over nodes per molecule.
        """
        if time_step is None:
            time_step = torch.zeros(num_graphs, dtype=torch.long, device=h.device)

        # Invariant geometric descriptors
        lig_center = scatter_mean(x, batch, dim=0, dim_size=num_graphs)
        d_center = (x - lig_center[batch]).norm(dim=-1)
        if min_prot_dist is None:
            min_prot_dist = torch.full_like(d_center, r_max)
        geo = torch.cat([self.rbf(d_center), self.rbf(min_prot_dist),
                         d_center.unsqueeze(-1)], dim=-1)
        feat = torch.cat([h, self.geo_proj(geo)], dim=-1)  # (N, 2D)

        # MCF-style global context + time
        mean_ctx = scatter_mean(feat, batch, dim=0, dim_size=num_graphs)
        max_ctx = scatter_max(feat, batch, dim=0, dim_size=num_graphs)[0]
        t_dtype = self.time_proj[0].weight.dtype
        t_ctx = self.time_proj(self.time_emb(time_step, dtype=t_dtype))  # (B, time_dim)
        ctx = torch.cat([mean_ctx, max_ctx, t_ctx], dim=-1)

        offsets = self.context_net(ctx).view(num_graphs, self.num_hyperedges, -1)
        prototypes = self.prototype_base.unsqueeze(0) + offsets

        # Each node is compared with the prototypes of *its own* molecule.
        logits = torch.einsum('nd,nkd->nk',
                              self.feat_proj(feat), prototypes[batch]) / self.scale

        # Normalise over nodes *within each molecule* so that every hyperedge
        # is a convex combination of its members (same convention as HyperACE).
        A = scatter_softmax(logits, batch, dim=0)
        return A


# ---------------------------------------------------------------------------
# Generic equivariant hyperedge message passing
# ---------------------------------------------------------------------------

class EquivariantHyperedgeMessage(nn.Module):
    """One round of vertex -> hyperedge -> vertex message passing.

    Scalar (invariant) messages update node features.  Vector messages update
    coordinates through ``centroid - x_i`` weighted by invariant scalars.
    """

    def __init__(self, hidden_dim, num_gaussians=16, type_dim=8, r_max=10.0):
        super().__init__()
        self.rbf = RBF(num_gaussians, r_max)
        self.type_emb = nn.Embedding(2, type_dim)
        phi_in = hidden_dim + num_gaussians + type_dim
        self.phi = nn.Sequential(
            nn.Linear(phi_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        msg_in = 2 * hidden_dim + num_gaussians
        self.beta = nn.Sequential(
            nn.Linear(msg_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.psi = nn.Sequential(
            nn.Linear(msg_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gamma = nn.Sequential(
            nn.Linear(msg_in, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h, x, member, hyperedge, weight, hyperedge_type,
                num_hyperedges, update_x_mask):
        """
        h : (N, D)
        x : (N, 3)
        member : (M,) global node indices
        hyperedge : (M,) global hyperedge indices in [0, num_hyperedges)
        weight : (M,) invariant member weights
        hyperedge_type : (num_hyperedges,) long, 0 = residue, 1 = ligand
        update_x_mask : (N,) bool, True for ligand atoms
        """
        num_nodes = h.size(0)
        if member.numel() == 0 or num_hyperedges == 0:
            return torch.zeros_like(h), torch.zeros_like(x)

        w = weight
        denom = scatter_add(w, hyperedge, dim=0, dim_size=num_hyperedges).clamp(min=1e-6)

        # Equivariant hyperedge centroid
        centroid = scatter_add(w.unsqueeze(-1) * x[member], hyperedge,
                               dim=0, dim_size=num_hyperedges) / denom.unsqueeze(-1)

        d = (x[member] - centroid[hyperedge]).norm(dim=-1)   # invariant
        rbf_d = self.rbf(d)
        type_h = self.type_emb(hyperedge_type)[hyperedge]

        # Vertex -> hyperedge
        phi = self.phi(torch.cat([h[member], rbf_d, type_h], dim=-1))
        m = scatter_add(w.unsqueeze(-1) * phi, hyperedge,
                        dim=0, dim_size=num_hyperedges) / denom.unsqueeze(-1)

        # Hyperedge -> vertex (invariant features)
        msg_in = torch.cat([h[member], m[hyperedge], rbf_d], dim=-1)
        beta = torch.sigmoid(self.beta(msg_in))
        psi = self.psi(msg_in)
        dh = scatter_add(beta * psi, member, dim=0, dim_size=num_nodes)

        # Hyperedge -> vertex (equivariant coordinates)
        gamma = self.gamma(msg_in)
        delta = beta * gamma * (centroid[hyperedge] - x[member])
        delta = delta * update_x_mask[member].unsqueeze(-1).to(delta.dtype)
        dx = scatter_add(delta, member, dim=0, dim_size=num_nodes)

        return dh, dx


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------

class HypergraphLayer(nn.Module):
    def __init__(self, hidden_dim, num_hyperedges=8, k_cross=4,
                 residue_radius=5.0, use_distance_weight=True, distance_sigma=5.0,
                 num_gaussians=16, r_max=10.0):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.k_cross = k_cross
        self.residue_radius = residue_radius
        self.use_distance_weight = use_distance_weight
        self.distance_sigma = distance_sigma
        self.r_max = r_max

        self.ligand_hyper_gen = AdaptiveLigandHyperedgeGen(
            hidden_dim, num_hyperedges, num_gaussians=num_gaussians,
            time_dim=32, r_max=r_max)
        self.residue_msg = EquivariantHyperedgeMessage(
            hidden_dim, num_gaussians=num_gaussians, r_max=r_max)
        self.ligand_msg = EquivariantHyperedgeMessage(
            hidden_dim, num_gaussians=num_gaussians, r_max=r_max)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    # -- hyperedge construction ------------------------------------------------
    def _build_residue_structure(self, prot_idx, prot_x, prot_batch,
                                 prot_resid, prot_is_ca, num_graphs):
        """Build the static protein-protein residue hypergraph."""
        device = prot_x.device
        if prot_x.numel() == 0:
            return None

        has_prot = scatter_add(torch.ones_like(prot_batch, dtype=torch.float),
                               prot_batch, dim=0, dim_size=num_graphs) > 0
        max_res = scatter_max(prot_resid, prot_batch, dim=0,
                              dim_size=num_graphs)[0]
        num_res_per_graph = torch.where(has_prot, max_res + 1,
                                        torch.zeros_like(max_res))
        resid_offset = torch.cumsum(num_res_per_graph, 0) - num_res_per_graph
        global_resid = prot_resid + resid_offset[prot_batch]
        num_res = int(num_res_per_graph.sum().item())
        if num_res == 0:
            return None

        res_batch = scatter_max(prot_batch, global_resid, dim=0,
                                dim_size=num_res)[0]

        # Residue centers: C-alpha when available, otherwise atom centroid.
        ones = torch.ones_like(global_resid, dtype=prot_x.dtype)
        res_n = scatter_add(ones, global_resid, dim=0, dim_size=num_res)
        res_center = scatter_add(prot_x, global_resid, dim=0,
                                 dim_size=num_res) / res_n.clamp(min=1).unsqueeze(-1)
        ca_mask = prot_is_ca.to(prot_x.dtype)
        ca_n = scatter_add(ca_mask, global_resid, dim=0, dim_size=num_res)
        ca_sum = scatter_add(prot_x * ca_mask.unsqueeze(-1), global_resid,
                             dim=0, dim_size=num_res)
        ca_center = ca_sum / ca_n.clamp(min=1).unsqueeze(-1)
        ca_pos = torch.where((ca_n > 0).unsqueeze(-1), ca_center, res_center)

        # Residue adjacency (C-alpha distance < radius, same molecule).
        dist = torch.cdist(ca_pos, ca_pos)
        same = res_batch.unsqueeze(0) == res_batch.unsqueeze(1)
        adj = (dist < self.residue_radius) & same
        pair_i, pair_j = adj.nonzero(as_tuple=True)  # (E,), (E,)

        # Expand residue pairs to atom members.  global_resid is non-decreasing
        # along protein atoms because PDB order is preserved.
        sorted_flag = torch.all(global_resid[1:] >= global_resid[:-1]) \
            if global_resid.numel() > 1 else torch.tensor(True, device=device)
        if not bool(sorted_flag):
            # Rare fallback: build membership with a small loop.
            mem_atom_list, mem_edge_list = [], []
            for e in range(pair_i.numel()):
                atoms = (global_resid == pair_j[e]).nonzero(as_tuple=False).squeeze(-1)
                mem_atom_list.append(prot_idx[atoms])
                mem_edge_list.append(pair_i[e].repeat(atoms.numel()))
            mem_atom = torch.cat(mem_atom_list)
            mem_edge = torch.cat(mem_edge_list)
        else:
            starts = torch.searchsorted(global_resid, pair_j, right=False)
            ends = torch.searchsorted(global_resid, pair_j, right=True)
            counts = ends - starts
            total = int(counts.sum().item())
            group = torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)
            cum = torch.cumsum(counts, 0) - counts
            within = torch.arange(total, device=device) - cum[group]
            positions = starts[group] + within
            mem_atom = prot_idx[positions]
            mem_edge = pair_i[group]

        mem_w = torch.ones(mem_atom.numel(), device=device, dtype=prot_x.dtype)
        return {
            'global_resid': global_resid,
            'res_batch': res_batch,
            'num_res': num_res,
            'mem_atom': mem_atom,
            'mem_edge': mem_edge,
            'mem_w': mem_w,
        }

    def _build_cross_modal(self, lig_idx, lig_x, lig_batch,
                           prot_x, prot_batch, global_resid, num_res,
                           num_graphs):
        """Assign every ligand atom to the residue hyperedges of its k nearest
        protein atoms.  No distance cutoff."""
        device = lig_x.device
        if num_res == 0 or lig_x.numel() == 0 or prot_x.numel() == 0:
            min_d = torch.full((lig_x.size(0),), self.r_max, device=device,
                               dtype=lig_x.dtype)
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty.to(lig_x.dtype), min_d

        dist = torch.cdist(lig_x, prot_x)  # (Nl, Np)
        same = lig_batch.unsqueeze(1) == prot_batch.unsqueeze(0)  # (Nl, Np)
        dist = dist.masked_fill(~same, 1e9)

        min_d = dist.min(dim=1).values
        min_d = torch.where(min_d < 1e8, min_d,
                            torch.full_like(min_d, self.r_max))

        k = min(self.k_cross, prot_x.size(0))
        topd, topi = dist.topk(k, dim=1, largest=False)
        valid = topd < 1e8
        top_resid = global_resid[topi]  # (Nl, k)

        if self.use_distance_weight:
            w = torch.exp(-(topd ** 2) / (2.0 * self.distance_sigma ** 2))
        else:
            w = torch.ones_like(topd)
        w = w * valid.to(w.dtype)

        lig_rep = torch.arange(lig_x.size(0), device=device).repeat_interleave(k)
        comb = lig_rep * num_res + top_resid.reshape(-1)
        comb = comb[valid.reshape(-1)]
        w_flat = w.reshape(-1)[valid.reshape(-1)]
        if comb.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty.to(lig_x.dtype), min_d

        uniq, inv = torch.unique(comb, return_inverse=True)
        w_agg = scatter_max(w_flat, inv, dim=0, dim_size=uniq.numel())[0]
        lig_local = uniq // num_res
        hedge = uniq % num_res
        member = lig_idx[lig_local]
        return member, hedge, w_agg, min_d

    def forward(self, h, x, mask_ligand, batch, num_graphs,
                protein_residue_id, protein_is_ca, time_step=None,
                fix_x=False):
        prot_idx = (~mask_ligand).nonzero(as_tuple=False).squeeze(-1)
        lig_idx = mask_ligand.nonzero(as_tuple=False).squeeze(-1)
        prot_x, lig_x = x[prot_idx], x[lig_idx]
        prot_batch, lig_batch = batch[prot_idx], batch[lig_idx]
        prot_resid = protein_residue_id
        prot_is_ca = protein_is_ca

        residue_struct = self._build_residue_structure(
            prot_idx, prot_x, prot_batch, prot_resid, prot_is_ca, num_graphs)

        dh = torch.zeros_like(h)
        dx = torch.zeros_like(x)

        # --- residue hyperedges (protein + cross-modal ligand members) ---
        if residue_struct is not None:
            num_res = residue_struct['num_res']
            global_resid = residue_struct['global_resid']
            cross_member, cross_hedge, cross_w, min_d = self._build_cross_modal(
                lig_idx, lig_x, lig_batch, prot_x, prot_batch,
                global_resid, num_res, num_graphs)
            member = torch.cat([residue_struct['mem_atom'], cross_member])
            hedge = torch.cat([residue_struct['mem_edge'], cross_hedge])
            weight = torch.cat([residue_struct['mem_w'], cross_w])
            hedge_type = torch.zeros(num_res, dtype=torch.long, device=h.device)
            dh_res, dx_res = self.residue_msg(
                h, x, member, hedge, weight, hedge_type, num_res, mask_ligand)
            dh = dh + dh_res
            dx = dx + dx_res
        else:
            min_d = torch.full((lig_x.size(0),), self.r_max, device=x.device,
                               dtype=x.dtype)

        # --- adaptive ligand hyperedges ---
        if lig_x.numel() > 0:
            A = self.ligand_hyper_gen(
                h[lig_idx], lig_x, lig_batch, num_graphs,
                time_step=time_step, min_prot_dist=min_d, r_max=self.r_max)
            K = self.num_hyperedges
            member = lig_idx.repeat_interleave(K)
            hedge = (lig_batch.unsqueeze(-1) * K +
                     torch.arange(K, device=x.device).unsqueeze(0)).reshape(-1)
            weight = A.reshape(-1)
            hedge_type = torch.ones(num_graphs * K, dtype=torch.long,
                                    device=h.device)
            dh_lig, dx_lig = self.ligand_msg(
                h, x, member, hedge, weight, hedge_type,
                num_graphs * K, mask_ligand)
            dh = dh + dh_lig
            dx = dx + dx_lig

        h = self.node_norm(h + self.node_mlp(dh))
        if not fix_x:
            x = x + dx * mask_ligand.unsqueeze(-1).to(dx.dtype)
        return h, x


class HypergraphBackbone(nn.Module):
    """Drop-in replacement for UniTransformerO2 with pure hypergraph message
    passing (no kNN pairwise graph)."""

    def __init__(self, hidden_dim=128, num_layers=9, num_ligand_hyperedges=8,
                 k_cross=4, residue_radius=5.0, use_distance_weight=True,
                 distance_sigma=5.0, num_gaussians=16, r_max=10.0,
                 **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r_max = r_max
        self.layers = nn.ModuleList([
            HypergraphLayer(
                hidden_dim=hidden_dim,
                num_hyperedges=num_ligand_hyperedges,
                k_cross=k_cross,
                residue_radius=residue_radius,
                use_distance_weight=use_distance_weight,
                distance_sigma=distance_sigma,
                num_gaussians=num_gaussians,
                r_max=r_max,
            ) for _ in range(num_layers)
        ])

    def forward(self, h, x, mask_ligand, batch, protein_residue_id=None,
                protein_is_ca=None, time_step=None, return_all=False,
                fix_x=False):
        if protein_residue_id is None:
            raise ValueError(
                'protein_residue_id is required by the hypergraph backbone. '
                'Make sure FeaturizeProteinResidue is in the transform list.')
        if protein_is_ca is None:
            protein_is_ca = torch.zeros_like(protein_residue_id, dtype=torch.bool)
        num_graphs = int(batch.max().item()) + 1

        all_h, all_x = [h], [x]
        for layer in self.layers:
            h, x = layer(h, x, mask_ligand, batch, num_graphs,
                         protein_residue_id, protein_is_ca,
                         time_step=time_step, fix_x=fix_x)
            all_h.append(h)
            all_x.append(x)

        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs
