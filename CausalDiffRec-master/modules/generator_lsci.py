import torch
import torch.nn as nn


class Graph_Editer_LSCI(nn.Module):
    """
    Environment generator that only edits variant edges; causal edges stay fixed.
    Low-rank factorization for memory (same spirit as Baseline GPU generator).
    """

    def __init__(self, K, n, device, rank=128):
        super().__init__()
        self.U = nn.Parameter(torch.FloatTensor(K, n, rank))
        self.V = nn.Parameter(torch.FloatTensor(K, n, rank))
        self.rank = rank
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(
        self,
        num_nodes: int,
        num_sample: int,
        k: int,
        causal_edge_index: torch.Tensor,
        variant_edge_index: torch.Tensor,
        noise_level: float = 0.5,
        num_user: int = None,
    ):
        device = self.U.device
        causal_edge_index = causal_edge_index.to(device)
        variant_edge_index = variant_edge_index.to(device)

        U_k = self.U[k]
        V_k = self.V[k]

        if variant_edge_index.numel() == 0:
            # nothing to edit
            log_p = torch.zeros((), device=device, requires_grad=True)
            return causal_edge_index.clone(), log_p

        original = torch.cat([causal_edge_index, variant_edge_index], dim=1)
        orig_keys = (original[0].long() * num_nodes + original[1].long()).unique()

        # Sample candidate edits only among destinations that appear in variant edges
        variant_dst = variant_edge_index[1].unique()
        src_parts, dst_parts, log_parts = [], [], []
        chunk = 512
        dst_list = variant_dst.tolist()
        for start in range(0, len(dst_list), chunk):
            cols_idx = dst_list[start:start + chunk]
            cols_t = torch.tensor(cols_idx, device=device, dtype=torch.long)
            groups = [(cols_t, torch.arange(num_nodes, device=device))]
            if num_user is not None:
                item_dst = cols_t >= int(num_user)
                groups = []
                if item_dst.any():
                    groups.append((cols_t[item_dst], torch.arange(int(num_user), device=device)))
                if (~item_dst).any():
                    groups.append((cols_t[~item_dst], torch.arange(int(num_user), num_nodes, device=device)))
            for group_cols, candidates in groups:
                if group_cols.numel() == 0 or candidates.numel() == 0:
                    continue
                probs = torch.softmax(U_k[candidates] @ V_k[group_cols].t(), dim=0)
                n_draw = min(int(num_sample), int(candidates.numel()))
                sampled_local = torch.multinomial(probs.t(), num_samples=n_draw)
                rows = candidates[sampled_local]
                cols = group_cols.unsqueeze(1).expand_as(rows)
                keys = rows.reshape(-1).long() * num_nodes + cols.reshape(-1).long()
                is_orig = torch.isin(keys, orig_keys).view_as(rows)
                rand = torch.rand_like(rows, dtype=torch.float32)
                keep = torch.where(is_orig, rand > noise_level, rand < noise_level)
                src_parts.append(rows[keep])
                dst_parts.append(cols[keep])
                log_parts.append(torch.log(torch.gather(probs.t(), 1, sampled_local) + 1e-8).sum())

        if src_parts and any(p.numel() for p in src_parts):
            edited_variant = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)
        else:
            edited_variant = variant_edge_index

        generated = torch.cat([causal_edge_index, edited_variant], dim=1)
        log_p = torch.stack(log_parts).sum() if log_parts else torch.zeros((), device=device)
        return generated, log_p
