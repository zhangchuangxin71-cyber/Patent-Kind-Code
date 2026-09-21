import torch
import torch.nn as nn


class Graph_Editer(nn.Module):
    """Vectorized low-rank environment generator.

    Semantically matches the successful GPU reproduction:
    sample num_sample sources per destination from softmax(U V^T[:, j]),
    then keep/add edges with noise_level.
    """

    def __init__(self, K, n, device, rank=128):
        super(Graph_Editer, self).__init__()
        self.U = nn.Parameter(torch.FloatTensor(K, n, rank))
        self.V = nn.Parameter(torch.FloatTensor(K, n, rank))
        self.rank = rank
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(self, n, num_sample, k, edge_index, noise_level=0.5, num_user=None):
        device = self.U.device
        edge_index = edge_index.to(device)
        U_k = self.U[k]
        V_k = self.V[k]
        orig_keys = (edge_index[0].long() * n + edge_index[1].long()).unique()

        src_parts, dst_parts, log_parts = [], [], []
        chunk = 512
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            cols_t = torch.arange(start, end, device=device)
            groups = [(cols_t, torch.arange(n, device=device))]
            if num_user is not None:
                item_dst = cols_t >= int(num_user)
                groups = []
                if item_dst.any():
                    groups.append((cols_t[item_dst], torch.arange(int(num_user), device=device)))
                if (~item_dst).any():
                    groups.append((cols_t[~item_dst], torch.arange(int(num_user), n, device=device)))
            for group_cols, candidates in groups:
                if group_cols.numel() == 0 or candidates.numel() == 0:
                    continue
                probs = torch.softmax(U_k[candidates] @ V_k[group_cols].t(), dim=0)
                n_draw = min(int(num_sample), int(candidates.numel()))
                sampled_local = torch.multinomial(probs.t(), num_samples=n_draw)
                rows = candidates[sampled_local]
                cols = group_cols.unsqueeze(1).expand_as(rows)
                keys = rows.reshape(-1).long() * n + cols.reshape(-1).long()
                is_orig = torch.isin(keys, orig_keys).view_as(rows)
                rand = torch.rand_like(rows, dtype=torch.float32)
                keep = torch.where(is_orig, rand > noise_level, rand < noise_level)
                src_parts.append(rows[keep])
                dst_parts.append(cols[keep])
                log_parts.append(torch.log(torch.gather(probs.t(), 1, sampled_local) + 1e-8).sum())

        if src_parts and any(p.numel() for p in src_parts):
            new_edge_index = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)
        else:
            new_edge_index = edge_index.clone()

        log_p = torch.stack(log_parts).sum()
        return new_edge_index, log_p
