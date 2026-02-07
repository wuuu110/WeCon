import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _safe_multinomial(weights_2d: torch.Tensor, fallback_idx_1d: torch.Tensor):


    w = weights_2d.clone()
    row_sum = w.sum(dim=-1)  # (N,)
    bad = row_sum <= 0
    if bad.any():
        w[bad] = 0.0
        w[bad].scatter_(dim=1, index=fallback_idx_1d[bad].unsqueeze(1), value=1.0)
    return w.multinomial(1).squeeze(1)

def sample_pomo_every4th_topk_else_random(
    probs: torch.Tensor,   
    k: int = 4,
    eps: float = 1e-12,
):

    assert probs.dim() == 3, f"Expected (B,P,A), got {probs.shape}"
    B, P, A = probs.shape
    device = probs.device
    k_eff = min(int(k), A)

    q = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    probs_for_argmax = torch.nan_to_num(probs, nan=-1e9, posinf=-1e9, neginf=-1e9)
    fallback = probs_for_argmax.argmax(dim=-1)  

    selected = torch.empty((B, P), dtype=torch.long, device=device)

    p_idx = torch.arange(P, device=device)
    mask_top = (p_idx % 8 == 0)   
    mask_rnd = ~mask_top          

    # ---- top-k pomo branches ----
    if mask_top.any():
        q_top = q[:, mask_top, :]              # (B, Pt, A)
        fb_top = fallback[:, mask_top]         # (B, Pt)
        Pt = q_top.size(1)

        top_idx = torch.topk(q_top, k=k_eff, dim=-1, largest=True).indices  # (B,Pt,k)
        w = torch.zeros_like(q_top)
        w.scatter_(dim=-1, index=top_idx, value=1.0)
        w = w * q_top  # only keep top-k weights (not renormed)

        sel = _safe_multinomial(
            w.reshape(B * Pt, A),
            fb_top.reshape(B * Pt)
        ).reshape(B, Pt)
        selected[:, mask_top] = sel

    # ---- random pomo branches (full) ----
    if mask_rnd.any():
        q_rnd = q[:, mask_rnd, :]              # (B, Pr, A)
        fb_rnd = fallback[:, mask_rnd]         # (B, Pr)
        Pr = q_rnd.size(1)

        sel = _safe_multinomial(
            q_rnd.reshape(B * Pr, A),
            fb_rnd.reshape(B * Pr)
        ).reshape(B, Pr)
        selected[:, mask_rnd] = sel

    prob = probs.gather(dim=-1, index=selected.unsqueeze(-1)).squeeze(-1)
    prob = torch.nan_to_num(prob, nan=0.0).clamp_min(eps)  # for log safety only

    return selected, prob
class KPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = KP_Encoder(**model_params)
        self.decoder = KP_Decoder(**model_params)
        self.encoded_nodes_and_dummy = None
        self.encoded_nodes = None
        self.encoded_graph = None
        # shape: (batch, problem, EMBEDDING_DIM)

    def pre_forward(self, reset_state, pref):
        self.encoded_nodes = self.encoder(reset_state.problems, pref)
        
        self.encoded_graph = self.encoded_nodes.mean(dim=1, keepdim=True)
        
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
        else:
            # shape: (batch, pomo, embedding)
            probs = self.decoder(self.encoded_graph, capacity = state.capacity, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if self.training or self.model_params['eval_type'] == 'softmax':
                #selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

                #prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                # shape: (batch, pomo)
                selected, prob = sample_pomo_every4th_topk_else_random(probs, k=5)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None


        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class KP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_pref = nn.Linear(2, embedding_dim)
        self.embedding = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, pref):
        # data.shape: (batch, problem, 2)

        embedded_pref = self.embedding_pref(pref)[None, None, :].repeat(data.shape[0], 1, 1)
        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_input, embedded_pref), dim=1)
        for layer in self.layers:
            out = layer(out)

        return out
class RF(nn.Module):
 
    def __init__(self, emb_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = emb_dim * 2  #

        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

        

    def forward(self, attn_out, pref):
        B, N, E = attn_out.size()
        # pref: (B,1,E) -> (B,N,E)
        pref_expand = pref.expand(B, N, E)

        x = torch.cat([attn_out, pref_expand], dim=-1)  # (B,N,2E)
        delta = self.fc2(F.relu(self.fc1(x)))           # (B,N,E)
        return attn_out + delta
class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        D  = model_params['embedding_dim']
        H  = model_params['head_num']
        Dh = model_params['qkv_dim']

        # ===== (1) nodes self-attn =====
        self.Wq_n = nn.Linear(D, H * Dh, bias=False)
        self.Wk_n = nn.Linear(D, H * Dh, bias=False)
        self.Wv_n = nn.Linear(D, H * Dh, bias=False)
        self.combine1 = nn.Linear(H * Dh, D)
        # ===== (2) pref reads nodes (pref as Q, nodes as KV) =====
        self.Wq_p = nn.Linear(D, H * Dh, bias=False)
        self.Wk_p = nn.Linear(D, H * Dh, bias=False)
        self.Wv_p = nn.Linear(D, H * Dh, bias=False)
        self.combine2 = nn.Linear(H * Dh, D)

        # ===== (2b) nodes read pref (nodes as Q, pref as KV) =====
        self.Wq_np = nn.Linear(D, H * Dh, bias=False)
        self.Wk_np = nn.Linear(D, H * Dh, bias=False)
        self.Wv_np = nn.Linear(D, H * Dh, bias=False)
        self.combine3 = nn.Linear(H * Dh, D)

        # ===== grf=====
        self.gate = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.Sigmoid()
        )
        self.write = nn.Linear(D, D, bias=False)
       

        # ===== Add&Norm =====
        self.addnorm_n1 = Add_And_Normalization_Module(**model_params)
        self.addnorm_n2 = Add_And_Normalization_Module(**model_params)
        self.addnorm_p1 = Add_And_Normalization_Module(**model_params)
        self.addnorm_p2 = Add_And_Normalization_Module(**model_params)
        self.addnorm_n3 = Add_And_Normalization_Module(**model_params)
        self.addnorm_n4 = Add_And_Normalization_Module(**model_params)

        # ===== FFN =====
        if model_params['ffd'] == 'ffd':
            self.ffn_nodes  = Feed_Forward_Module(**model_params)
            self.ffn_pref   = Feed_Forward_Module(**model_params)
            self.ffn_nodes2 = Feed_Forward_Module(**model_params)  #
        elif model_params['ffd'] == 'siglu':
            assert D == 128
            self.ffn_nodes  = ParallelGatedMLP()
            self.ffn_pref   = ParallelGatedMLP()
            self.ffn_nodes2 = ParallelGatedMLP()
        else:
            raise NotImplementedError

    def forward(self, input1):
        H = self.model_params['head_num']
        nodes = input1[:, :-1, :]   # (B, N, D)
        pref  = input1[:, -1:, :]   # (B, 1, D)

        # ===== (1) nodes self-attn =====
        qn = reshape_by_heads(self.Wq_n(nodes), head_num=H)  # (B,H,N,Dh)
        kn = reshape_by_heads(self.Wk_n(nodes), head_num=H)
        vn = reshape_by_heads(self.Wv_n(nodes), head_num=H)
        nodes_attn = multi_head_attention(qn, kn, vn)        # (B,N,H*Dh)
        nodes_attn = self.combine1(nodes_attn)               # (B,N,D)
        nodes1 = self.addnorm_n1(nodes, nodes_attn)
        nodes_ff = self.ffn_nodes(nodes1)
        nodes2 = self.addnorm_n2(nodes1, nodes_ff)           # (B,N,D)

        # ===== (2) pref reads nodes =====
        qp = reshape_by_heads(self.Wq_p(pref),  head_num=H)  # (B,H,1,Dh)
        kp = reshape_by_heads(self.Wk_p(nodes2), head_num=H) # (B,H,N,Dh)
        vp = reshape_by_heads(self.Wv_p(nodes2), head_num=H)
        pref_ctx = multi_head_attention(qp, kp, vp)          # (B,1,H*Dh)
        pref_ctx = self.combine2(pref_ctx)                   # (B,1,D)
        pref1 = self.addnorm_p1(pref, pref_ctx)
        pref_ff = self.ffn_pref(pref1)
        pref2 = self.addnorm_p2(pref1, pref_ff)              # (B,1,D)

        
        q_np = reshape_by_heads(self.Wq_np(nodes2),   head_num=H)  # (B,H,N,Dh)
        k_np = reshape_by_heads(self.Wk_np(pref2),  head_num=H)  # (B,H,T,Dh)
        v_np = reshape_by_heads(self.Wv_np(pref2),  head_num=H)
        nodes_pref = multi_head_attention(q_np, k_np, v_np)        # (B,N,H*Dh)
        nodes_pref = self.combine3(nodes_pref)                     # (B,N,D)
        nodes2b = self.addnorm_n3(nodes2, nodes_pref)
        nodes2b_ff = self.ffn_nodes2(nodes2b)
        nodes2c = self.addnorm_n4(nodes2b, nodes2b_ff)             # (B,N,D)
        
        # ===== (3) grf =====
        pref_exp = pref2.expand(-1, nodes2c.size(1), -1)           
        
        g = self.gate(torch.cat([nodes2c, pref_exp], dim=-1))      # (B,N,D)
        nodes3 = nodes2c + g * self.write(pref_exp)                # (B,N,D)

        out = torch.cat([nodes3, pref2], dim=1)                    # (B,N+1,D)
        return out
 




########################################
# DECODER
########################################

class KP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(1 + embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.pref_emb = None
        self.rf = RF(emb_dim=embedding_dim, hidden_dim=self.model_params['ff_hidden_dim'])
        
        
    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        
        self.single_head_key = encoded_nodes[:, :-1].transpose(1, 2)
        self.pref_emb = encoded_nodes[:, -1:,:]
     
    def forward(self, graph, capacity, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        batch_size = capacity.size(0)
        group_size = capacity.size(1)

        #  Multi-Head Attention
        #######################################################
        input1 = graph.expand(batch_size, group_size, embedding_dim)
        input2 = capacity[:, :, None]
        input_cat = torch.cat((input1, input2), dim=2)
        
        #  Multi-Head Attention
        #######################################################
        q = reshape_by_heads(self.Wq(input_cat), head_num=head_num)

        out_concat = multi_head_attention(q, self.k, self.v, ninf_mask=torch.cat(
            (ninf_mask, torch.zeros(ninf_mask.shape[0], ninf_mask.shape[1], 1)), dim=-1))
       
        mh_atten_out = self.multi_head_combine(out_concat)
        mh_atten_out = self.rf(mh_atten_out, self.pref_emb)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        #score_masked = score_clipped + ninf_mask
        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, ninf_mask=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be either 1 or group
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch, head_num, n, TSP_SIZE)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, problem_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape = (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        
        embedding_dim = model_params['embedding_dim']
        self.norm_type = model_params['norm_type'] # instance or layer
        if self.norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)
        elif self.norm_type == 'rms': # layer
            self.norm = RMSNorm(embedding_dim)
        elif self.norm_type == 'scale':
            self.norm = ScaleNorm(embedding_dim)
        else:
            raise NotImplementedError

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)
        added = input1 + input2
        # shape: (batch, problem, embedding)
        if self.norm_type == 'instance':
            out = self.norm(added.transpose(1, 2)).transpose(1, 2) # (batch, problem, embedding)
        else:  # layer rms
            out = self.norm(added) # (batch, problem, embedding)
        return out



class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""
    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()
        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError
        self.multiple_of = multiple_of * model_parallel_size
        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        ) # 512

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)       



class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))