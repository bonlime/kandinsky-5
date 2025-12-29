import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from .utils import get_freqs, nablaT_v2
from .attention import SelfAttentionEngine

@torch.compile()
@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_scale_shift_norm(norm, x, scale, shift):
    return (norm(x) * (scale + 1.0) + shift).to(torch.bfloat16)

@torch.compile()
@torch.autocast(device_type="cuda", dtype=torch.float32)
def apply_gate_sum(x, out, gate):
    return (x + gate * out).to(torch.bfloat16)

@torch.compile()
@torch.autocast(device_type="cuda", enabled=False)
def apply_rotary(x, rope):
    x_ = x.reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(torch.bfloat16)


class TimeEmbeddings(nn.Module):
    def __init__(self, model_dim, time_dim, max_period=10000.0):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        self.register_buffer(
            "freqs", get_freqs(model_dim // 2, max_period), persistent=False
        )
        self.in_layer = nn.Linear(model_dim, time_dim, bias=True)
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, time_dim, bias=True)

    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, time):
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = self.out_layer(self.activation(self.in_layer(time_embed)))
        return time_embed


class TextEmbeddings(nn.Module):
    def __init__(self, text_dim, model_dim):
        super().__init__()
        self.in_layer = nn.Linear(text_dim, model_dim, bias=True)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, text_embed):
        text_embed = self.in_layer(text_embed)
        return self.norm(text_embed).type_as(text_embed)


class VisualEmbeddings(nn.Module):
    def __init__(self, visual_dim, model_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = nn.Linear(math.prod(patch_size) * visual_dim, model_dim)

    def forward(self, x):
        duration, height, width, dim = x.shape
        x = (
            x.view(
                duration // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                dim,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .flatten(3, 6)
        )
        return self.in_layer(x)


class RoPE1D(nn.Module):
    def __init__(self, dim, max_pos=1024, max_period=10000.0):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        freq = get_freqs(dim // 2, max_period)
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer(f"args", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, pos):
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class RoPE3D(nn.Module):
    def __init__(self, axes_dims, max_pos=(128, 128, 128), max_period=10000.0):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(zip(axes_dims, max_pos)):
            freq = get_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, shape, pos, scale_factor=(1.0, 1.0, 1.0)):
        duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(duration, 1, 1, -1).repeat(1, height, width, 1),
                args_h.view(1, height, 1, -1).repeat(duration, 1, width, 1),
                args_w.view(1, 1, width, -1).repeat(duration, height, 1, 1),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Modulation(nn.Module):
    def __init__(self, time_dim, model_dim, num_params):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = nn.Linear(time_dim, num_params * model_dim)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    @torch.compile()
    @torch.autocast(device_type="cuda", dtype=torch.float32)
    def forward(self, x):
        return self.out_layer(self.activation(x))


class MultiheadSelfAttentionEnc(nn.Module):
    def __init__(self, num_channels, head_dim, attention_engine="auto", text_token_padding=False):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        self.num_chunks = 2

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)
        if text_token_padding:
            self.attn_engine = SelfAttentionEngine("sdpa")
        else:
            self.attn_engine = SelfAttentionEngine(attention_engine)

    @torch.compile()
    def compute_qk(self, x, rope, proj_fn, norm_fn, shape):
        result = proj_fn(x).view(*shape, self.num_heads, -1)
        return apply_rotary(norm_fn(result.float()).type_as(result), rope)
    
    @torch.compile()
    def scaled_dot_product_attention(self, query, key, value, attention_mask=None):
        args = {"q": query, "k": key, "v": value}
        if attention_mask is not None:
            args["attn_mask"] = attention_mask
        out = self.attn_engine.get_attention()(**args)[0].flatten(-2, -1)
        return out
    
    @torch.compile()
    def out_l(self, x):
        return self.out_layer(x)
    
    def _forward(self, x, rope, attention_mask):
        shape = x.shape[:-1]
        q = self.compute_qk(x, rope, self.to_query, self.query_norm, shape)
        k = self.compute_qk(x, rope, self.to_key, self.key_norm, shape)
        v = self.to_value(x).view(*shape, self.num_heads, -1)
        out = self.scaled_dot_product_attention(q, k, v, attention_mask)
        return self.out_l(out)
    
    def _forward_chunked(self, x, rope, attention_mask):
        def process_chunks(proj_fn, norm_fn):
            _, L, _ = x.shape
            chunk_size = (L + self.num_chunks - 1) // self.num_chunks
            chunks = []
            print(f'MultiheadSelfAttentionEnc: L: {L}, chunk_size: {chunk_size}')
            for i in range(0, L, chunk_size):
                end_idx = min(i + chunk_size, L)
                x_chunk = x[:, i:end_idx]
                rope_chunk = rope[:, i:end_idx]
                chunks.append(self.compute_qk(
                    x_chunk, rope_chunk, proj_fn, norm_fn, x_chunk.shape[:-1]))
            return torch.cat(chunks, dim=1)

        q = process_chunks(self.to_query, self.query_norm)
        k = process_chunks(self.to_key, self.key_norm)
        v = self.to_value(x).view(*x.shape[:-1], self.num_heads, -1)
        out = self.scaled_dot_product_attention(q, k, v, attention_mask)
        return self.out_l(out)
    
    def forward(self, x, rope, attention_mask=None):
        if x.shape[1] > 8192:
            return self._forward_chunked(x, rope, attention_mask)
        else:
            return self._forward(x, rope, attention_mask)
        #return self._forward(x, rope, attention_mask)


class MultiheadSelfAttentionDec(nn.Module):
    def __init__(self, num_channels, head_dim, attention_engine="auto"):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        self.num_chunks = 2

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

        self.attn_engine = SelfAttentionEngine(attention_engine)

    @torch.compile()
    def compute_qk(self, x, rope, proj_fn, norm_fn, shape):
        result = proj_fn(x).view(*shape, self.num_heads, -1)
        return apply_rotary(norm_fn(result.float()).type_as(result), rope)

    @torch.compile()
    def attention(self, query, key, value):
        out = self.attn_engine.get_attention()(
            q=query,
            k=key,
            v=value)[0].flatten(-2, -1)
        return out

    @torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
    def nabla(self, query, key, value, sparse_params=None):
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        block_mask = nablaT_v2(
            query,
            key,
            sparse_params["sta_mask"],
            thr=sparse_params["P"],
        )
        out = (
            flex_attention(
                query,
                key,
                value,
                block_mask=block_mask
            )
            .transpose(1, 2)
            .contiguous()
        )
        out = out[0].flatten(-2, -1)
        return out

    @torch.compile()
    def out_l(self, x):
        return self.out_layer(x)
    
    def _forward(self, x, rope, sparse_params):
        shape = x.shape[:-1]
        q = self.compute_qk(x, rope, self.to_query, self.query_norm, shape)
        k = self.compute_qk(x, rope, self.to_key, self.key_norm, shape)
        v = self.to_value(x).view(*shape, self.num_heads, -1)

        if sparse_params is not None:
            out = self.nabla(q, k, v, sparse_params=sparse_params)
        else:
            out = self.attention(q, k, v)
        return self.out_l(out)

    def _forward_chunked(self, x, rope, sparse_params):
        def process_chunks(proj_fn, norm_fn):
            _, L, _ = x.shape
            chunk_size = (L + self.num_chunks - 1) // self.num_chunks
            chunks = []
            print(f'MultiheadSelfAttentionDec: L: {L}, chunk_size: {chunk_size}')
            for i in range(0, L, chunk_size):
                end_idx = min(i + chunk_size, L)
                x_chunk = x[:, i:end_idx]
                rope_chunk = rope[i:end_idx]
                chunks.append(self.compute_qk(
                    x_chunk, rope_chunk, proj_fn, norm_fn, x_chunk.shape[:-1]))
            return torch.cat(chunks, dim=1)

        q = process_chunks(self.to_query, self.query_norm)
        k = process_chunks(self.to_key, self.key_norm)
        v = self.to_value(x).view(*x.shape[:-1], self.num_heads, -1)

        if sparse_params is not None:
            out = self.nabla(q, k, v, sparse_params=sparse_params)
        else:
            out = self.attention(q, k, v)
        return self.out_l(out)
    
    def forward(self, x, rope, sparse_params=None):
        if x.shape[1] > 8192:
            return self._forward_chunked(x, rope, sparse_params)
        else:
            return self._forward(x, rope, sparse_params)
        #return self._forward(x, rope, sparse_params)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, num_channels, head_dim,  attention_engine="auto", text_token_padding=False):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

        if text_token_padding:
            self.attn_engine = SelfAttentionEngine("sdpa")
        else:
            self.attn_engine = SelfAttentionEngine(attention_engine)

    @torch.compile()
    def get_qkv(self, x, cond):
        query = self.to_query(x)
        key = self.to_key(cond)
        value = self.to_value(cond)

        shape, cond_shape = query.shape[:-1], key.shape[:-1]
        query = query.reshape(*shape, self.num_heads, -1)
        key = key.reshape(*cond_shape, self.num_heads, -1)
        value = value.reshape(*cond_shape, self.num_heads, -1)

        return query, key, value

    @torch.compile()
    def norm_qk(self, q, k):
        q = self.query_norm(q.float()).type_as(q)
        k = self.key_norm(k.float()).type_as(k)
        return q, k

    @torch.compile()
    def attention(self, query, key, value, attention_mask=None):
        args = {"q": query, "k": key, "v": value}
        if attention_mask is not None:
            args["attn_mask"] = attention_mask
        out = self.attn_engine.get_attention()(**args)[0].flatten(-2, -1)
        return out

    @torch.compile()
    def out_l(self, x):
        return self.out_layer(x)

    def forward(self, x, cond, attention_mask=None):
        query, key, value = self.get_qkv(x, cond)
        query, key = self.norm_qk(query, key)

        out = self.attention(query, key, value, attention_mask)
        out = self.out_l(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)
        self.num_chunks = 4

    @torch.compile()
    def _forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))

    @torch.compile()
    def _forward_chunked(self, x):
        B, L, _ = x.shape
        chunk_size = (L + self.num_chunks - 1) // self.num_chunks
        output = torch.empty(B, L, self.out_layer.out_features, dtype=x.dtype, device=x.device)
        print(f'FeedForward: L: {L}, chunk_size: {chunk_size}')
        for i in range(0, L, chunk_size):
            end_idx = min(i + chunk_size, L)
            def compute_chunk(x_chunk):
                activated = self.activation(self.in_layer(x_chunk))
                return self.out_layer(activated)
            output[:, i:end_idx] = compute_chunk(x[:, i:end_idx])

            output[:, i:end_idx] = self._forward(x[:, i:end_idx])
        return output

    def forward(self, x):
        if x.shape[1] > 8192:
            return self._forward_chunked(x)
        else:
            return self._forward(x)
        #return self._forward(x)


class OutLayer(nn.Module):
    def __init__(self, model_dim, time_dim, visual_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.out_layer = nn.Linear(
            model_dim, math.prod(patch_size) * visual_dim, bias=True
        )

    def forward(self, visual_embed, text_embed, time_embed):
        shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
        visual_embed = apply_scale_shift_norm(
            self.norm,
            visual_embed,
            scale[:, None, None],
            shift[:, None, None],
        ).type_as(visual_embed)
        x = self.out_layer(visual_embed)

        duration, height, width, _ = x.shape
        x = (
            x.view(
                duration,
                height,
                width,
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 4, 1, 5, 2, 6, 3)
            .flatten(0, 1)
            .flatten(1, 2)
            .flatten(2, 3)
        )
        return x
