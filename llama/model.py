# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
)

from export_libtorch import Container


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_seq_len: int = 128


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(1)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.weight_q = nn.Parameter(torch.Tensor(args.dim, args.dim))
        self.weight_k = nn.Parameter(torch.Tensor(args.dim, args.dim))
        self.weight_v = nn.Parameter(torch.Tensor(args.dim, args.dim))
        self.weight_o = nn.Parameter(torch.Tensor(args.dim, args.dim))

        temp = self.weight_q.view(
            self.n_local_heads, self.head_dim, self.n_local_heads, self.head_dim
        ).transpose(1, 2)
        self.wq = [
            [temp[i, j, :] for j in range(self.n_local_heads)]
            for i in range(self.n_local_heads)
        ]

        temp = self.weight_k.view(
            self.n_local_heads, self.head_dim, self.n_local_heads, self.head_dim
        ).transpose(1, 2)
        self.wk = [
            [temp[i, j, :] for j in range(self.n_local_heads)]
            for i in range(self.n_local_heads)
        ]

        temp = self.weight_v.view(
            self.n_local_heads, self.head_dim, self.n_local_heads, self.head_dim
        ).transpose(1, 2)
        self.wv = [
            [temp[i, j, :] for j in range(self.n_local_heads)]
            for i in range(self.n_local_heads)
        ]

        temp = self.weight_o.view(
            self.n_local_heads, self.head_dim, self.n_local_heads, self.head_dim
        ).transpose(1, 2)
        self.wo = [
            [temp[i, j, :] for j in range(self.n_local_heads)]
            for i in range(self.n_local_heads)
        ]

        self.cache_k = [
            torch.zeros(
                (
                    args.max_seq_len,
                    self.head_dim,
                )
            )
            for _ in range(self.n_local_heads)
        ]
        self.cache_v = [
            torch.zeros(
                (
                    args.max_seq_len,
                    self.head_dim,
                )
            )
            for _ in range(self.n_local_heads)
        ]

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        seqlen, _ = x.shape

        temp = x.view(seqlen, self.n_local_heads, self.head_dim).transpose(0, 1)
        x = [temp[i, :, :] for i in range(self.n_local_heads)]
        output = torch.zeros((self.n_local_heads, seqlen, self.head_dim))

        for i in range(self.n_local_heads):
            xq, xk, xv = (torch.zeros((seqlen, self.head_dim)) for _ in range(3))
            for j in range(self.n_local_heads):
                xq += torch.mm(x[j], self.wq[j][i])
                xk += torch.mm(x[j], self.wk[j][i])
                xv += torch.mm(x[j], self.wv[j][i])
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            self.cache_k[i][start_pos : start_pos + seqlen] = xk
            self.cache_v[i][start_pos : start_pos + seqlen] = xv
            key = self.cache_k[i][: start_pos + seqlen]
            value = self.cache_v[i][: start_pos + seqlen]
            score = torch.mm(xq, key.transpose(0, 1)) / math.sqrt(self.head_dim)
            if seqlen > 1:
                score = score + mask
            score = F.softmax(score.float(), dim=-1).type_as(xq)
            qkv = torch.mm(score, value)
            for j in range(self.n_local_heads):
                output[j] += torch.mm(qkv, self.wo[i][j])

        return output.transpose(0, 1).contiguous().view(seqlen, -1)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.weight_1 = nn.Parameter(torch.Tensor(dim, hidden_dim))
        self.weight_2 = nn.Parameter(torch.Tensor(hidden_dim, dim))
        self.weight_3 = nn.Parameter(torch.Tensor(dim, hidden_dim))

        self.sub_matrix_dim = 128

        self.n_submatrix = dim // self.sub_matrix_dim
        self.n_hidden_submatrix = hidden_dim // self.sub_matrix_dim

        temp = self.weight_1.view(
            self.n_submatrix,
            self.sub_matrix_dim,
            self.n_hidden_submatrix,
            self.sub_matrix_dim,
        ).transpose(1, 2)
        self.w1 = [
            [temp[i, j, :] for j in range(self.n_hidden_submatrix)]
            for i in range(self.n_submatrix)
        ]

        temp = self.weight_2.view(
            self.n_hidden_submatrix,
            self.sub_matrix_dim,
            self.n_submatrix,
            self.sub_matrix_dim,
        ).transpose(1, 2)
        self.w2 = [
            [temp[i, j, :] for j in range(self.n_submatrix)]
            for i in range(self.n_hidden_submatrix)
        ]

        temp = self.weight_3.view(
            self.n_submatrix,
            self.sub_matrix_dim,
            self.n_hidden_submatrix,
            self.sub_matrix_dim,
        ).transpose(1, 2)
        self.w3 = [
            [temp[i, j, :] for j in range(self.n_hidden_submatrix)]
            for i in range(self.n_submatrix)
        ]

    def forward(self, x):
        seqlen, _ = x.shape

        temp = x.view(seqlen, self.n_submatrix, self.sub_matrix_dim).transpose(0, 1)
        x = [temp[i, :, :] for i in range(self.n_submatrix)]

        output = torch.zeros((self.n_submatrix, seqlen, self.sub_matrix_dim))

        for i in range(self.n_hidden_submatrix):
            gate, up = (torch.zeros((seqlen, self.sub_matrix_dim)) for _ in range(2))
            for j in range(self.n_submatrix):
                gate += torch.mm(x[j], self.w1[j][i])
                up += torch.mm(x[j], self.w3[j][i])

            temp = F.silu(gate) * up

            for j in range(self.n_submatrix):
                output[j] += torch.mm(temp, self.w2[i][j])

        return output.transpose(0, 1).contiguous().view(seqlen, -1)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.n_heads = params.n_heads
        self.head_dim = params.dim // params.n_heads

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.weight_output = nn.Parameter(torch.Tensor(params.dim, params.vocab_size))

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads,
            self.params.max_seq_len * 2,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        (seqlen,) = tokens.shape
        h = self.tok_embeddings(tokens)
        # encrypt
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        input = torch.tensor(h)
        in_out = {"input": input.to("cpu")}
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            output = torch.tensor(h)
            in_out["output"] = output.to("cpu")
            container = torch.jit.script(Container(in_out))
            container.save("transformer_in_out.pt")
            break
        h = self.norm(h)

        return torch.mm(h, self.weight_output).float()
