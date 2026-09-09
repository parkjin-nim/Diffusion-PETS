import math
import torch
import torch.nn.functional as F

from torch import nn


def exists(x):
    """
    Check if the input is not None.

    Args:
        x: The input to check.

    Returns:
        bool: True if the input is not None, False otherwise.
    """
    return x is not None

def default(val, d):
    """
    Return the value if it exists, otherwise return the default value.

    Args:
        val: The value to check.
        d: The default value or a callable that returns the default value.

    Returns:
        The value if it exists, otherwise the default value.
    """
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    """
    Return the input tensor unchanged.

    Args:
        t: The input tensor.
        *args: Additional arguments (unused).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        The input tensor unchanged.
    """
    return t

def extract(a, t, x_shape):
    """
    Extracts values from tensor `a` at indices specified by tensor `t` and reshapes the result.
    Args:
        a (torch.Tensor): The input tensor from which values are extracted.
        t (torch.Tensor): The tensor containing indices to extract from `a`.
        x_shape (tuple): The shape of the tensor `x` which determines the final shape of the output.
    Returns:
        torch.Tensor: A tensor containing the extracted values, reshaped to match the shape of `x` except for the first dimension.
    """

    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cond_fn(x, t, classifier=None, y=None, classifier_scale=1.):
    """
    Compute the gradient of the classifier's log probabilities with respect to the input.

    Args:
        classifier (nn.Module): The classifier model used to compute logits.
        x (torch.Tensor): The input tensor for which gradients are computed.
        t (torch.Tensor): The time step tensor.
        y (torch.Tensor, optional): The target labels tensor. Must not be None.
        classifier_scale (float, optional): Scaling factor for the gradients. Default is 1.

    Returns:
        torch.Tensor: The gradient of the selected log probabilities with respect to the input tensor, scaled by classifier_scale.
    """
    assert y is not None
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

# normalization functions

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding module.

    This module generates sinusoidal positional embeddings for input tensors.
    The embeddings are computed using sine and cosine functions with different frequencies.

    Attributes:
        dim (int): The dimension of the positional embeddings.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# learnable positional embeds

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding module.

    This module generates learnable positional embeddings for input tensors.
    The embeddings are learned during training and can adapt to the specific task.

    Attributes:
        d_model (int): The dimension of the positional embeddings.
        dropout (float): The dropout rate applied to the embeddings.
        max_len (int): The maximum length of the input sequences.
    """
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
    

class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)
    

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x
