import math
from typing import Optional, Union, Tuple, List

import torch
from torch import nn


class Swish(nn.Module):
    """Swish actiavation function: x * sigmoid(x)"""

    def forward(self, x):
        return x * torch.sigmoid(x)
    


class TimeEmbedding(nn.Module):

    def __init__(self, n_channels: int):
        """Embeddings for $t$"""

        super().__init__()

        self.n_channels = n_channels  # number of dimensions in the embedding
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)


    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        # PE^{(1)}_{t,i} = sin( \frac{t}{10000^{\frac{i}{d - 1}}} ) 
        # PE^{(2)}_{t,i} = cos( \frac{t}{10000^{\frac{i}{d - 1}}} )
        # where $d$ is `half_dim`

        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)  # shape (C/8, )
        emb = t[:, None] * emb[None, :]  # t * 10000^{- \frac{i}{d - 1}}; shape (B, C/8)
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)  # shape (B, C/4)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb
    


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """A residual block has two convolution layers with group normalization."""

        super().__init__()

        # First layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Second layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Project shortcut connection if necessary
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x.shape = [batch_size, in_channels, height, width]
        t.shape = [batch_size, time_channels]
        """

        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)
    


class AttentionBlock(nn.Module):

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None):
        """
        Attention block similar to [transformer multi-head attention](../../transformers/mha.html)
        n_channels: number of channels in the input
        n_heads: number of heads in multi-head attention
        d_k: number of dimensions in each head
        """

        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels

        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k


    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        x.shape = [batch_size, in_channels, height, width]
        t.shape = [batch_size, time_channels]
        """

        # t is not used (but kept in the arguments to match with `ResidualBlock`)
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape

        # Change x to shape [batch_size, seq, n_channels]
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to [batch_size, seq, n_heads, 3 * d_k]
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape [batch_size, seq, n_heads, d_k]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Calculate scaled dot-product: \frac{Q K^\top}{\sqrt{d_k}}
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension: softmax( \frac{Q K^\top}{\sqrt{d_k}} )
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        # Reshape to [batch_size, seq, n_heads * d_k]
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to [batch_size, seq, n_channels]
        res = self.output(res)

        # Add skip connection
        res += x
        # Change to shape [batch_size, in_channels, height, width]
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res
    

class ForwardBlock(nn.Module):
    """
    Down Block [res, attn]: used in the first half of U-Net at each resolution
    Up Block [res, attn]: used in the second half of U-Net at each resolution
    Middle Block [res, attn, res2]: used at the lowest resolution of the U-Net
    """

    def __init__(self, block_name: str, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()

        assert block_name in ['down', 'up', 'middle'], "incorrect block_name"

        if block_name == 'up':
            # since we concatenate the output of the same resolution from the first half of U-Net
            in_channels = in_channels + out_channels

        # Residual Block
        self.res = ResidualBlock(in_channels, out_channels, time_channels)

        # Attention Block
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

        # Second Residual Block for Middle Block
        if block_name == 'middle':
            self.res2 = ResidualBlock(out_channels, in_channels, time_channels)

        self.block_name = block_name

    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        if self.block_name == 'middle':
            x = self.res2(x, t)
        return x



class SampleBlock(nn.Module):
    """
    Upsample Block: scale up the feature map by 2 times
    Downsample Block: scale down the feature map by 1/2 times
    """

    def __init__(self, block_name, n_channels):
        super().__init__()

        assert block_name in ['up', 'down'], "incorrect block_name"

        stride = (2, 2)
        padding = (1, 1)

        if block_name == 'up':
            kernel_size = (4, 4)
            self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size, stride, padding)
        else:
            kernel_size = (3, 3)
            self.conv = nn.Conv2d(n_channels, n_channels, kernel_size, stride, padding)

        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # t is not used (but kept in the arguments to match with `ResidualBlock`)
        _ = t
        return self.conv(x)



class UNet(nn.Module):

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        image_channels: number of channels in the image. 3 for RGB.
        n_channels: number of channels in the initial feature map that we transform the image into
        ch_mults: list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        is_attn: list of booleans that indicate whether to use attention at each resolution
        n_blocks: number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        time_channels = n_channels * 4
        self.time_emb = TimeEmbedding(time_channels)

        # ========== First half of U-Net - decreasing resolution ==========
        downs = []
        # Number of channels
        out_channels = in_channels = n_channels

        for i, ch_mult in enumerate(ch_mults):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mult
            # Add `n_blocks`
            for _ in range(n_blocks):
                downs.append(ForwardBlock('down', in_channels, out_channels, time_channels, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < len(ch_mults) - 1:
                downs.append(SampleBlock('down', in_channels))

        # Combine the set of modules
        self.downs = nn.ModuleList(downs)
        # ==============================

        # ========== Middle Block ==========
        self.middle = ForwardBlock('middle', out_channels, out_channels, time_channels, has_attn=True)
        # ==============================

        # ========== Second half of U-Net - increasing resolution ==========
        ups = []
        # Number of channels
        in_channels = out_channels

        for i, ch_mult in reversed(list(enumerate(ch_mults))):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                ups.append(ForwardBlock('up', in_channels, out_channels, time_channels, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mult
            ups.append(ForwardBlock('up', in_channels, out_channels, time_channels, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                ups.append(SampleBlock('up', in_channels))
        
        # Combine the set of modules
        self.ups = nn.ModuleList(ups)
        # ==============================

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))


    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x.shape = [batch_size, in_channels, height, width]
        t.shape = [batch_size]
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.downs:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.ups:
            if isinstance(m, SampleBlock):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        # Final normalization and convolution
        x = self.final(self.act(self.norm(x)))
        return x