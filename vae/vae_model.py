import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_size, Seq_len, Dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (B, S, D) -> (B, S, D * 3) -> 3 * (B, S, D) NOTE: nn.Linear multiplys last dimension of any given vector
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (B, S, D) -> (B, S, H, D/H) -> (B, H, S, D/H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (B, H, S, S)
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (B, H, S, S) @ (B, H, S, D/H) -> (B, H, S, D/H)
        output = weight @ v

        # (B, H, S, D/H) -> (B, S, H, D/H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (B, S, D)
        return output

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Features, L)
        residue = x

        # x: (B, Features, L) -> x: (B, L, Features)
        x = x.transpose(-1, -2)

        # x: (B, L, Features) -> x: (B, Features, L) 
        x = self.attention(x)
        x = x.transpose(-1, -2)
        
        x += residue
        return x

class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, In_channels, L)
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        
        return x + self.residual_layer(residue)

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
                # (Batch_Size, Channel, L) -> (B, 128, L)
                nn.Conv1d(12, 128, kernel_size=3, padding=1),
                # (B, 128, L) -> (B, 128, L)
                VAE_ResidualBlock(128, 128),
                # (B, 128, L) -> (B, 128, L)
                VAE_ResidualBlock(128, 128),

                # (B, 128, L) -> (B, 128, L/2)
                nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=0),
                # (B, 128, L/2) -> (B, 256, L/2)
                VAE_ResidualBlock(128, 256),
                VAE_ResidualBlock(256, 256),
                
                nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(256, 512),
                VAE_ResidualBlock(512, 512), 

                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(512, 512), 
                VAE_ResidualBlock(512, 512), 

                # (B, 512, L/8) -> (B, 512, L/8)
                VAE_ResidualBlock(512, 512), 
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                
                # (B, 512, L/8) -> (B, 512, L/8)
                nn.GroupNorm(32, 512),
                nn.SiLU(),

                #NOTE: BottleNeck
                # (B, 512, L/8) -> (B, 8, L/8)
                nn.Conv1d(512, 8, kernel_size=3, padding=1),
                nn.Conv1d(8, 8, kernel_size=1, padding=0),
                )

    def forward(self, x: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
        # x: (B, L, C)
        # noise: (B, C_Out, L/8)
        # output: (B, C_Out, L/8)
        # NOTE: apply noise after encoding

        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        # x: (B, C, L) -> (B, 8, L/8)
        for module in self:
            if getattr(module, 'stride', None) == (2,):
                # Padding(left, right)
                x = F.pad(x, (0, 1))
            x = module(x)

        # (B, 8, L/8) -> 2 x (B, 4, L/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # z ~ N(0, 1) -> x ~ N(mean, variance)
        # x = mean + stdev * z
        # (B, 4, L/8) -> (B, 4, L/8)
        if noise is None:
            noise = torch.randn(stdev.shape, device=stdev.device)
        x = mean + stdev * noise

        # Scale the output by a constant (magic number)
        x *= 0.18215

        return x, mean, log_variance


class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(
                nn.Conv1d(4, 4, kernel_size=1, padding=0),
                nn.Conv1d(4, 512, kernel_size=3, padding=1),

                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),

                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),

                # (B, 512, L/8) -> (B, 512, L/4)
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),

                # (B, 512, L/4) -> (B, 512, L/2)
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),

                VAE_ResidualBlock(512, 256),
                VAE_ResidualBlock(256, 256),
                VAE_ResidualBlock(256, 256),

                # (B, 256, L/2) -> (B, 256, L)
                nn.Upsample(scale_factor=2),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                
                VAE_ResidualBlock(256, 128),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),

                nn.GroupNorm(32, 128),
                nn.SiLU(),

                # (B, 128, L) -> (B, 12, L)
                nn.Conv1d(128, 12, kernel_size=3, padding=1)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L/8)

        x /= 0.18215

        for module in self:
            x = module(x)

        # (B, 12, L) -> (B, L, 12)
        x = x.transpose(1, 2)
        return x

class VAE_Decoder_FirstCycle(nn.Module):
    """
    Smaller decoder to produce only the first-cycle ECG segment.
    Input:  z (B, 4, L/8)
    Output: (B, TARGET_LEN, 12)
    """
    def __init__(self, target_len: int, base_channels: int = 128):
        super().__init__()
        self.target_len = target_len

        self.net = nn.Sequential(
            nn.Conv1d(4, base_channels, kernel_size=1, padding=0),
            VAE_ResidualBlock(base_channels, base_channels),
            VAE_ResidualBlock(base_channels, base_channels),

            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(target_len),

            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, 12, kernel_size=3, padding=1),  # -> (B, 12, TARGET_LEN)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z / 0.18215
        x = self.net(z)
        x = x.transpose(1, 2)
        return x

def loss_function(recons, x, mu, log_var, kld_weight=1) -> dict:
    r"""
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :para recons: reconstruction vector
    :para x: original vector
    :para mu: mean of latent gaussian distribution
    :log_var: log of latent gaussian distribution variance
    :kld_weight: weight of kl-divergence term
    """
    recons_loss = F.mse_loss(recons, x, reduction='sum').div(x.size(0))
    q_z_x = Normal(mu, log_var.mul(.5).exp())
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    kld_loss = kl_divergence(q_z_x, p_z).sum(1).mean()

    loss = recons_loss + kld_weight * kld_loss

    return {'loss': loss, 'mse':recons_loss.detach(), 'KLD':kld_loss.detach()}

class MiniDecoderMSE(nn.Module):
    def __init__(self, target_len: int = 61, z_channels: int = 4, base: int = 128):
        super().__init__()
        self.target_len = target_len
        self.z_channels = z_channels
        self.base = base

        self.in_proj = nn.Conv1d(z_channels, base, kernel_size=3, padding=1)
        self.block1 = VAE_ResidualBlock(base, base)
        self.block2 = VAE_ResidualBlock(base, base)
        self.block3 = VAE_ResidualBlock(base, base)

        self.time_fc = nn.Linear(128, target_len)

        self.refine = nn.Sequential(
            nn.GroupNorm(32, base),
            nn.SiLU(),
            nn.Conv1d(base, base, kernel_size=3, padding=1),
            VAE_ResidualBlock(base, base),
        )
        self.out = nn.Conv1d(base, 12, kernel_size=1)

        self.amp_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(z_channels, 32),
            nn.SiLU(),
            nn.Linear(32, 24)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(max(1, fan_in))
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(z)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.time_fc(x)                 # (B, base, target_len)
        x = self.refine(x)
        x = self.out(x)                     # (B, 12, target_len)

        s_b = self.amp_head(z)              # (B, 24)
        s, b = s_b[:, :12], s_b[:, 12:]
        s = F.softplus(s) + 1e-3
        x = x * s.unsqueeze(-1) + b.unsqueeze(-1)

        return x.transpose(1, 2)            # (B, target_len, 12)

