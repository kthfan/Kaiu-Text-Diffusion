
import functools
import torch
from torch import nn
from utils import timestep_embedding, zero_module, smart_group_norm, conv1x1, conv3x3



Pooling = functools.partial(nn.AvgPool2d, 2, 2)
Upsampling = functools.partial(nn.UpsamplingNearest2d, scale_factor=2)

class BasicResBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 emb_channels=None, 
                 is_up=False, 
                 is_down=False, 
                 dropout=0.,
                 norm_layer=smart_group_norm,
                 act_layer=nn.SiLU,
                 down_layer=Pooling,
                 up_layer=Upsampling):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.is_up = is_up
        self.is_down = is_down
        self.dropout = dropout
        
        self.shortcut_layers = nn.Sequential(
            up_layer() if self.is_up else down_layer() if self.is_down else nn.Identity(),
            conv1x1(in_channels, out_channels) if out_channels != in_channels else nn.Identity(),
        )
        
        self.in_residual_layers = nn.Sequential(
            norm_layer(in_channels),
            act_layer(),
            up_layer() if self.is_up else down_layer() if self.is_down else nn.Identity(),
            conv3x3(in_channels, out_channels),
        )
        
        self.out_residual_layers = nn.Sequential(
            norm_layer(out_channels),
            act_layer(),
            nn.Dropout2d(self.dropout) if self.dropout > 0 else nn.Identity(),
            zero_module(conv3x3(out_channels, out_channels)),
        )
        
        if self.emb_channels is not None:
            self.emb_layers = nn.Sequential(
                act_layer(),
                nn.Linear(emb_channels, out_channels),
            )
    
    def forward(self, x, t=None):
        x0 = self.shortcut_layers(x)
        x = self.in_residual_layers(x)
        if self.emb_channels is not None:
            x = x + self.emb_layers(t)[..., None, None]
        x = self.out_residual_layers(x)
        return x0 + x
    

class AttentionBlock(nn.Module):
    def __init__(self, 
                 channels, 
                 num_heads=8,
                 norm_layer=smart_group_norm):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = norm_layer(channels)
        self.qkv = conv1x1(channels, channels * 3)
        self.proj_out = zero_module(conv1x1(channels, channels))

    def forward(self, x, *args, **kwargs):
        qkv = self.qkv(self.norm(x))
        h = qkv_attention(qkv, self.num_heads)
        h = self.proj_out(h)
        return x + h

def qkv_attention(qkv, n_heads=8):
    b, ch3, h, w = qkv.shape
    ch = ch3 // 3
    scale = 1 / (ch // n_heads)**0.25
    qkv = qkv.view(b * n_heads, ch3 // n_heads, h * w)
    q, k, v = qkv.split(ch // n_heads, dim=1)
    m = torch.einsum('bcs, bct -> bst', scale * q, scale * k)
    m = torch.softmax(m, dim=-1)
    a = torch.einsum('bst, bct -> bcs', m, v)
    a = a.reshape(b, ch, h, w)
    return a

class UNetBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 emb_channels=None, 
                 is_up=False, 
                 is_down=False, 
                 use_att=False,
                 norm_layer=smart_group_norm,
                 act_layer=nn.SiLU,
                 down_layer=Pooling,
                 up_layer=Upsampling):
        super().__init__()
        self.resblock = BasicResBlock(in_channels, out_channels, emb_channels, 
                                      is_up=is_up, is_down=is_down,
                                      norm_layer=norm_layer, act_layer=act_layer,
                                      down_layer=down_layer, up_layer=up_layer)
        self.attblock = AttentionBlock(out_channels, norm_layer=norm_layer) \
                            if not is_up and not is_down and use_att else nn.Identity()
    
    def forward(self, x, t=None):
        return self.attblock(self.resblock(x, t))

    
class UNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 base_channels, 
                 out_channels, 
                 channels_mult, 
                 num_blocks, 
                 use_attentions,
                 norm_layer=smart_group_norm,
                 act_layer=nn.SiLU,
                 down_layer=Pooling,
                 up_layer=Upsampling):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.channels_mult = channels_mult
        self.num_blocks = num_blocks
        self.use_attentions = use_attentions
        emb_channels = 4 * base_channels
        
        self.emb_layers = nn.Sequential(
            nn.Linear(base_channels, emb_channels),
            act_layer(),
            nn.Linear(emb_channels, emb_channels),
        )
        
        _in_chs = []
        self.in_layers = nn.Sequential(
            conv3x3(in_channels, base_channels)
        )
        _ch = base_channels
        _in_chs.append(_ch)
        
        self.in_blocks = nn.ModuleList()
        
        for i, (mul, nb, use_att) in enumerate(zip(channels_mult, num_blocks, use_attentions[:-1])):
            for j in range(nb):
                self.in_blocks.append(
                    UNetBlock(_ch, base_channels * mul, emb_channels,
                              is_down = i != len(num_blocks) - 1 and j == nb - 1,
                              use_att = use_att,
                              norm_layer = norm_layer,
                              act_layer = act_layer,
                              down_layer = down_layer,
                              up_layer = up_layer),
                )
                _ch = base_channels * mul
                _in_chs.append(_ch)
        
        self.mid_blocks = nn.ModuleList([
            UNetBlock(_ch, _ch, emb_channels, 
                      use_att=use_attentions[-1],
                      norm_layer = norm_layer,
                      act_layer = act_layer,
                      down_layer = down_layer,
                      up_layer = up_layer),
            BasicResBlock(_ch, _ch, emb_channels,
                          norm_layer = norm_layer,
                          act_layer = act_layer,
                          down_layer = down_layer,
                          up_layer = up_layer)
        ])
        
        self.out_blocks = nn.ModuleList()
        for i, (mul, nb, use_att) in enumerate(zip(channels_mult[::-1], num_blocks[::-1], use_attentions[-2::-1])):
            for j in range(nb):
                self.out_blocks.append(
                    UNetBlock(_ch + _in_chs.pop(), base_channels * mul, emb_channels, 
                             is_up = i != 0 and j == 0,
                             use_att=use_att,
                             norm_layer = norm_layer,
                             act_layer = act_layer,
                             down_layer = down_layer,
                             up_layer = up_layer)
                )
                _ch = base_channels * mul
        
        _ch = _ch + _in_chs.pop()
        self.out_layers = nn.Sequential(
            norm_layer(_ch),
            act_layer(),
            zero_module(conv3x3(_ch, out_channels))
        )
    
    def forward(self, x, t=None):
        t_emb = timestep_embedding(t, self.base_channels)
        t_emb = self.emb_layers(t_emb)
        latents = []
        
        x = self.in_layers(x)
        latents.append(x)
        
        for block in self.in_blocks:
            x = block(x, t_emb)
            latents.append(x)
        
        for block in self.mid_blocks:
            x = block(x, t_emb)
            
        for block in self.out_blocks:
            h = latents.pop()
            x = torch.cat([x, h], dim=1)
            x = block(x, t_emb)
        
        x = torch.cat([x, latents.pop()], dim=1)
        x = self.out_layers(x)
        
        return x