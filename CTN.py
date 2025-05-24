from torch import nn 
import torch 

from einops import rearrange, repeat
import math 


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x * wei + residual * (1 - wei)
        return xo




class CNN_Trans(nn.Module):
    def __init__(self, patch_size=2,  dim=16, depth=1, heads=4, mlp_dim=16, 
                         channels = 3, dim_head = 4, dropout = 0., emb_dropout = 0.):
        super().__init__()

        # convolutional branch
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(), 
                                   nn.Dropout(0.)
                                   )
        
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.Dropout(0.)
                                   )
        # self.conv3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, padding=1),
        #                            nn.BatchNorm2d(8),
        #                            nn.ReLU(),
        #                            nn.Dropout(0.)
        #                            )
        

        self.avgpool = nn.AvgPool2d(2)

        

        # transformer branch 
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, padding=0, stride=patch_size)
        )
        num_patches = 54
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # self.fusion = nn.Sequential(
        #     nn.Conv2d(32, 16, 1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 4, 1),
        #     nn.BatchNorm2d(4),
        #     # nn.ReLU()
        # )

        self.fusion = AFF(channels=16, r=2)

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(
            nn.Linear(874, 512),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
 
        
    def forward(self, img, aux):
        B,_,_,_ = img.shape

        img = nn.functional.pad(img, (0,1,0,1))

        img_c = self.conv1(img)
        img_c = self.conv2(img_c) 
        img_c = self.avgpool(img_c)


        
        img_t = self.to_patch_embedding(img).flatten(2).transpose(1,2)

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # img = torch.cat((cls_tokens, img), dim=1)
        img_t += self.pos_embedding
        # img_t = self.dropout(img_t)
        img_t = self.transformer(img_t).transpose(-1,-2).contiguous().view(B,16,9,6)

        # out = torch.cat([img_c, img_t], dim=1)
        # out = self.fusion(out)
        # out = img_c + img_t
        out = self.fusion(img_c, img_t)
        out = self.flatten(out)
        out = torch.cat([out, aux], dim=1)
        out = self.fc3(self.fc2(self.fc1(out)))

        return out




if __name__=='__main__':
    img = torch.randn(12,3,17,11)
    aux = torch.randn(12,10)
    model = CNN_Trans()
    out = model(img, aux)
    print(out.shape)
