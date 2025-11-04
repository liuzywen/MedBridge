import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrUpBlock


class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout=0, max_len:int=50000) -> None:

        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):

        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]

class GuideDecoderLayer_ABP(nn.Module):

    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=24, embed_dim:int=768):

        super(GuideDecoderLayer_ABP, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.self_attn_normtext = nn.LayerNorm(in_channels)
        self.cross_attn_normtext = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
        self.self_attntext = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.cross_attntext = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim,in_channels),
            nn.LeakyReLU(),
        )

        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels,max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm1text = nn.LayerNorm(in_channels)
        self.norm2text = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(1.421),requires_grad=True)


    def forward(self,x,txt):

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''

        txt = self.text_project(txt)

        # Self-Attention  X
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2

        # Self-Attention  TXT
        txt2 = self.norm1text(txt)
        qtext = ktext = self.txt_pos(txt2)
        txt2 = self.self_attntext(qtext, ktext, value=txt2)[0]
        txt2 = self.self_attn_normtext(txt2)
        txt = txt + txt2

        # Cross-Attention x
        vis2 = self.norm2(vis)
        vis2,attationvis = self.cross_attn(query=self.vis_pos(vis2),
                                   key=self.txt_pos(txt),
                                   value=txt)
        vis2 = self.cross_attn_norm(vis2)
        vis1 = vis + self.scale*vis2

        # Cross-Attention txt
        txt2 = self.norm2text(txt)
        txt2, attationtxt= self.cross_attntext(query=self.txt_pos(txt2),
                                  key=self.vis_pos(vis),
                                  value=vis)
        txt2 = self.cross_attn_normtext(txt2)
        txt1 = txt + self.scale * txt2

        return vis1,txt1
class promptLayer(nn.Module):

    def __init__(self, in_channels:int, output_text_len:int=4, input_text_len:int=4, embed_dim:int=768):

        super(promptLayer, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels,max_len=4)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.scale = nn.Parameter(torch.tensor(1.421),requires_grad=True)


    def forward(self,x,txt):

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        txt = self.text_project(txt)

        # Self-Attention
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2

        # Cross-Attention，并行
        vis2 = self.norm2(vis)
        txt2, _ = self.cross_attn(query=self.txt_pos(txt),
                                  key=self.vis_pos(vis2),
                                  value=vis2)
        txt2 = self.cross_attn_norm(txt2)
        txt2 = vis + self.scale * txt2


        return txt2
class self_Layer(nn.Module):

    def __init__(self, in_channels:int):

        super(self_Layer, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)

        self.vis_pos = PositionalEncoding(in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.dropout1 = nn.Dropout(0.1)


    def forward(self,x):
        h=x.shape[-1]
        x = rearrange(x, 'B C H W -> B (H W) C')

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        # Self-Attention
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + self.dropout1(vis2)
        vis = rearrange(vis, 'B (H W) C -> B C H W', H=h, W=h)
        return vis


class GuideDecoderLayer(nn.Module):

    def __init__(self, in_channels: int, output_text_len: int, input_text_len: int = 26, embed_dim: int = 768):

        super(GuideDecoderLayer, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels, max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout1 = nn.Dropout(0.1)
        self.scale = nn.Parameter(torch.tensor(1.00), requires_grad=True)

        # self.query_projector1 = nn.Sequential(nn.LayerNorm(embed_dim),
        #                                       nn.Linear(embed_dim, in_channels))
        # self.key_projector1 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                     nn.Linear(in_channels, in_channels))
        # self.value_projector1 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                       nn.Linear(in_channels, in_channels))
        #
        # self.query_projector2 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                       nn.Linear(in_channels, in_channels))
        # self.key_projector2 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                     nn.Linear(in_channels, in_channels))
        # self.value_projector2 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                   nn.Linear(in_channels, in_channels))
        #
        # self.text_project1 = nn.LayerNorm(in_channels)
        # self.text_norm = nn.LayerNorm(in_channels)

    def forward(self, x, txt):

        if txt is not None:
            h = x.shape[-1]
            x = rearrange(x, 'B C H W -> B (H W) C')
            # Self-Attention
            vis2 = self.norm1(x)
            q = k = self.vis_pos(vis2)
            vis2, att0 = self.self_attn(q, k, value=vis2)
            vis2 = self.self_attn_norm(vis2)
            vis = x + self.dropout1(vis2)

            txt = self.text_project(txt)
            vis2 = self.norm2(vis)
            vis2, ATT = self.cross_attn(query=self.vis_pos(vis2),
                                        key=self.txt_pos(txt),
                                        value=txt)
            vis2 = self.cross_attn_norm(vis2)
            vis = vis + self.scale * vis2
            vis = rearrange(vis, 'B (H W) C -> B C H W', H=h, W=h)


            return vis,txt
        else:
            h = x.shape[-1]
            x = rearrange(x, 'B C H W -> B (H W) C')
            # Self-Attention
            vis2 = self.norm1(x)
            q = k = self.vis_pos(vis2)
            vis2, att0 = self.self_attn(q, k, value=vis2)
            vis2 = self.self_attn_norm(vis2)
            vis = x + self.dropout1(vis2)
            vis = rearrange(vis, 'B (H W) C -> B C H W', H=h, W=h)
            return vis


class GuideDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, spatial_size, text_len) -> None:

        super().__init__()

        self.guide_layer = GuideDecoderLayer(in_channels, text_len)  # for skip
        self.decoder = UnetrUpBlock(2, in_channels, out_channels, 3, 2, norm_name='BATCH')


    def forward(self, vis, skip_vis, txt):
        if txt is not None:
            vis,txt = self.guide_layer(vis, txt)
            output = self.decoder(vis, skip_vis)
        else:
            vis = self.guide_layer(vis, txt)
            output = self.decoder(vis, skip_vis)

        return output,txt

class GuideDecoder1(nn.Module):

    def __init__(self, in_channels, out_channels, spatial_size, text_len) -> None:

        super().__init__()

        self.guide_layer = GuideDecoderLayer1(in_channels, text_len)  # for skip
        self.decoder = UnetrUpBlock(2, in_channels, out_channels, 3, 2, norm_name='BATCH')


    def forward(self, vis, skip_vis, txt):
        if txt is not None:
            vis,txt = self.guide_layer(vis, txt)
            output = self.decoder(vis, skip_vis)
        else:
            vis = self.guide_layer(vis, txt)
            output = self.decoder(vis, skip_vis)

        return output,txt

class GuideDecoderLayer1(nn.Module):

    def __init__(self, in_channels: int, output_text_len: int, input_text_len: int = 24, embed_dim: int = 768):

        super(GuideDecoderLayer1, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels, max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout1 = nn.Dropout(0.1)
        self.scale = nn.Parameter(torch.tensor(1.00), requires_grad=True)

        # self.query_projector1 = nn.Sequential(nn.LayerNorm(embed_dim),
        #                                       nn.Linear(embed_dim, in_channels))
        # self.key_projector1 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                     nn.Linear(in_channels, in_channels))
        # self.value_projector1 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                       nn.Linear(in_channels, in_channels))
        #
        # self.query_projector2 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                       nn.Linear(in_channels, in_channels))
        # self.key_projector2 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                     nn.Linear(in_channels, in_channels))
        # self.value_projector2 = nn.Sequential(nn.LayerNorm(in_channels),
        #                                   nn.Linear(in_channels, in_channels))
        #
        # self.text_project1 = nn.LayerNorm(in_channels)
        # self.text_norm = nn.LayerNorm(in_channels)

    def forward(self, x, txt):

        if txt is not None:
            h = x.shape[-1]
            x = rearrange(x, 'B C H W -> B (H W) C')
            # Self-Attention
            vis2 = self.norm1(x)
            q = k = self.vis_pos(vis2)
            vis2, att0 = self.self_attn(q, k, value=vis2)
            vis2 = self.self_attn_norm(vis2)
            vis = x + self.dropout1(vis2)

            txt = self.text_project(txt)
            vis2 = self.norm2(vis)
            vis2, ATT = self.cross_attn(query=self.vis_pos(vis2),
                                        key=self.txt_pos(txt),
                                        value=txt)
            vis2 = self.cross_attn_norm(vis2)
            vis = vis + self.scale * vis2
            vis = rearrange(vis, 'B (H W) C -> B C H W', H=h, W=h)


            return vis,txt
        else:
            h = x.shape[-1]
            x = rearrange(x, 'B C H W -> B (H W) C')
            # Self-Attention
            vis2 = self.norm1(x)
            q = k = self.vis_pos(vis2)
            vis2, att0 = self.self_attn(q, k, value=vis2)
            vis2 = self.self_attn_norm(vis2)
            vis = x + self.dropout1(vis2)
            vis = rearrange(vis, 'B (H W) C -> B C H W', H=h, W=h)
            return vis