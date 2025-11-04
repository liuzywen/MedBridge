import torch
import random
import cv2
import torch.nn as nn
from einops import rearrange, repeat
from .layerss import GuideDecoder,GuideDecoder1
from .layers_multimask import GuideDecoder_multimask
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
from .SAM2UNet import SAM2UNet,SAM2UNet_RFB
import open_clip
import torch.nn.functional as F
from util.layerss import promptLayer,GuideDecoderLayer,self_Layer
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from util.segment_anything.modeling.image_encoder_layer_wise import ImageEncoderViT
from util.segment_anything.modeling.prompt_encoder import PromptEncoder
from util.segment_anything.modeling.mask_decoder import MaskDecoder
from util.segment_anything.modeling.transformer import TwoWayTransformer
from functools import partial
from util.assemFormer import AssembleFormer
from util.mcattn import  MoCAttention
from util.ASPP import ASPP
from util.fgModules import FGBottleneck as NeckBlock
from util.fgModules import CSLayer ,FGLink
from util.layers import StochasticDepth
from util.utils import pair, setMethod, callMethod
from monai.networks.blocks.unetr_block import UnetrUpBlock
from util.llmblock.llm4seg import LLM4Seg
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gaussian_kde
class BERTModelprompt(nn.Module):

    def __init__(self, bert_type, project_dim,num_prompts=20):

        super(BERTModelprompt, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.prompts = nn.Parameter(torch.randn(num_prompts, 768))
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask,prompt):
        # 创建新的 input_ids 和 attention_mask 以适应拼接后的嵌入
        batch_size, seq_len = input_ids.shape
        if prompt is True:
            new_input_ids = torch.cat([input_ids,torch.full((batch_size, self.prompts.shape[0]), self.model.config.pad_token_id,
                                                  dtype=input_ids.dtype, device=input_ids.device)], dim=1)
            new_attention_mask = torch.cat([attention_mask,torch.ones(batch_size, self.prompts.shape[0], dtype=attention_mask.dtype,
                                                       device=attention_mask.device)], dim=1)

            # 将拼接后的嵌入送入 BERT 模型
            output = self.model(input_ids=new_input_ids, attention_mask=new_attention_mask, output_hidden_states=True,
                                return_dict=True)

        else:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                return_dict=True)
        cls_token = None
        return {'feature':output['hidden_states'],'project':cls_token}
class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}

class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder2 = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModelprompt(bert_type, project_dim)

        spatial_dim = [7, 14, 28, 56]  # 224*224
        # spatial_dim = [16, 32, 64, 128]
        feature_dim = [768, 384, 192, 96]

        self.conv_linear = nn.Sequential(nn.GELU(),
                                         nn.Linear(1536, 768, bias=False))
        self.conv_2 = nn.Sequential(nn.AdaptiveAvgPool2d(2),
                                    nn.Conv2d(768, 768, 2, bias=False))
        self.conv_3 = nn.Sequential(nn.AdaptiveAvgPool2d(3),
                                    nn.Conv2d(768, 768, 3, bias=False))
        self.conv_4 = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                     nn.Conv2d(768, 768, 4, bias=False))
        self.conv_5 = nn.Sequential(nn.AdaptiveAvgPool2d(5),
                                     nn.Conv2d(768, 768, 5, bias=False))
        self.conv_6 = nn.Sequential(nn.AdaptiveAvgPool2d(6),
                                     nn.Conv2d(768, 768, 6, bias=False))
        self.conv_7 = nn.Conv2d(768, 768, 7, bias=False)

        self.query_projector = nn.Sequential(nn.LayerNorm(768),
                                             nn.Linear(768, 768))
        self.key_projector = nn.Sequential(nn.LayerNorm(768),
                                           nn.Linear(768, 768))
        self.value_projector = nn.Sequential(nn.LayerNorm(768),
                                             nn.Linear(768, 768))

        #
        # self.conv_linear = nn.Sequential(nn.GELU(),
        #                                  nn.Linear(1536, 768, bias=False))
        # self.conv_2 = nn.Sequential(nn.AdaptiveAvgPool2d(2),
        #                             nn.Conv2d(768, 768, 2, bias=False))
        # self.conv_3 = nn.Sequential(nn.AdaptiveAvgPool2d(4),
        #                             nn.Conv2d(768, 768, 4, bias=False))
        # self.conv_4 = nn.Sequential(nn.AdaptiveAvgPool2d(7),
        #                             nn.Conv2d(768, 768, 7, bias=False))
        # self.conv_5 = nn.Sequential(nn.AdaptiveAvgPool2d(10),
        #                             nn.Conv2d(768, 768, 10, bias=False))
        # self.conv_6 = nn.Sequential(nn.AdaptiveAvgPool2d(13),
        #                             nn.Conv2d(768, 768, 13, bias=False)
        # self.conv_7 = nn.Conv2d(768, 768, 16, bias=False)

        self.llm4seg1 = LLM4Seg(unfreeze=False, need_init=False, mode="Instruct", channel=feature_dim[0], layer=14,hw=spatial_dim[0] * spatial_dim[0]+6)
        self.decoder16 = GuideDecoder1(feature_dim[0],feature_dim[1],spatial_dim[0],24)
        self.decoder8 = GuideDecoder1(feature_dim[1],feature_dim[2],spatial_dim[1],12)
        self.decoder4 = GuideDecoder1(feature_dim[2],feature_dim[3],spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)
        # 新增缓存机制
        self.register_buffer('cached_text_embed_LLM', None)  # 注册为 buffer，会随模型保存/加载
        self.has_cached = False  # 标记是否已缓存

    def compute_text_embed_LLM(self, text_llm):
        """计算 text_embed_LLM 并缓存（仅在训练时调用）"""
        text_embed_llms = []
        for single_text_llm in text_llm:
            text_output_llm = self.text_encoder(
                single_text_llm['input_ids'],
                single_text_llm['attention_mask'],
                False
            )
            text_embed_llms.append(text_output_llm['feature'][-1])

        text_embed_LLM = torch.mean(torch.stack(text_embed_llms), dim=0)
        self.cached_text_embed_LLM = text_embed_LLM  # 缓存结果
        self.has_cached = True
        return text_embed_LLM
    def forward(self, data):

        image, text,text_llm = data
        B=image.shape[0]
        if image.shape[1] == 1:
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output2 = self.encoder2(image)
        image_features2, image_project2 = image_output2['feature'], image_output2['project']
        image_features = image_features2[1:]

        text_output = self.text_encoder(text['input_ids'], text['attention_mask'],True)
        text_embeds, text_project = text_output['feature'], text_output['project']

        if self.training or not self.training:
            # 训练模式：首次计算并缓存，后续直接使用缓存
            if not self.has_cached:
                text_embed_LLM = self.compute_text_embed_LLM(text_llm)
                text_embed_LLM=text_embed_LLM.repeat(B, 1, 1)
            else:
                text_embed_LLM = self.cached_text_embed_LLM
                text_embed_LLM = text_embed_LLM.repeat(B, 1, 1)
        x0 = image_features[0]
        x1 = image_features[1]
        x2 = image_features[2]
        x3 = image_features[3]

        res = []
        res.append(self.conv_2(x3).squeeze(-1).permute(0, 2, 1))
        res.append(self.conv_3(x3).squeeze(-1).permute(0, 2, 1))
        res.append(self.conv_4(x3).squeeze(-1).permute(0, 2, 1))
        res.append(self.conv_5(x3).squeeze(-1).permute(0, 2, 1))
        res.append(self.conv_6(x3).squeeze(-1).permute(0, 2, 1))
        res.append(self.conv_7(x3).squeeze(-1).permute(0, 2, 1))
        s = torch.cat(res, dim=1)

        B, C, H, W = x3.shape
        x3 = x3.flatten(2).transpose(1, 2)
        embed_query = self.query_projector(s)
        embed_key = self.key_projector(x3)
        embed_value = self.value_projector(x3)
        embed_att = embed_query @ (embed_key.transpose(-1, -2) / (embed_key.shape[-1] ** 0.5))
        embed_att = embed_att.nan_to_num()
        embed_feat = (embed_att.softmax(-1) @ embed_value)
        embed_fuse = torch.cat([s, embed_feat], dim=-1)
        x3 = torch.cat([self.conv_linear(embed_fuse), x3], dim=1)
        x_s = self.llm4seg1(x3)
        xs = x_s[:,6:,:].transpose(1, 2).reshape(B, C, H, W)
        os16,_ = self.decoder16(xs, x2,text_embeds[-1])#[1, 384, 14, 14]
        os8 ,_= self.decoder8(os16, x1, text_embeds[-1])#[1, 192, 28, 28]
        os4,_ = self.decoder4(os8, x0, text_embeds[-1])

        os1 = self.decoder1(os4)
        out = self.out(os1).sigmoid()
        return out,text_embeds[-1],text_embed_LLM
