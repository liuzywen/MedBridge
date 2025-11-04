import json
import os
import cv2
import torch
import pandas as pd
from monai.transforms import (Compose, Lambdad, NormalizeIntensityd,RandCoarseShuffled,RandRotated,RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from util.prompt_templates import BIOMEDCOOP_TEMPLATES,LLM_text,LLM_textk
class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train',image_size=[224,224]):

        super(QaTa, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['Description'])

        if mode == 'train':
            self.image_list = self.image_list[:int(0.8*len(self.image_list))]
            self.caption_list = self.caption_list[:int(0.8*len(self.caption_list))]
        elif mode == 'valid':
            self.image_list = self.image_list[int(0.8*len(self.image_list)):]
            self.caption_list = self.caption_list[int(0.8*len(self.caption_list)):]
        else:
            pass   # for mode is 'test'
        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path,'Images',self.image_list[idx].replace('mask_',''))
        image_path=self.image_list[idx].replace('mask_','')
        gt = os.path.join(self.root_path,'Ground-truths', self.image_list[idx])
        caption = self.caption_list[idx]
        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24,
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,gt,token,mask = data['image'],data['gt'],data['token'],data['mask']
        gt = torch.where(gt==255,1,0)
        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)}

        llm_texts = []
        for i in range(10):
            llm_text = LLM_text["covid"][i]
            llmtoken_output0 = self.tokenizer.encode_plus(llm_text.strip(), padding='max_length',
                                                          max_length=44,
                                                          truncation=True,
                                                          return_attention_mask=True,
                                                          return_tensors='pt')
            llmtoken0, llmmask0 = llmtoken_output0['input_ids'], llmtoken_output0['attention_mask']
            data_llm = {
                "llmtoken": llmtoken0,  # 文本token (列表或字符串)
                "mask": llmmask0  # 掩码数据 (H,W)
            }
            # 正确转换方式
            transform = ToTensord(keys=["llmtoken", "mask"])  # 只转换图像和掩码
            data_transformedllm = transform(data_llm)
            llmtoken0 = data_transformedllm["llmtoken"]
            llmmask0 = data_transformedllm["mask"]
            llmtoken = {'input_ids': llmtoken0.squeeze(dim=0).long(), 'attention_mask': llmmask0.squeeze(dim=0).long()}
            llm_texts.append(llmtoken)

        return ([image, text,llm_texts], gt,image_path)

    def transform(self,image_size=[224,224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image","gt","token","mask"]),

            ])

        return trans


class MOS(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224, 224]):

        super(MOS, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        self.image_list = list(self.data['Image'])
        self.caption_list = list(self.data['text'])

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path, 'frames', self.image_list[idx])
        image_path= self.image_list[idx]
        gt = os.path.join(self.root_path, 'masks', self.image_list[idx])
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                  max_length=24,
                                                  truncation=True,
                                                  return_attention_mask=True,
                                                  return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = {'image': image, 'gt': gt, 'token': token, 'mask': mask}
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']
        gt = torch.where(gt == 255, 1, 0)
        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)}

        llm_texts = []
        for i in range(50):
            llm_text = LLM_text["covid"][i]
            llmtoken_output0 = self.tokenizer.encode_plus(llm_text.strip(), padding='max_length',
                                                          max_length=44,
                                                          truncation=True,
                                                          return_attention_mask=True,
                                                          return_tensors='pt')
            llmtoken0, llmmask0 = llmtoken_output0['input_ids'], llmtoken_output0['attention_mask']
            data_llm = {
                "llmtoken": llmtoken0,  # 文本token (列表或字符串)
                "mask": llmmask0  # 掩码数据 (H,W)
            }
            # 正确转换方式
            transform = ToTensord(keys=["llmtoken", "mask"])  # 只转换图像和掩码
            data_transformedllm = transform(data_llm)
            llmtoken0 = data_transformedllm["llmtoken"]
            llmmask0 = data_transformedllm["mask"]
            llmtoken = {'input_ids': llmtoken0.squeeze(dim=0).long(), 'attention_mask': llmmask0.squeeze(dim=0).long()}
            llm_texts.append(llmtoken)

        return ([image, text,llm_texts], gt,image_path)

    def transform(self, image_size=[224, 224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                RandZoomd(['image', 'gt'], min_zoom=0.95, max_zoom=1.2, mode=["bicubic", "nearest"], prob=0.1),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),
            ])

        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),

            ])

        return trans

class Kvasir_SEG(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224, 224]):

        super(Kvasir_SEG, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = json.load(f)
        self.image_list = [item['img_name'] for item in self.data]
        self.caption_list = [item['prompts']['p6'] for item in self.data]

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path, 'images', self.image_list[idx])
        image_path= self.image_list[idx]
        gt = os.path.join(self.root_path, 'masks', self.image_list[idx])
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                  max_length=24,
                                                  truncation=True,
                                                  return_attention_mask=True,
                                                  return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = {'image': image, 'gt': gt, 'token': token, 'mask': mask}
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']
        gt = torch.where(gt == 255, 1, 0)
        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)}

        llm_texts = []
        for i in range(50):
            llm_text = LLM_textk["covidk"][i]
            llmtoken_output0 = self.tokenizer.encode_plus(llm_text.strip(), padding='max_length',
                                                          max_length=44,
                                                          truncation=True,
                                                          return_attention_mask=True,
                                                          return_tensors='pt')
            llmtoken0, llmmask0 = llmtoken_output0['input_ids'], llmtoken_output0['attention_mask']
            data_llm = {
                "llmtoken": llmtoken0,  # 文本token (列表或字符串)
                "mask": llmmask0  # 掩码数据 (H,W)
            }
            # 正确转换方式
            transform = ToTensord(keys=["llmtoken", "mask"])  # 只转换图像和掩码
            data_transformedllm = transform(data_llm)
            llmtoken0 = data_transformedllm["llmtoken"]
            llmmask0 = data_transformedllm["mask"]
            llmtoken = {'input_ids': llmtoken0.squeeze(dim=0).long(), 'attention_mask': llmmask0.squeeze(dim=0).long()}
            llm_texts.append(llmtoken)

        return ([image, text,llm_texts], gt,image_path)

    def transform(self, image_size=[224, 224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                RandZoomd(['image', 'gt'], min_zoom=0.95, max_zoom=1.2, mode=["bicubic", "nearest"], prob=0.1),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),
            ])

        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),

            ])

        return trans

class BKAI(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[224, 224]):

        super(BKAI, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = json.load(f)
        self.image_list = [item['img_name'] for item in self.data]
        self.caption_list = [item['prompts']['p6'] for item in self.data]

        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path, 'images', self.image_list[idx])
        image_path= self.image_list[idx]
        gt = os.path.join(self.root_path, 'masks', self.image_list[idx])
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                  max_length=24,
                                                  truncation=True,
                                                  return_attention_mask=True,
                                                  return_tensors='pt')
        token, mask = token_output['input_ids'], token_output['attention_mask']

        data = {'image': image, 'gt': gt, 'token': token, 'mask': mask}
        data = trans(data)

        image, gt, token, mask = data['image'], data['gt'], data['token'], data['mask']
        gt = torch.where(gt == 255, 1, 0)
        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)}

        llm_texts = []
        for i in range(50):
            llm_text = LLM_text["covid"][i]
            llmtoken_output0 = self.tokenizer.encode_plus(llm_text.strip(), padding='max_length',
                                                          max_length=44,
                                                          truncation=True,
                                                          return_attention_mask=True,
                                                          return_tensors='pt')
            llmtoken0, llmmask0 = llmtoken_output0['input_ids'], llmtoken_output0['attention_mask']
            data_llm = {
                "llmtoken": llmtoken0,  # 文本token (列表或字符串)
                "mask": llmmask0  # 掩码数据 (H,W)
            }
            # 正确转换方式
            transform = ToTensord(keys=["llmtoken", "mask"])  # 只转换图像和掩码
            data_transformedllm = transform(data_llm)
            llmtoken0 = data_transformedllm["llmtoken"]
            llmmask0 = data_transformedllm["mask"]
            llmtoken = {'input_ids': llmtoken0.squeeze(dim=0).long(), 'attention_mask': llmmask0.squeeze(dim=0).long()}
            llm_texts.append(llmtoken)

        return ([image, text,llm_texts], gt,image_path)

    def transform(self, image_size=[224, 224]):

        if self.mode == 'train':  # for training mode
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                RandZoomd(['image', 'gt'], min_zoom=0.95, max_zoom=1.2, mode=["bicubic", "nearest"], prob=0.1),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),
            ])

        else:  # for valid and test mode: remove random zoom
            trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader'),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt", "token", "mask"]),

            ])

        return trans
# def make_transforms(__C, image_set ):
#
#     imsize = __C.INPUT_SHAPE[0]
#
#     if image_set == 'train':
#         scales = []
#         if __C.AUG_SCALE:
#             # scales=[256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608]
#             for i in range(7):
#                 scales.append(imsize - 32 * i)
#         else:
#             scales = [imsize]
#
#         if __C.AUG_CROP:
#             crop_prob = 0.5
#         else:
#             crop_prob = 0.
#
#         return T.Compose([
#             T.RandomSelect(
#                 T.RandomResize(scales),
#                 T.Compose([
#                     T.RandomResize([400, 500, 600], with_long_side=False),
#                     T.RandomSizeCrop(384, 600),
#                     T.RandomResize(scales),
#                 ]),
#                 p=crop_prob
#             ),
#             T.ColorJitter(0.4, 0.4, 0.4),
#             T.GaussianBlur(aug_blur=__C.AUG_BLUR),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize, aug_translate=__C.AUG_TRANSLATE)
#         ])
#
#     if image_set in ['val', 'test', 'testA', 'testB']:
#         return T.Compose([
#             T.RandomResize([imsize]),
#             T.ToTensor(),
#             T.NormalizeAndPad(mean=__C.MEAN,std=__C.STD,size=imsize),
#         ])
#
#     raise ValueError(f'unknown {image_set}')
# class MOSMedTestDataSet(Data.Dataset):
#     def __init__(self, __C,split):
#         super(MOSMedTestDataSet, self).__init__()
#         self.__C = __C
#         self.split=split
#         assert  __C.DATASET in ["MOSMed",'covid19','refcoco', 'refcoco+', 'refcocog','referit','vg','merge']
#         # --------------------------
#         # ---- Raw data loading ---
#         # --------------------------
#         # stat_refs_list=json.load(open(__C.ANN_PATH[__C.DATASET], 'r'))
#         df = pd.read_csv(open(__C.ANN_PATH[__C.DATASET+"test"]))
#         # total_refs_list=[]
#         '''if __C.DATASET in ['vg','merge']:
#             total_refs_list = json.load(open(__C.ANN_PATH['merge'], 'r'))+json.load(open(__C.ANN_PATH['refcoco+'], 'r'))+json.load(open(__C.ANN_PATH['refcocog'], 'r'))+json.load(open(__C.ANN_PATH['refcoco'], 'r'))'''
#         self.lang_enc = __C.LANG_ENC
#         # splits=split.split('+')
#         self.refs_anno=[]
#         # for split_ in splits:
#         #     self.refs_anno+= stat_refs_list[split_]
#         for i, row in df.iterrows():
#             gt_path = row['Image']  # 图片文件名所在列的列名
#             text = row['text']  # 文本所在列的列名
#             file_path = gt_path
#             gt_path = os.path.join(__C.MASK_PATH[__C.DATASET], gt_path)
#             file_path = os.path.join(__C.IMAGE_PATH[__C.DATASET], file_path)
#             self.refs_anno.append({'file_path': file_path,
#                                    'gt_path': gt_path,
#                                    'text': text})
#         if self.lang_enc == 'bert':
#             self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#         # refs=[]
#         #
#         # for split in stat_refs_list:
#         #     for ann in stat_refs_list[split]:
#         #         for ref in ann['refs']:
#         #             refs.append(ref)
#         # for split in total_refs_list:
#         #     for ann in total_refs_list[split]:
#         #         for ref in ann['refs']:
#         #             refs.append(ref)
#         self.image_path=__C.IMAGE_PATH[__C.DATASET]
#         self.mask_path=__C.MASK_PATH[__C.DATASET]
#         self.input_shape=__C.INPUT_SHAPE
#         self.flip_lr=__C.FLIP_LR if split=='train' else False
#         # Define run data size
#         self.data_size = len(self.refs_anno)
#
#         print(' ========== Dataset size:', self.data_size)
#         # ------------------------
#         # ---- Data statistic ----
#         # ------------------------
#         # Tokenize
#         # self.token_to_ix,self.ix_to_token, self.pretrained_emb, max_token = self.tokenize(stat_refs_list, __C.USE_GLOVE)
#         # self.token_size = self.token_to_ix.__len__()
#         # print(' ========== Question token vocab size:', self.token_size)
#         # #keys = list(self.token_to_ix.keys())
#         #print('keys:', len(keys))
#         #self.tokenizer.add_tokens(keys)
#         #self.tokenizer.save_pretrained('/data/huangxiaorui/SAM_research/SimREC_Reseach-TMM_version/vocab/bert_vocab_ori')
#         self.max_token = __C.MAX_TOKEN
#         # if self.max_token == -1:
#         #     self.max_token = max_token
#         print('Trimmed to:', self.max_token)#应该为30
#         print('Finished!')
#         print('')
#
#         # self.candidate_transforms ={}
#         # if  self.split == 'train':
#         #     if 'RandAugment' in self.__C.DATA_AUGMENTATION:
#         #         self.candidate_transforms['RandAugment']=RandAugment(2,9)
#         #     if 'ElasticTransform' in self.__C.DATA_AUGMENTATION:
#         #         self.candidate_transforms['ElasticTransform']=A.ElasticTransform(p=0.5)
#         #     if 'GridDistortion' in self.__C.DATA_AUGMENTATION:
#         #         self.candidate_transforms['GridDistortion']=A.GridDistortion(p=0.5)
#         #     if 'RandomErasing' in self.__C.DATA_AUGMENTATION:
#         #         self.candidate_transforms['RandomErasing']=transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8),
#         #                                                                       value="random")
#         self.transforms=make_transforms(__C,self.split)
#         #self.transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize(__C.MEAN, __C.STD)])
#         from torchvision import transforms
#         # self.image_preprocessor = transforms.Compose([
#         #     transforms.ToTensor(),
#         #     transforms.Resize((512, 512), interpolation=3),
#         #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         # ])
#
#
#     # def tokenize(self, stat_refs_list, use_glove):
#     #     token_to_ix = {
#     #         'PAD': 0,
#     #         'UNK': 1,
#     #         'CLS': 2,
#     #     }
#     #
#     #     spacy_tool = None
#     #     pretrained_emb = []
#     #     if use_glove:
#     #         spacy_tool = en_vectors_web_lg.load()
#     #         pretrained_emb.append(spacy_tool('PAD').vector)
#     #         pretrained_emb.append(spacy_tool('UNK').vector)
#     #         pretrained_emb.append(spacy_tool('CLS').vector)
#     #
#     #     max_token = 0
#     #     for split in stat_refs_list:
#     #         for ann in stat_refs_list[split]:
#     #             for ref in ann['refs']:
#     #                 words = re.sub(
#     #                     r"([.,'!?\"()*#:;])",
#     #                     '',
#     #                     ref.lower()
#     #                 ).replace('-', ' ').replace('/', ' ').split()
#     #
#     #                 if len(words) > max_token:
#     #                     max_token = len(words)
#     #
#     #                 for word in words:
#     #                     if word not in token_to_ix:
#     #                         token_to_ix[word] = len(token_to_ix)
#     #                         if use_glove:
#     #                             pretrained_emb.append(spacy_tool(word).vector)
#     #
#     #     pretrained_emb = np.array(pretrained_emb)
#     #     ix_to_token={}
#     #     for item in token_to_ix:
#     #         ix_to_token[token_to_ix[item]]=item
#     #
#     #     return token_to_ix, ix_to_token,pretrained_emb, max_token
#
#
#     def proc_ref(self, ref, token_to_ix, max_token):
#         ques_ix = np.zeros(max_token, np.int64)
#
#         words = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             ref.lower()
#         ).replace('-', ' ').replace('/', ' ').split()
#
#         for ix, word in enumerate(words):
#             if word in token_to_ix:
#                 ques_ix[ix] = token_to_ix[word]
#             else:
#                 ques_ix[ix] = token_to_ix['UNK']
#
#             if ix + 1 == max_token:
#                 break
#
#         return ques_ix
#
#     # ----------------------------------------------
#     # ---- Real-Time Processing Implementations ----
#     # ----------------------------------------------
#
#     def load_refs(self, idx):
#         ref = self.refs_anno[idx]['text']
#         return ref
#
#     def preprocess_info(self,img,mask,box,iid,lr_flip=False):
#         h, w, _ = img.shape
#         # img = img[:, :, ::-1]
#         imgsize=self.input_shape[0]
#         new_ar = w / h
#         if new_ar < 1:
#             nh = imgsize
#             nw = nh * new_ar
#         else:
#             nw = imgsize
#             nh = nw / new_ar
#         nw, nh = int(nw), int(nh)
#
#
#         dx = (imgsize - nw) // 2
#         dy = (imgsize - nh) // 2
#
#         img = cv2.resize(img, (nw, nh))
#         sized = np.ones((imgsize, imgsize, 3), dtype=np.uint8) * 127
#         sized[dy:dy + nh, dx:dx + nw, :] = img
#         info_img = (h, w, nh, nw, dx, dy,iid)
#
#         mask=np.expand_dims(mask,-1).astype(np.float32)
#         mask=cv2.resize(mask, (nw, nh))
#         mask=np.expand_dims(mask,-1).astype(np.float32)
#         sized_mask = np.zeros((imgsize, imgsize, 1), dtype=np.float32)
#         sized_mask[dy:dy + nh, dx:dx + nw, :]=mask
#         sized_mask=np.transpose(sized_mask, (2, 0, 1))
#         sized_box=label2yolobox(box,info_img,self.input_shape[0],lrflip=lr_flip)
#         return sized,sized_mask,sized_box, info_img
#
#     def load_img_feats(self, idx):
#         img_path=None
#         if self.__C.DATASET in ['refcoco','refcoco+','refcocog']:
#             img_path=os.path.join(self.image_path,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
#         elif self.__C.DATASET == 'covid19':
#             img_path = self.refs_anno[idx]['file_path']
#         elif self.__C.DATASET == 'MOSMed':
#             img_path = self.refs_anno[idx]['file_path']
#         elif self.__C.DATASET=='referit':
#             img_path = os.path.join(self.image_path, '%d.jpg' % self.refs_anno[idx]['iid'])
#         elif self.__C.DATASET=='vg':
#             img_path = os.path.join(self.image_path, self.refs_anno[idx]['url'])
#         elif self.__C.DATASET == 'merge':
#             if self.refs_anno[idx]['data_source']=='coco':
#                 iid='COCO_train2014_%012d.jpg'%int(self.refs_anno[idx]['iid'].split('.')[0])
#             else:
#                 iid=self.refs_anno[idx]['iid']
#             img_path = os.path.join(self.image_path,self.refs_anno[idx]['data_source'], iid)
#         else:
#             assert NotImplementedError
#
#         #image= cv2.imread(img_path)
#         image= Image.open(img_path).convert('RGB')
#         if self.__C.DATASET in ['MOSMed','covid19','refcoco','refcoco+','refcocog','referit']:
#             # mask= cv2.imread(self.refs_anno[idx]['gt_path'],cv2.IMREAD_GRAYSCALE)
#             # # 调整维度为 [h, w, 1]
#             # mask = np.expand_dims(mask, axis=-1)
#             mask = Image.open(self.refs_anno[idx]['gt_path']).convert('L')
#         else:
#             mask=np.zeros([image.shape[0],image.shape[1],1],dtype=np.float)
#
#         # box=np.array([self.refs_anno[idx]['bbox']])
#         box=None
#         # mask = Image.fromarray(mask * 255)
#         return image,mask,box,self.refs_anno[idx]['gt_path'],self.refs_anno[idx]['file_path']
#
#     def __getitem__(self, idx):
#         # print("__getitem__dataset")
#         ref_iter = self.load_refs(idx)#对应的文本
#         image_iter,mask_iter,gt_box_iter,mask_id,iid= self.load_img_feats(idx)
#         # print("vv1",image_iter)
#         # print("vv2", mask_iter)
#         # print("vv3", gt_box_iter)
#         # print("vv4", mask_id)
#         # print("vv5", iid)
#
#         w, h = image_iter.size
#         input_dict = {'img': image_iter,
#                       # 'box': box_xywh_to_xyxy(torch.from_numpy(gt_box_iter[0]).float()),
#                       'box': None,
#                       'mask': mask_iter,
#                       'text': ref_iter}
#         input_dict = self.transforms(input_dict)
#         examples = read_examples(input_dict['text'], idx)
#         features = convert_examples_to_features(
#             examples=examples, seq_length=self.max_token, tokenizer=self.tokenizer)
#         # print("vv3", input_dict['img'])
#         # print("vv4", mask_id)
#         # print("vv5", iid)
#         # input()
#         ref_iter = features[0].input_ids
#         ref_mask = features[0].input_mask
#         # print("vv5", ref_iter)
#         # print("vv5", ref_mask)
#         # input()
#         ref_iter = np.array(ref_iter)
#         ref_mask_iter = np.array(ref_mask)
#         # info_iter = [h, w, *input_dict['info_img'], iid]
#         info_iter = None
#         #image_iter, mask_iter, box_iter,info_iter=self.preprocess_info(image_iter,mask_iter,gt_box_iter.copy(),iid,flip_box)
#         return torch.from_numpy(ref_iter).long(),  input_dict['img'],  input_dict['mask'], mask_id,ref_mask_iter
#
#     def __len__(self):
#         return self.data_size
#
#     def shuffle_list(self, list):
#         random.shuffle(list)