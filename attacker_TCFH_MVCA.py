from  transformers import  CLIPModel
import numpy as np
import torch

torch.set_printoptions(profile="full")
from torch.autograd import Variable as V

import torch.nn as nn
from dct import *
import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random

import numpy as np

np.set_printoptions(threshold=np.inf)


from models.vit import VisionTransformer


from functools import partial
from models.xbert import BertConfig, BertModel

import torch
import torch.nn as nn
import numpy as np
from utils import *
import torch.nn.functional as F
import torch_dct as dct
import scipy.stats as st

# abc=0
import torch.nn.functional as F
import torch_dct as tdct
import scipy.stats as st
import torch.nn as nn
from torchvision.utils import save_image
from eda import eda

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
NUM_AUGS = [1]
PUNC_RATIO = 0.3


num_copies=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_block=3
def vertical_shift(   x):
    _, _, w, _ = x.shape
    step = np.random.randint(low=0, high=w, dtype=np.int32)
    return x.roll(step, dims=2)

def horizontal_shift(   x):
    _, _, _, h = x.shape
    step = np.random.randint(low=0, high=h, dtype=np.int32)
    return x.roll(step, dims=3)

def vertical_flip(   x):
    return x.flip(dims=(2,))

def horizontal_flip(   x):
    return x.flip(dims=(3,))

def rotate180(   x):
    return x.rot90(k=2, dims=(2, 3))

def scale(   x):
    return torch.rand(1)[0] * x

def resize(   x):
    """
    Resize the input
    """
    _, _, w, h = x.shape
    scale_factor = 0.8
    new_h = int(h * scale_factor) + 1
    new_w = int(w * scale_factor) + 1
    x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
    return x

def dct(x):
    """
    Discrete Fourier Transform
    """
    dctx = tdct.dct_2d(x)  # torch.fft.fft2(x, dim=(-2, -1))
    _, _, w, h = dctx.shape
    low_ratio = 0.4
    low_w = int(w * low_ratio)
    low_h = int(h * low_ratio)
    # dctx[:, :, -low_w:, -low_h:] = 0
    dctx[:, :, -low_w:, :] = 0
    dctx[:, :, :, -low_h:] = 0
    dctx = dctx  # *    mask.reshape(1, 1, w, h)
    idctx = tdct.idct_2d(dctx)
    return idctx

def add_noise(   x):
    return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)

def gkern(   kernel_size=3, nsig=3):
    x = np.linspace(-nsig, nsig, kernel_size)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return torch.from_numpy(stack_kernel.astype(np.float16)).to('cuda')

def drop_out(   x):
    return F.dropout2d(x, p=0.1, training=True)

def blocktransform( x, choice=-1):
    _, _, w, h = x.shape
    y_axis = [0, ] + np.random.choice(list(range(1, h)),    num_block - 1, replace=False).tolist() + [
        h, ]  #    num_block=3
    x_axis = [0, ] + np.random.choice(list(range(1, w)),    num_block - 1, replace=False).tolist() + [w, ]
    y_axis.sort()
    x_axis.sort()

    x_copy = x.clone()
    for i, idx_x in enumerate(x_axis[1:]):
        for j, idx_y in enumerate(y_axis[1:]):
            chosen = choice if choice >= 0 else np.random.randint(0, high=len(   op), dtype=np.int32)
            x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] =    op[chosen](
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

    return x_copy

def transform_SIA(x, **kwargs):
    """
    Scale the input for BlockShuffle
    """
    return torch.cat([   blocktransform(x) for _ in range(   num_copies)])


op = [resize,vertical_shift,horizontal_shift, vertical_flip,    horizontal_flip,
                      rotate180,    scale,    add_noise,    dct,    drop_out]

def insert_punctuation_marks(sentence, punc_ratio=0.1):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(punc_ratio * len(words) + 1))
	qs = random.sample(range(0, len(words)), q)

	for j, word in enumerate(words):
		if j in qs:
			new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line



class SGAttacker():
    def __init__(self, model, img_attacker, txt_attacker):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

    def attack(self, imgs, txts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):
        # original state
        with torch.no_grad():
            origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]
            img_embdding=origin_img_output['image_embed']
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,img_embdding=img_embdding)

        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']
            txt_embdding = txts_output['text_embed']
        adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img, device,
                                                       scales=scales, txt_embeds=txt_supervisions,txt_embdding=txt_embdding)

        with torch.no_grad():
            adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
            img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            img_embdding = origin_img_output['image_embed']
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,img_embdding=img_embdding)

        return adv_imgs, adv_txts


class ImageAttacker():
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.vision_proj_m = nn.Linear(768, 256)
        self.vision_proj = nn.Linear(768, 256)
        self.text_proj_m = nn.Linear(768, 256)
        self.visual_encoder_m = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.visual_encoder = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):#jubu,全局
        # torch.Size([4, 16, 256])
        #torch.Size([4, 256])
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim)  # (N * n_locals) * d
        m_n = m.reshape(-1, dim)  # N * d

        # print(temp)#0.97
        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
        u_p = torch.matmul(l, m.permute(0, 2, 1)).unsqueeze(2) / temp  # N * n_locals * 1 * 1

        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1 - temp_mask))

        u_n = torch.mm(m_n, l_n.t()) / temp
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1)  # N x N x n_locals x 1

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device)  # N*N*1*1
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1 - temp_mask))

        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return -loss / 10

    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img):
        device = adv_imgs_embeds.device

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1

        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos

        return loss
    def loss_i2i(self, queries, keys, temperature=0.1):
        b, device = queries.shape[0], queries.device  # 输入1的q形状应该是
        n = b * 2  # 520
        projs = torch.cat((queries, keys))
        logits = projs @ projs.t()

        mask = torch.eye(n, device=device).bool()
        logits = logits[~mask].reshape(n, n - 1)
        logits /= temperature

        labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        loss /= n
        return -loss

    def txt_guided_attack(self, model, imgs, txt2img, device, scales=None, txt_embeds=None, txt_embdding=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model=model.to(device)
        self.visual_encoder_m=self.visual_encoder_m.to(device)
        self.visual_encoder=self.visual_encoder.to(device)
        self.vision_proj_m=self.vision_proj_m.to(device)
        self.vision_proj=self.vision_proj.to(device)
        self.text_proj_m=self.text_proj_m.to(device)
        b, _, _, _ = imgs.shape

        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) + 1

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(
            device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        momentum = torch.zeros_like(adv_imgs)
        v = torch.zeros_like(adv_imgs).detach().to(device)
        alpha=0.5/255
        beta=3/2
        rho = 0.5
        noise = 0
        decay = 1.0

        for i in range(self.steps):
            adv_imgs.requires_grad_()
            nes_images = adv_imgs + decay * alpha * momentum
            scaled_imgs = self.get_scaled_imgs(nes_images, scales, device)



            if self.normalization is not None:
                adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
            else:
                adv_imgs_output = model.inference_image(scaled_imgs)
            adv_imgs_embeds = adv_imgs_output['image_feat']
            model.zero_grad()
            with torch.enable_grad():
                loss_list = []
                loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(scales_num):
                    loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], txt_embeds, txt2img)
                    loss_list.append(loss_item.item())
                    loss += loss_item
            print(loss, 'lossit')

            # adv_imgs = F.interpolate(adv_imgs, size=(384, 384), mode='bilinear', align_corners=False)
            # imgs = F.interpolate(imgs, size=(384, 384), mode='bilinear', align_corners=False)
            # image_embeds = self.visual_encoder(adv_imgs)
            # K_image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

            if self.normalization is not None:
                imgs_output = model.inference_image(self.normalization(imgs))
            else:
                imgs_output = model.inference_image(imgs)
            imgs_embeds = imgs_output['image_feat']#nx512
            if self.normalization is not None:
                adv_imgs_output_new = model.inference_image(self.normalization(adv_imgs))
            else:
                adv_imgs_output_new = model.inference_image(adv_imgs)
            adv_imgs_embeds_new = adv_imgs_output_new['image_feat']#nx512

            loss_item_i2i = self.loss_i2i(adv_imgs_embeds_new, imgs_embeds)
            loss_item_i2i=loss_item_i2i/2
            loss-=loss_item_i2i
            print(loss,'lossi2i')

            text_feat_m_l = F.normalize(self.text_proj_m(txt_embdding[:,0:,:]),dim=-1)
            loss_i2i_inMod_l = self.in_batch_g2l_loss(text_feat_m_l, adv_imgs_embeds, self.temp)##文本局部、图像全局
            loss -= loss_i2i_inMod_l
            print(loss,'lossi2i_inmodl')

            loss.backward()


            # adv_imgs_grad = adv_imgs.grad
            # grad=(adv_imgs_grad+v)/torch.mean(torch.abs(adv_imgs_grad+v),dim=(1,2,3),keepdim=True)
            # grad=grad+momentum*2   # momentum = 1 * momentum + adv_imgs_grad
            # momentum=grad          #perturbation = self.step_size * momentum.sign()
            # # Calculate Gradient Variance
            # GV_grad = torch.zeros_like(adv_imgs).detach().to(device)
            # for _ in range(5):
            #     neighbor_images = adv_imgs.detach() + torch.randn_like(
            #         adv_imgs
            #     ).uniform_(-self.eps * beta, self.eps * beta)
            #
            #     neighbor_images.requires_grad = True
            #     scaled_imgs = self.get_scaled_imgs(neighbor_images, scales, device)
            #
            #     if self.normalization is not None:
            #         adv_imgs_output1 = model.inference_image(self.normalization(scaled_imgs))
            #     else:
            #         adv_imgs_output1 = model.inference_image(scaled_imgs)
            #     adv_imgs_embeds1 = adv_imgs_output1['image_feat']
            #
            #     with torch.enable_grad():
            #         loss_list1 = []
            #         loss1 = torch.tensor(0.0, dtype=torch.float32).to(device)
            #         for i in range(scales_num):
            #             loss_item1 = self.loss_func(adv_imgs_embeds1[i * b:i * b + b], txt_embeds, txt2img)
            #             loss_list1.append(loss_item1.item())
            #             loss1 += loss_item1
            #     loss1.backward()
            #     GV_grad += neighbor_images.grad
            # v = GV_grad / 5 - adv_imgs_grad
            # adv_images = adv_imgs.detach() + alpha*grad.sign()
            # adv_imgs = torch.min(torch.max(adv_images, imgs - self.eps), imgs + self.eps)
            # adv_imgs = torch.clamp(adv_images, 0.0, 1.0)

            adv_imgs_grad = adv_imgs.grad
            grad=(adv_imgs_grad+v)/torch.mean(torch.abs(adv_imgs_grad+v),dim=(1,2,3),keepdim=True)
            grad=grad+momentum*2   # momentum = 1 * momentum + adv_imgs_grad
            momentum=grad          #perturbation = self.step_size * momentum.sign()
            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(adv_imgs).detach().to(device)
            GV_grad2 = torch.zeros_like(adv_imgs).detach().to(device)
            for _ in range(5):
                neighbor_images = adv_imgs.detach() + torch.randn_like(
                    adv_imgs
                ).uniform_(-self.eps * beta, self.eps * beta)

                neighbor_images.requires_grad = True
                scaled_imgs = self.get_scaled_imgs(neighbor_images, scales, device)

                if self.normalization is not None:
                    adv_imgs_output1 = model.inference_image(self.normalization(scaled_imgs))
                else:
                    adv_imgs_output1 = model.inference_image(scaled_imgs)
                adv_imgs_embeds1 = adv_imgs_output1['image_feat']

                with torch.enable_grad():
                    loss_list1 = []
                    loss1 = torch.tensor(0.0, dtype=torch.float32).to(device)
                    for i in range(scales_num):
                        loss_item1 = self.loss_func(adv_imgs_embeds1[i * b:i * b + b], txt_embeds, txt2img)
                        loss_list1.append(loss_item1.item())
                        loss1 += loss_item1
                loss1.backward()
                GV_grad += neighbor_images.grad
            v1 = GV_grad / 2 - adv_imgs_grad
            for _ in range(5):
                # neighbor_images2 = adv_imgs.detach() + torch.randn_like(
                #     adv_imgs
                # ).uniform_(-self.eps * beta, self.eps * beta)
                #
                x = imgs.clone()
                sigma = 16
                gauss = torch.randn(x.size()[0], 3, x.shape[-2], x.shape[-1]) * (sigma / 255)
                gauss = gauss.cuda()
                x_dct = dct_2d(x + gauss).cuda()
                mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
                x_idct = idct_2d(x_dct * mask)
                x_idct = V(x_idct, requires_grad=True)
                # to_pil = ToPILImage()
                # for i, img in enumerate((x_idct)):
                #     # Convert the tensor image to PIL image
                #     pil_img = to_pil(img)
                #     # Save the PIL image
                #     pil_img.save(f'output_image_{i}.png')


                neighbor_images2=x_idct

                neighbor_images2.requires_grad = True
                scaled_imgs2 = self.get_scaled_imgs(neighbor_images2, scales, device)

                if self.normalization is not None:
                    adv_imgs_output2 = model.inference_image(self.normalization(scaled_imgs2))
                else:
                    adv_imgs_output2 = model.inference_image(scaled_imgs2)
                adv_imgs_embeds2 = adv_imgs_output2['image_feat']

                with torch.enable_grad():
                    loss_list2 = []
                    loss2 = torch.tensor(0.0, dtype=torch.float32).to(device)
                    for i in range(scales_num):
                        loss_item2 = self.loss_func(adv_imgs_embeds2[i * b:i * b + b], txt_embeds, txt2img)
                        loss_list2.append(loss_item2.item())
                        loss2 += loss_item2
                loss2.backward()
                GV_grad2 += neighbor_images2.grad
            v2 = GV_grad2 / 2 - adv_imgs_grad

            v=0.5*v1+0.5*v2
            adv_images = adv_imgs.detach() + alpha*grad.sign()
            adv_imgs = torch.min(torch.max(adv_images, imgs - self.eps), imgs + self.eps)
            adv_imgs = torch.clamp(adv_images, 0.0, 1.0)

        return adv_imgs

    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])

        reverse_transform = transforms.Resize(ori_shape,
                                              interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]),
                           int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                                interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)

            reversed_imgs = reverse_transform(scaled_imgs)

            result.append(reversed_imgs)

        return torch.cat([imgs, ] + result, 0)



filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)
    

class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.text_proj_m = nn.Linear(768, 256)
        self.vision_proj_m = nn.Linear(768, 256)


    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim)  # (N * n_locals) * d
        m_n = m.reshape(-1, dim)  # N * d

        # print(temp)#0.97
        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
        u_p = torch.matmul(l, m.permute(0, 2, 1)).unsqueeze(2) / temp  # N * n_locals * 1 * 1

        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1 - temp_mask))

        u_n = torch.mm(m_n, l_n.t()) / temp
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1)  # N x N x n_locals x 1

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device)  # N*N*1*1
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1 - temp_mask))

        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return -loss / 50
    def img_guided_attack(self, net, texts, img_embeds=None,img_embdding=None):
        device = self.ref_net.device
        self.vision_proj_m=self.vision_proj_m.to(device)

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)



                loss = self.loss_func(replace_embeds, img_embeds, i)

                # # print(loss,'lossti')
                # img_feat_m_l = F.normalize(self.vision_proj_m(img_embdding[:, 0:, :]), dim=-1)

                # img_feat_m_l_repeated = img_feat_m_l.repeat(5, 1, 1)
                # odd_tensors = img_feat_m_l_repeated[::2]
                # even_tensors = img_feat_m_l_repeated[1::2]
                # img_feat_m_l = torch.cat([odd_tensors, even_tensors], dim=0)

                """
                torch.Size([2, 577, 256])
                torch.Size([10, 256])
                """
                # print(img_feat_m_l.size())
                # print(origin_embeds.size())
                # loss_t2t_inMod_l = self.in_batch_g2l_loss(img_feat_m_l, origin_embeds, self.temp)  ##图像局部、文本全局
                # loss += loss_t2t_inMod_l
                # print(loss_t2t_inMod_l,'loss_t2t_inMod_l')

                """
                orch.Size([10, 30, 256])
                torch.Size([10, 256])
                """
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss

    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1),
                                 origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))

                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
