import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
#from build_dalle import build_dalle
from clip import clip
from sentence_transformers import util
from .dist import DIST
from .weight import WeightLearningModule
from .pooling import TopKPooling_1

import torch

class FeatureExtractor:
    def __init__(self):
        pass

    def _get_hook(self, storage_dict, layer_idx):
        def hook(module, input, output):
            storage_dict[layer_idx] = output
        return hook

    def get_img_layers(self, model, input_image, target_layers):
        if isinstance(target_layers, int):
            target_layers = [target_layers]
        intermediate_outputs = {}
        hook_handles = []
        resblocks = model.visual.transformer.resblocks
        
        for layer_idx in target_layers:
            if 0 <= layer_idx < len(resblocks):
                hook_fn = self._get_hook(intermediate_outputs, layer_idx)
                handle = resblocks[layer_idx].register_forward_hook(hook_fn)
                hook_handles.append(handle)
            else:
                print(f"Warning: Layer index {layer_idx} is out of bounds.")
        try:
            with torch.no_grad():
                model.encode_image(input_image)
        finally:
            for handle in hook_handles:
                handle.remove()

        return intermediate_outputs

    def get_text_layers(self, model, input_text, target_layers):
        if isinstance(target_layers, int):
            target_layers = [target_layers]

        intermediate_outputs = {}
        hook_handles = []
        resblocks = model.transformer.resblocks

        for layer_idx in target_layers:
            if 0 <= layer_idx < len(resblocks):
                hook_fn = self._get_hook(intermediate_outputs, layer_idx)
                handle = resblocks[layer_idx].register_forward_hook(hook_fn)
                hook_handles.append(handle)
            else:
                print(f"Warning: Layer index {layer_idx} is out of bounds.")

        try:
            with torch.no_grad():
                model.encode_text(input_text)
        finally:
            for handle in hook_handles:
                handle.remove()

        return intermediate_outputs

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


class F3SD(nn.Module):
    def __init__(self, args, config: dict):
        super().__init__()
        self.args = args
        self.config = config
        self.loss_config = config['loss_config']
        self.device = torch.device(args.gpu)
        
        self.extractor = FeatureExtractor()

        # set model backbone
        self.clip_model, self.preprocess = clip.load(config['clip_model'], device=self.device, jit=False)
        self.embed_dim = self.clip_model.embed_dim
        self.beta = nn.Parameter(torch.tensor(0.5,device=self.device))
        self.gamma = nn.Parameter(torch.tensor(0.1,device=self.device))
        self.temp = nn.Parameter(torch.tensor(0.07,device=self.device))
        self.dist = DIST(beta = self.beta, gamma = self.gamma, temp = self.temp)
        self.text_poolm = TopKPooling_1(self.args.top_pooling, dim=1)
        self.image_poolm = TopKPooling_1(self.args.top_pooling, dim=1)

        self.img_weight = WeightLearningModule(self.embed_dim * 3)
        self.txt_weight = WeightLearningModule(self.embed_dim * 3)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.unfreeze_transformer_layers(self.clip_model, self.config['distill_layer'])
     
        self.print_model_param_nums(self.clip_model)

        # projection layer for image, one is for cross-modal retrieval, the other is for uni-modal retrieval
        # for corss-modal retrieval
        if self.is_mode_on("contrastive"):
            self.ln_cross_image_projection = nn.LayerNorm(self.embed_dim)
            self.ln_cross_text_projection = nn.LayerNorm(self.embed_dim)
            self.cross_image_projection = nn.Linear(self.embed_dim, self.embed_dim)
            self.cross_text_projection = nn.Linear(self.embed_dim, self.embed_dim)

        # set tau
        if self.is_mode_on("contrastive"):
            self.__init_tau = self.loss_config['contrastive']['tau']
            self.tau = nn.Parameter(torch.tensor(self.__init_tau, device=self.device))

        self.initialize_parameters()
    def unfreeze_transformer_layers(self, model, i):
        num_resblocks = len(model.visual.transformer.resblocks)
        last_layer_idx = num_resblocks - 1

        for layer_idx in [i,last_layer_idx]:
            for param in model.visual.transformer.resblocks[layer_idx].parameters():
                param.requires_grad = True

        num_text_resblocks = len(model.transformer.resblocks)
        text_last_layer_idx = num_text_resblocks - 1
        for layer_idx in [i, text_last_layer_idx]:
            for param in model.transformer.resblocks[layer_idx].parameters():
                param.requires_grad = True

    def print_model_param_nums(self, model=None):
    
	    total = sum([param.nelement() if param.requires_grad else 0 for param in model.parameters()])
	    print('  + Number of params: %.2fM' % (total / 1e6))
    def is_all_gather(self):
        """check if all_gather"""
        return "is_all_gather" in self.config and self.config['is_all_gather']

    def is_mode_on(self, modeName: str) -> bool:
        return self.loss_config[modeName]['is_on']

    def is_add_cross_soft_mode(self):
        """check if add softlabel"""
        return self.is_mode_on("cross_softlabel") and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "add"

    def is_dot_cross_soft_mode(self):
        """check if dot softlabel"""
        return self.is_mode_on("cross_softlabel") and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "dot"

    def is_each_cross_soft_mode(self):
        """check if each softlabel"""
        return self.is_mode_on("cross_softlabel") and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "each"

    def is_mean_contrastive_loss_mode(self, lossName):
        return self.loss_config[lossName]['contrastive_loss_mode'] == "mean"

    def is_sum_contrastive_loss_mode(self, lossName):
        return self.loss_config[lossName]['contrastive_loss_mode'] == "sum"

    def encode_image(self, image, cross_modal=True):
        """Returns the image embedding "z" of shape [batch_size, projection_dim]."""
        _, img_feats_3 = self.clip_model.encode_image(image)
        image_features = img_feats_3[:,0,:]
        img_feats = img_feats_3[:,1:,:].float()
        image_n_top_k = self.encode_n_topk(image_features, img_feats)
        image_pool_topk = self.image_poolm(img_feats)
        img_alpha, img_beta = self.img_weight(image_features, image_n_top_k, image_pool_topk)
        image_features = image_features + img_alpha * image_n_top_k + img_beta * image_pool_topk
        return self._encode_image_features(image_features, cross_modal=cross_modal)

    def encode_text(self, text, cross_modal=True):
        """Returns the text embedding "z" of shape [batch_size, projection_dim]."""
        _, txt_feats_3 = self.clip_model.encode_text(text)
        text_features = txt_feats_3[torch.arange(txt_feats_3.shape[0]),text.argmax(dim=-1)]
        text_feats = txt_feats_3.float()
        text_n_top_k = self.encode_n_topk(text_features, text_feats)
        text_pool_topk = self.text_poolm(text_feats)
        text_alpha, text_beta = self.txt_weight(text_features,text_n_top_k,text_pool_topk)
        text_features = text_features + text_alpha * text_n_top_k + text_beta * text_pool_topk
        return self._encode_text_features(text_features, cross_modal=cross_modal)

    def _encode_image_features(self, image_features, cross_modal=True):
        """encode from clip model"""
        img_feats = image_features
        if cross_modal and (self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel")):
            image_features = self.ln_cross_image_projection(image_features)
            image_features = self.cross_image_projection(image_features)
        return image_features + img_feats

    def _encode_text_features(self, text_features, cross_modal=True):
        """encode from clip model"""
        txt_feats = text_features
        if cross_modal and (self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel")):
            text_features = self.ln_cross_text_projection(text_features)
            text_features = self.cross_text_projection(text_features)

        return text_features + txt_feats
    def encode_n_topk(self,i_feats,image_feats,image=None):
        bs_I = image_feats.shape[0]
        if image is not None:
            k = self.args.encode_n_top
        else:
            k = self.args.encode_n_top
        n_top_ks = []
        for batch_index in range(bs_I):
            i_feat_example = i_feats[batch_index]
            i_emb_example = image_feats[batch_index]
            i_similarities = F.cosine_similarity(i_feat_example.unsqueeze(0), i_emb_example, dim=1)
            i_top_K_indices = torch.topk(i_similarities,k=k,largest=False).indices
            i_n_top_K_tokens = i_emb_example[i_top_K_indices]
            n_top_k = torch.mean(i_n_top_K_tokens, dim=0)
            n_top_ks.append(n_top_k.float())
        n_top_ks = torch.stack(n_top_ks,dim=0)
        return n_top_ks
    def get_similarity(self, image_features, text_features, cross_modal=True):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if cross_modal:
            """if cross-modal retrieval, return the similarity between image and text"""
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            return logits_per_image, logits_per_text
        else:
            """if uni-modal retrieval, return the similarity between image and image, text and text"""
            logits_image_image = image_features @ image_features.t()
            logits_text_text = text_features @ text_features.t()
            return logits_image_image, logits_text_text

    def initialize_parameters(self):
        """Initialize the model parameters."""
        if self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel"):
            nn.init.normal_(self.cross_image_projection.weight, std=0.02)
            nn.init.normal_(self.cross_text_projection.weight, std=0.02)

        if self.is_mode_on("contrastive"):
            if self.loss_config['contrastive']['is_block_tau']:
                self.tau.requires_grad_(False)


    def load_state_dict(self, state_dict, strict=True):
        """load state dict"""
        if state_dict is None:
            return "state_dict is None"
        msg = super().load_state_dict(state_dict, strict)
        return msg

    def ContrastiveLoss(self, logits_per_image, logits_per_text, idx=None):
        # contrastive loss
        if idx is None:
            sim_targets = torch.eye(logits_per_image.shape[0], device=self.device)
        else:
            idx = idx.view(-1, 1)
            pos_idx = torch.eq(idx, idx.t()).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        if self.is_mean_contrastive_loss_mode("contrastive"):
            loss_i2t = -torch.mean(F.log_softmax(logits_per_image / self.tau, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.mean(F.log_softmax(logits_per_text / self.tau, dim=1) * sim_targets, dim=1).mean()
        elif self.is_sum_contrastive_loss_mode("contrastive"):
            loss_i2t = -torch.sum(F.log_softmax(logits_per_image / self.tau, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits_per_text / self.tau, dim=1) * sim_targets, dim=1).mean()
        else:
            raise ValueError("contrastive loss mode error")
        contrastive_loss = loss_i2t + loss_t2i

        return contrastive_loss

    def KLContrastiveSimLoss(self, logits, softlabel, tau, softlabel_tau, lossName, use_loss="kl"):
        """
        KL divergence loss
        make logits and softlabel have the same distribution
        logits to softlabel
        """
        # softmax for softlabel
        sim_targets = F.softmax(softlabel / softlabel_tau, dim=1)

        # log softmax
        logit_inputs = F.log_softmax(logits / tau, dim=1)

        if use_loss == "kl":
            # KL divergence
            loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')
        elif use_loss == "contrastive":
            # Switch to the same loss as ContrastiveLoss, but sim_targets is soft
            if self.is_mean_contrastive_loss_mode(lossName):
                loss = -torch.mean(logit_inputs * sim_targets, dim=1).mean()
            elif self.is_sum_contrastive_loss_mode(lossName):
                loss = -torch.sum(logit_inputs * sim_targets, dim=1).mean()
            else:
                raise ValueError("contrastive loss mode error")
        else:
            raise ValueError("loss mode error")

        return loss


    @torch.no_grad()
    def clamp_tau(self):
        # clip tau to prevent overflow
        if self.is_mode_on("contrastive"):
            self.tau.clamp_(min=self.loss_config['contrastive']['tau_min'], max=self.loss_config['contrastive']['tau_max'])
            self.beta.clamp_(min=0.1,max=5)
            self.gamma.clamp_(min=0.1,max=5)
            self.temp.clamp_(min=0.01,max=0.1)

    def _process_last_img_layer(self, x):
        x = x.permute(1, 0, 2) 
        img_feats_3 = x
        x_global = self.clip_model.visual.ln_post(x[:, 0, :])
        img_feats_3 = self.clip_model.visual.ln_post(img_feats_3)
        if self.clip_model.visual.proj is not None:
            x_global = x_global @ self.clip_model.visual.proj
            img_feats_3 = img_feats_3 @ self.clip_model.visual.proj
        return x_global, img_feats_3

    def _process_last_text_layer(self, x, text_input):
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        txt_feats_3 = x
        x_global = x[torch.arange(x.shape[0]), text_input.argmax(dim=-1)] 
        x_global = x_global @ self.clip_model.text_projection
        txt_feats_3 = txt_feats_3 @ self.clip_model.text_projection

        return x_global, txt_feats_3

    def forward(self, image, text, epoch=None, idx=None):
        rankNum = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        worldSize = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        
        self.clamp_tau()

        last_layer_idx = len(self.clip_model.visual.transformer.resblocks) - 1
        distill_layer_idx = self.config['distill_layer']
        
        target_layers = list(set([distill_layer_idx, last_layer_idx])) 

    
        image_layer_outputs = self.get_img_layers(self.clip_model, image, target_layers)
        
        i_img_feats = image_layer_outputs[distill_layer_idx].float()

        last_img_layer_raw = image_layer_outputs[last_layer_idx]


        image_features, img_feats_3 = self._process_last_img_layer(last_img_layer_raw)
        
        last_img_feats = last_img_layer_raw.float()

        text_last_layer_idx = len(self.clip_model.transformer.resblocks) - 1
        text_target_layers = list(set([distill_layer_idx, text_last_layer_idx]))

        text_layer_outputs = self.get_text_layers(self.clip_model, text, text_target_layers)

        i_text_feats = text_layer_outputs[distill_layer_idx].float()
        last_text_layer_raw = text_layer_outputs[text_last_layer_idx]


        text_features, txt_feats_3 = self._process_last_text_layer(last_text_layer_raw, text)
        
        last_text_feats = last_text_layer_raw.float()

        text_feats = txt_feats_3
        image_feats = img_feats_3[:, 1:, :] 

        image_n_top_k = self.encode_n_topk(image_features, image_feats)
        text_n_top_k = self.encode_n_topk(text_features, text_feats)

        image_pool_topk = self.image_poolm(image_feats)
        text_pool_topk = self.text_poolm(text_feats)

        img_alpha, img_beta = self.img_weight(image_features, image_n_top_k, image_pool_topk)
        txt_alpha, txt_beta = self.txt_weight(text_features, text_n_top_k, text_pool_topk)

        image_features = image_features + img_alpha * image_n_top_k + img_beta * image_pool_topk
        text_features = text_features + txt_alpha * text_n_top_k + txt_beta * text_pool_topk

        if self.is_all_gather() and idx is not None:
            idx_all = allgather(idx, rankNum, worldSize)
        else:
            idx_all = idx

        if self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel"):
            cross_image_features, cross_text_features = self._encode_image_features(
                image_features, cross_modal=True), self._encode_text_features(text_features, cross_modal=True)
            if self.is_all_gather():
                cross_image_features, cross_text_features = allgather(
                    cross_image_features, rankNum, worldSize), allgather(cross_text_features, rankNum, worldSize)
            logits_per_image, logits_per_text = self.get_similarity(cross_image_features, cross_text_features, cross_modal=True)

        cross_modal_loss = torch.tensor(0.0, device=self.device)
        contrastive_loss = torch.tensor(0.0, device=self.device)

        if self.is_mode_on("contrastive"):
            # Contrastive Loss
            contrastive_loss = self.ContrastiveLoss(logits_per_image, logits_per_text, idx_all)
            contrastive_loss /= 2.0
            contrastive_loss = contrastive_loss * self.loss_config['contrastive']['rate']
            
            cross_modal_loss = (self.dist(i_img_feats, last_img_feats) + self.dist(i_text_feats, last_text_feats)) / 2.0

        return cross_modal_loss, contrastive_loss
