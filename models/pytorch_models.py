import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, TransformerConv
from torch.nn import Dropout, Linear
from sklearn.metrics import balanced_accuracy_score
from timm.models.layers import DropPath, trunc_normal_

from .build import MODELS
from utils import misc
from utils.logger import *
from knn_cuda import KNN
from chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
# trade-off precision for performance set "high" | "medium" 
torch.set_float32_matmul_precision('medium')


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
        Group points using FPS and KNN.

        :param xyz: Batch of point clouds with or without labels.
        :return: Tuple of neighborhood, center, and optionally expanded labels.
        '''
        # check if input contains labels (this is the case during finetuning)
        labels = None
        if isinstance(xyz, list):
            # Assuming the first element is data and the second is labels
            xyz, labels = xyz[0], xyz[1]

        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        # Handle labels if provided
        if labels is not None:
            # Expand the labels, we should have a label for each point in neighborhood
            labels_expanded = labels.view(batch_size * num_points, -1)[idx, :]
            labels_expanded = labels_expanded.view(batch_size, self.num_group, self.group_size, -1).contiguous()

            return neighborhood, center, labels_expanded

        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        # define the transformer argparse
        self.mask_ratio = getattr(cfg.transformer_config, 'mask_ratio', 0) # if nothing provided, mask nothing (this is the case during finetuning)
        self.trans_dim = cfg.transformer_config.trans_dim
        self.depth = cfg.transformer_config.depth 
        self.drop_path_rate = cfg.transformer_config.drop_path_rate
        self.num_heads = cfg.transformer_config.num_heads 
        print_log(f'[args] {cfg.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  cfg.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = getattr(cfg.transformer_config, 'mask_type', None) # if nothing provided, mask type is None (this is the case during finetuning)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            Mask neighbouring point patches, i.e. block masking. 
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def get_latent_representation(self, neighborhood, center):
        group_input_tokens = self.encoder(neighborhood)  # B G C
        batch_size, seq_len, C = group_input_tokens.size()
        pos = self.pos_embed(center)  # add position embedding
        x = self.blocks(group_input_tokens, pos)
        x = self.norm(x)
        return x  # This is the latent representation

    def forward(self, neighborhood, center, noaug = False, gae=False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        if self.mask_type == 'block':
            print_log("BLOCK MASKING!")
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)
        else:
            ValueError(f"Masking type: {self.mask_type} has not been implemented")

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        
        # return the pos embedding to GAE
        if gae:
            return x_vis, bool_masked_pos, pos
        else:
            return x_vis, bool_masked_pos


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.cfg = cfg
        self.trans_dim = cfg.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(cfg)
        self.group_size = cfg.group_size
        self.num_group = cfg.num_group
        self.drop_path_rate = cfg.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = cfg.transformer_config.decoder_depth
        self.decoder_num_heads = cfg.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = cfg.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2()
        else:
            raise NotImplementedError

    def add_unique_colors_to_groups(self, full_rebuild: torch.Tensor) -> torch.Tensor:
        """
        Add unique colors to each group in full_rebuild.

        :param full_rebuild: A tensor of shape [num_groups, group_size, 3], representing point clouds.
        :return: Tensor with shape [num_groups, group_size, 6], where last 3 channels are RGB colors.
        """
        num_groups, group_size, _ = full_rebuild.shape

        # Generate random and distinct colors
        rgb_colors = np.random.rand(num_groups, 3) * 255
        np.random.shuffle(rgb_colors)

        # Ensure the RGB values are distinct enough to avoid shades of gray
        rgb_colors = np.where(rgb_colors < 128, rgb_colors / 2, rgb_colors + (255 - rgb_colors) / 2)

        # Convert the color list to a single numpy array before converting to a PyTorch tensor
        colors_tensor = torch.tensor(rgb_colors, dtype=torch.float32, device=full_rebuild.device).unsqueeze(1).repeat(1, group_size, 1)

        # Concatenate the colors with the full_rebuild tensor
        full_rebuild_with_colors = torch.cat([full_rebuild, colors_tensor], dim=-1)
        return full_rebuild_with_colors

    def forward(self, pts, vis=False, **kwargs):
        neighborhood, center = self.group_divider(pts)

        # Extract only the first sample for visualization
        if vis:
            neighborhood = neighborhood[0:1]
            center = center[0:1]
            points = pts[0:1].reshape(-1, 3)
            
        # Encode the point cloud neighborhood and get a mask indicating which points are masked
        x_vis, mask = self.MAE_encoder(neighborhood, center)

        # Extract the batch size (B), and the feature dimension (C) of the visible points
        B, _, C = x_vis.shape  # B: Batch size, C: Channel/Feature dimension

        # Generate positional embeddings for visible (unmasked) points and reshape to match batch size and feature dimension
        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)

        # Generate positional embeddings for masked points and reshape similarly
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        # Get the number of masked points (N)
        _, N, _ = pos_emd_mask.shape

        # Expand the mask token to match the batch size and number of masked points
        mask_token = self.mask_token.expand(B, N, -1)

        # Concatenate the encoded visible points and the mask tokens to form the full set of tokens
        x_full = torch.cat([x_vis, mask_token], dim=1)

        # Concatenate the positional embeddings for visible and masked points
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        # Decode the full set of tokens (visible and masked) to reconstruct the point cloud
        x_rec = self.MAE_decoder(x_full, pos_full, N)

        # Reshape the decoded output to match the batch size and number of points
        B, M, C = x_rec.shape  # B: Batch size, M: Number of points per batch, C: Channel/Feature dimension

        # Process the reconstructed points through an additional layer to predict the XYZ coordinates
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # Reshape to B*M points, each with 3 coordinates
        
        # Extract the ground truth points for the masked regions
        gt_points = neighborhood[mask].reshape(B * M, -1, 3)

        # Calculate the reconstruction loss between the predicted and ground truth points
        loss1 = self.loss_func(rebuild_points, gt_points)


        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)


            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0).reshape(-1, 3)
            #-----------------
            # Generate a color map with a number of unique colors equal to the number of groups
            colored_full_rebuild = self.add_unique_colors_to_groups(full_rebuild)
            colored_full_rebuild = colored_full_rebuild.reshape(-1, 6)
            #-----------------
            full_rebuild = full_rebuild.reshape(-1, 3)
            ret2 = full_vis.reshape(-1, 3)
            ret1 = full.reshape(-1, 3)
            # Extracting masked original points
            masked_vis_points = neighborhood[mask].reshape(B * M, -1, 3)
            full_masked_vis = masked_vis_points + center[mask].unsqueeze(1)
            full_masked_vis = full_masked_vis.reshape(-1, 3)

            # extract the latent representation for T-SNE
            latent = self.MAE_encoder.get_latent_representation(neighborhood, center)
            latent = latent.squeeze()

            return ret1, ret2, full_center, colored_full_rebuild, points, full_masked_vis, latent, loss1
        else:
            return loss1
        

@MODELS.register_module()
class GroupAndEncode(nn.Module):
    """ 
    Pytorch Group and encode. Only used with a pretrained PointMAE for GNN dataset creation. 

    Uses the grouping module and the pretrained Point-MAE encoder to return latents for each patch plus the patch labels. 
    This can then be used to create the GNN data. 
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.trans_dim = cfg.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(cfg)
        self.group_size = cfg.group_size
        self.num_group = cfg.num_group

        print_log(f'[GroupAndEncode] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger='GroupAndEncode')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

    def forward(self, pts, labels, **kwargs):
        """
        Forward pass.

        :param pts: Tensor representing the input point clouds.
        :param labels: Tensor containing the labels for each point in the point clouds.
        :return: Encoded patches, majority patch labels, and pairwise (patch) distance.
        """
        # Divide the point cloud into groups
        neighborhood, center, labels_expanded = self.group_divider([pts, labels])


        # Encode the point cloud neighborhood
        x_encoded, _, _ = self.MAE_encoder(neighborhood, center, gae=True)

        # apply majority voting to labels (end up with 1 label per patch)
        labels_expanded_unsqueezed = labels_expanded.squeeze(-1)
        modes, _ = torch.mode(labels_expanded_unsqueezed, dim=2)
        majority_patch_labels = modes.unsqueeze(-1)
        return x_encoded, majority_patch_labels



@MODELS.register_module()
class GNN(torch.nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.noise_percentage = cfg.noise_percentage

        self.ln_mlp = LayerNorm(cfg.gnn_input_channels)
        self.dropout = Dropout(cfg.dropout)

        # Initialize lists for conv, norm, and linear layers
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.lins = nn.ModuleList()

        initial_channels = cfg.gnn_input_channels 
        for i in range(cfg.depth):
            in_channels = initial_channels if i == 0 else cfg.heads * cfg.gnn_hidden_channels
            self.convs.append(TransformerConv(in_channels, cfg.gnn_hidden_channels, heads=cfg.heads))
            self.lns.append(LayerNorm(cfg.heads * cfg.gnn_hidden_channels))
            self.lins.append(Linear(in_channels, cfg.heads * cfg.gnn_hidden_channels))

        # Final convolution to produce outputs for classes
        final_in_channels = cfg.heads * cfg.gnn_hidden_channels
        self.conv_final = TransformerConv(final_in_channels, cfg.classes, heads=cfg.heads_final_conv, concat=False)

        self.ln_final = LayerNorm(cfg.classes)
        self.lin_final = Linear(final_in_channels, cfg.classes)

    def loss_func(self, output, target):
        """
        :param output: The model output.
        :param target: The true labels.
        :return: Loss value.
        """
        loss = nn.CrossEntropyLoss()
        return loss(output, target)
    
    def print_summary(self):
        """
        Print the summary of the model layers.
        """
        print("----------------\nMODEL SUMMARY:\n----------------")
        print("Convolutional Layers:")
        for i, conv in enumerate(self.convs):
            print(f"Conv Layer {i+1}: {conv}")
            print(f"LayerNorm {i+1}: {self.lns[i]}")
            print(f"Linear {i+1}: {self.lins[i]}")
        print("Final Conv Layer:", self.conv_final)
        print("Final LayerNorm:", self.ln_final)
        print("Final Linear:", self.lin_final)
        print("-----------------\nEND SUMMARY:\n-----------------")

    def forward(self, x, edge_index, edge_attr=None, y=None):

         # Add noise to x for experiment 1. 
        if self.noise_percentage > 0:
            # Calculate the number of elements to replace with noise
            num_elements = x.numel()
            num_noise_elements = int(self.noise_percentage / 100.0 * num_elements)
            
            # Generate random indices to replace with noise
            noise_indices = torch.randperm(num_elements)[:num_noise_elements]
            
            # Generate noise
            noise = torch.randn(num_noise_elements, device=x.device)
            
            # Flatten x, apply noise, then reshape back to original shape
            x_flat = x.view(-1)
            x_flat[noise_indices] = noise
            x = x_flat.view_as(x)


        for conv, ln, lin in zip(self.convs, self.lns, self.lins):
            x = F.elu(conv(x, edge_index) + lin(x))
            x = ln(x)
            x = self.dropout(x)

        # Process the final layer
        x = self.conv_final(x, edge_index) + self.lin_final(x)
        x = self.ln_final(x)

        y = y.long()
        mask = y < self.cfg.classes

        loss = self.loss_func(x[mask], y[mask]).mean()
        preds = torch.argmax(x, dim=1)[mask]
        actuals = y[mask]

        accuracy = (preds == actuals).float().mean().item()
        balanced_acc = balanced_accuracy_score(actuals.detach().cpu().numpy(), preds.detach().cpu().numpy())
        return loss, balanced_acc, preds, actuals


@MODELS.register_module()
class MLP(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.mlp = nn.Sequential(
            nn.Linear(384, 512),  
            nn.ReLU(),
            nn.Linear(512, 1028),  
            nn.ReLU(),
            nn.Linear(1028, 512),  
            nn.ReLU(),
            nn.Linear(512, 512),  
            nn.ReLU(),
            nn.Linear(512, cfg.classes)  
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # x = [128, 256, 384] batch_size, feature, feature_dim
        # y = [128, 256] batch_size, labels. 
        batch_size, num_items, feature_dim = x.shape

        # Flatten x to [128*256, 384]
        x = x.view(batch_size * num_items, feature_dim)

        # Pass through the MLP
        output = self.mlp(x)

        # Reshape output to [128, 256, num_classes]
        output = output.view(batch_size, num_items, -1)

        # Calculate loss if labels are provided
        if y is not None:
            # Flatten y to [128*256]
            y = y.view(batch_size * num_items)
            loss = self.loss_fn(output.view(batch_size * num_items, -1), y)
        else:
            loss = None

        # # Convert logits to predictions
        _, preds = torch.max(output, dim=-1)

        # Calculate the balanced accuracy over all predictions and y if y is provided
        if y is not None:
            balanced_acc = balanced_accuracy_score(y.cpu().numpy(), preds.view(-1).cpu().numpy())
        else:
            balanced_acc = None

        return balanced_acc, loss