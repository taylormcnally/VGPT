
from msilib import sequence
from typing import Any, Callable, List, Optional, Text, Tuple, Union
import math
from typing import Tuple

import apex
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import pytorch_lightning as pl

from megatron import get_args
from megatron import mpu
from megatron.model import MegatronModule 
from megatron.model.transformer import ParallelTransformer, ParallelTransformerLayer
from megatron.model.transformer import DropPath
from megatron.model import LayerNorm


class Reduction(nn.Module):
	'''
	Reduce high dimensional voxel size for transformer encoder attention mapping. EdGeS.

	inputs: (batch_size, sequence/S, height, width, channels)
	outputs: (batch_size, sequence/S, height/X, width/X, channels/C)


	Step 1: Learn token embedding.
												tubulets with positional  
														encoding         
											   ┌─────┐  ┌─────┐ ┌─────┐  
	┌───────────────────┐                      │     │█ │     │█│     │█ 
	│                   │█                     │ 1,1 │█ │ 1,2 │█│  3  │█ 
	│Temporal RGB Tensor│█                     └─────┘█ └─────┘█└─────┘█ 
	(batch_size,128,512,512,3)────────────────▶┌─────┐█ ┌─────┐█┌─────┐█ 
	│                   │█                     │     │█ │     │█│     │█ 
	│                   │█                     │  4  │█ │  5  │█│ 6.. │█ 
	└───────────────────┘█                     └─────┘█ └─────┘█└─────┘█ 
	█████████████████████                      

	'''
	def __init__(self, 
			tensor_shape: Tuple[int, int, int, int, int],
			embed_dim=768,
			conv_kernel_size: int = 3,
			conv_stride: int = 1,
			conv_padding: int = 1): 
		super().__init__()
		self.layer_norm = nn.LayerNorm()
		self.conv = nn.Conv3d(tensor_shape[-1], embed_dim, in_channels=3, out_channels=3, kernel_size=4, stride=1) #
		self.sigmoid  = nn.Sigmoid()
		self.gelu = nn.GELU()
		self.batch_norm = nn.BatchNorm3d(num_features=1)

	def _init_weights(self, m):
		"""
		Initialize the weights of the attention maps.
		"""
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv3d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()


	def forward(self, w):
		#per this paper: use a deep temporal conv and a wide non-temporal conv
		x = self.layer_norm(w) # (batch_size, sequence/S, height, width, channels)

		_, _, height, width, sequence = x.shape
		#apply block_voxels to get tubelets of 3,16,16?
		for i in range(3):
			x = self.conv(x)
			x = self.gelu(x)
		#generate attention maps
		x = self.conv(x)
		x = self.sigmoid(x)
		
		return x, height, width, sequence


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DecoderBlock(nn.Module):
	def __init__(self,
					in_features,
					hidden_features=None,
					out_features=None,
					act_layer=nn.GELU,
					drop=0.):
		super().__init__()
		out_features = out_features or in_features

		self.self_attention = nn.MultiheadAttention(in_features, num_heads=8)
	

	def forward(self, x, height, width, sequence):
		return x


class Synthesis(nn.Module):
	'''
	Upscale Transformer decoder voxel output from tokenized form to resolution of input.
	
	inputs: (batch_size, channels, sequence, height, width)
	outputs: (batch_size, channels*C, sequence, height*X, width*X)

	Step 4: Reconstruct voxel from tokenized form.
	
	tubelets with positional                                               
			encoding                                                      
	┌─────┐  ┌─────┐ ┌─────┐                       ┌───────────────────┐   
	│1,1  │█ │1,2  │█│1,3  │█                      │                   │█  
	│     │█ │     │█│     │█                      │Temporal RGB Tensor│█  
	└─────┘█ └─────┘█└─────┘█   ──────────────▶  (batch_size,128,256,256,3)
	┌─────┐█ ┌─────┐█┌─────┐█                      │                   │█  
	│1,4  │█ │1,5  │█│1,6  │█                      │                   │█  
	│     │█ │     │█│     │█                      └───────────────────┘█  
	└─────┘█ └─────┘█└─────┘█                       █████████████████████  



	'''
	def __init__(self, 			
			s: int,
			x: int,
			rgb: int = 3,
			num_of_tokens: int = 1_000_000,
			): 
		super().__init__()
		#convtranspose 3d to original resolution
		self.upscale = nn.ConvTranspose3d(16, 3, (1, 2, 2), stride=(1, 2, 2))
		# add color channel
		self.rgb = nn.ConvTranspose3d(16, 3, (1, 2, 2), stride=(1, 2, 2))


	def forward(self, w):
		x = self.upscale(w)
		x = self.rgb(x)
		return x

#torch.nn.einsum('bijc,bijd->bcd', inputs, weights)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x



class PositionEmbedding(nn.Module):
	"""Defines learnable positional embeddings."""
	def build(self, input_shape: tf.TensorShape) -> None:
		input_dim = input_shape[-1]
		input_height = input_shape[-3]
		input_width = input_shape[-2]
		self.embedding_weight = self.add_weight(
			"embedding_weight",
			shape=(1, input_height, input_width, input_dim),
			initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
			trainable=True)
		super().build(input_shape)

	def call(self, inputs: nn.Tensor) -> nn.Tensor:
		return inputs + self.embedding_weight


class VGPT(MegatronModule):
	def __init__(self,
				depths: List[int],
				pre_process = True,
				post_process = True
				):
		"""
		Args:
			output_size: An integer for the output size.
			output_dim: An integer for the output channel dimension.
			attn_type: A string for attention type ("multi_head" or "multi_query").
			norm_type: A string for the type of normalization.
			activation: Activation function.
		**kwargs: Additional arguments for `tf.keras.layers.Layer`.
		"""
		super(VGPT, self).__init__(share_word_embeddings=False)
		args = get_args()

		self.depths = depths

		self.hidden_size = args.hidden_size
		self.patch_dim = args.patch_dim
		self.img_h = args.img_h
		self.img_w = args.img_w

		#reduction
		self.reduction_1 = Reduction(input_size

	def train_step(self, images):
		pass

	def validation_step(self, images):
		pass
