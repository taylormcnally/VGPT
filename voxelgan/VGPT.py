
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
from megatron.model.transformer import ParallelTransformer



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
			tokens = 768,
			conv_kernel_size: int = 3,
			conv_stride: int = 1,
			conv_padding: int = 1): 
		super().__init__()
		self.layer_norm = nn.LayerNorm()
		self.conv = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=4, stride=1)
		self.attention = nn.Conv3d(in_features=tokens, out_features=tokens)
		self.sigmoid  = nn.Sigmoid()
		self.gelu = nn.GELU()
		self.batch_norm = nn.BatchNorm3d(num_features=1)

	def forward(self, w):
		x = self.layer_norm(w) # (batch_size, sequence/S, height, width, channels)

		#apply block_voxels to get tubelets of 3,16,16
		for i in range(3):
			x = self.conv(x)
			x = self.gelu(x)
		#generate attention maps
		x = self.conv(x)
		x = self.sigmoid(x)
		
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


class Mlp(nn.Module):
	def __init__(self,
					in_features,
					hidden_features=None,
					out_features=None,
					act_layer=nn.GELU,
					drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.dwconv = DWConv(hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

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
			#TODO: implement 3d conv config
			pass

	def forward(self, x, S, H, W):
		x = self.fc1(x)
		x = self.dwconv(x, H, W)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x


class Encoder(nn.Module):
	def __init__(self,
					in_features,
					hidden_features=None,
					out_features=None,
					act_layer=nn.GELU,
					drop=0.):
		super().__init__()




class Decoder(nn.Module):
	def __init__(self,
					in_features,
					hidden_features=None,
					out_features=None,
					act_layer=nn.GELU,
					drop=0.):
		super().__init__()

		


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

		self.hidden_size = args.hidden_size
		self.patch_dim = args.patch_dim
		self.img_h = args.img_h
		self.img_w = args.img_w

        # Transformer
        self.transformer = ParallelTransformer(
            self.init_method,
            self.scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=self.post_layer_norm,
            drop_path_rate=self.drop_path_rate
		)

	def train_step(self, images):
		pass

	def validation_step(self, images):
		pass
