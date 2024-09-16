import torch

import torch.nn as nn
import torch.nn.functional as F

class ViTEncoder(nn.Module):
	def __init__(self, input_size, patch_size, hidden_dim, num_heads, num_layers):
		super(ViTEncoder, self).__init__()
		self.input_size = input_size
		self.patch_size = patch_size
		self.hidden_dim = hidden_dim
		self.num_heads = num_heads
		self.num_layers = num_layers
		
		self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
		self.positional_embedding = nn.Parameter(torch.randn(1, self.input_size[0]*self.input_size[1] // patch_size**2, hidden_dim))
		
		self.transformer_blocks = nn.ModuleList([
			TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
		])

		self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True)
		
		
	def forward(self, x):
		x = self.patch_embedding(x)
		x = x.flatten(2).transpose(1, 2)
		x = x + self.positional_embedding
		
		for transformer_block in self.transformer_blocks:
			x = transformer_block(x)
		
		x = x.view(-1, self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size, self.hidden_dim)
		x = x.mean(dim=1)
		
		x, hidden = self.lstm(x)
		return x, hidden

class TransformerBlock(nn.Module):
	def __init__(self, hidden_dim, num_heads):
		super(TransformerBlock, self).__init__()
		self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
		self.layer_norm1 = nn.LayerNorm(hidden_dim)
		self.feed_forward = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim * 4),
			nn.GELU(),
			nn.Linear(hidden_dim * 4, hidden_dim)
		)
		self.layer_norm2 = nn.LayerNorm(hidden_dim)
		
	def forward(self, x):
		attention_output, _ = self.attention(x, x, x)
		x = x + attention_output
		x = self.layer_norm1(x)
		
		feed_forward_output = self.feed_forward(x)
		x = x + feed_forward_output
		x = self.layer_norm2(x)
		
		return x