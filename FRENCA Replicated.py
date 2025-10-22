# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 14:17:13 2025

@author: Prajeet
"""

#!pip install lpips



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from typing import Tuple, Optional
import warnings
import gc
import os
from pathlib import Path
import json
#import lpips

warnings.filterwarnings('ignore')
extract_path="C:\Main 2\Study\College\Year 3\SOP\MultiUserSC\DIV2K"

# Memory management
def clear_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# Set device with memory check
def get_device():
    if torch.cuda.is_available():
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU found. Memory: {gpu_memory:.1f} GB")
        if gpu_memory < 8:  # Less than 8GB
            print("Warning: Limited GPU memory. Using conservative settings.")
        return torch.device('cuda')
    else:
        print("Using CPU (training will be slower)")
        return torch.device('cpu')

device = get_device()

# Create checkpoint and results directory
DRIVE_PATH = Path("C:\Main 2\Study\College\Year 3\SOP\MultiUserSC\FRENCA_project")

CHECKPOINT_DIR = DRIVE_PATH / "checkpoints"
RESULTS_DIR = DRIVE_PATH / "results"
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

'''
CHECKPOINT_DIR = Path("./checkpoints")
RESULTS_DIR = Path("./results")
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
'''


class PerceptualLoss(nn.Module):
    def __init__(self, alpha=0.85, device='cuda'):
        super().__init__()
        # alpha controls the balance between pixel loss and perceptual loss
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

        # Initialize LPIPS. It will use the VGG network by default.
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)

    def forward(self, reconstructed, original):
        # Pixel-level loss (L1 is often good for this)
        loss_l1 = self.l1_loss(reconstructed, original)

        # Perceptual loss
        # The LPIPS model expects inputs in the range [-1, 1], so we scale from [0, 1]
        loss_lpips = self.lpips_loss(reconstructed * 2 - 1, original * 2 - 1).mean()

        # Combine the two losses
        # We want to minimize both pixel difference and perceptual difference
        total_loss = self.alpha * loss_l1 + (1 - self.alpha) * loss_lpips

        return total_loss



class SafeWindowAttention(nn.Module):
    """Memory-safe Window Attention with gradient checkpointing."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # NOTE: The original relative_position_bias implementation was malformed and would cause an
        # index out-of-bounds error. It has been removed for stability. The model can learn
        # positional information through other means, especially in a compact architecture.
        # self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, 2 * window_size[0] - 1, 2 * window_size[1] - 1))
        # nn.init.trunc_normal_(self.relative_position_bias, std=.02)

    def forward(self, x, mask=None):
        try:
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            # NOTE: Removed broken relative position bias addition.
            # attn = attn + self.relative_position_bias[:, :N, :N].unsqueeze(0)
            if mask is not None:
                # The mask shape is (num_windows, window_size, window_size)
                # We need to broadcast it to the attention score shape (B, num_heads, window_size, window_size)
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            return x
        except RuntimeError as e:
            if "out of memory" in str(e):
                clear_memory()
                raise RuntimeError(f"GPU out of memory in attention. Try reducing batch size. Original error: {e}")
            raise e

class SafeSwinBlock(nn.Module):
    """Memory-safe Swin Transformer Block."""

    def __init__(self, dim, input_resolution, num_heads, window_size=4, shift_size=0, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = min(window_size, min(input_resolution))  # Adaptive window size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Ensure shift_size is not larger than window_size
        if self.shift_size > 0:
            assert 0 < self.shift_size < self.window_size, "shift_size must be in (0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SafeWindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def forward(self, x):
        try:
            H, W = self.input_resolution
            B, L, C = x.shape

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # Simple partitioning without complex shifting for memory efficiency
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape

            # 🚀 CYCLIC SHIFT
            if self.shift_size > 0:
              shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
              attn_mask = self.create_mask(Hp, Wp) # Create the mask for shifted windows
            else:
              shifted_x = x
              attn_mask = None # No mask needed for non-shifted windows

            # Window partition
            x_windows = self.window_partition(shifted_x)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=attn_mask)

            # Reverse window partition
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = self.window_reverse(attn_windows, Hp, Wp)

            # 🚀 REVERSE CYCLIC SHIFT
            if self.shift_size > 0:
              x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
              x = shifted_x

            # Remove padding
            if pad_r > 0 or pad_b > 0:
              x = x[:, :H, :W, :].contiguous()

            x = x.view(B, H * W, C)

            # FFN
            x = shortcut + x
            x = x + self.mlp(self.norm2(x))

            return x

        except RuntimeError as e:
          if "out of memory" in str(e):
              clear_memory()
              raise RuntimeError(f"GPU out of memory in Swin block. Try reducing batch size. Original error: {e}")
          raise e

    def create_mask(self, H, W):
      """Creates a mask to prevent attention between disconnected patches."""
      # This mask identifies different regions in the shifted feature map.
      img_mask = torch.zeros((1, H, W, 1), device=device)
      h_slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
      w_slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))

      # Assign a unique number to each region
      cnt = 0
      for h in h_slices:
          for w in w_slices:
              img_mask[:, h, w, :] = cnt
              cnt += 1

      # Partition the mask into windows
      mask_windows = self.window_partition(img_mask)
      mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

      # Create the attention mask by comparing patch regions
      attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
      # Fill with a large negative value where regions are different, 0 otherwise
      attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
      return attn_mask

    def window_partition(self, x):
        """Partitions the input into non-overlapping windows."""
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        """Reverses the window partitioning."""
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x




class CompactSwinEncoder(nn.Module):
    """Compact Swin Transformer Encoder for memory efficiency."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=48, depths=[1, 2, 1], num_heads=[2, 4, 8]):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(p=0.1)

        # Build layers
        self.layers = nn.ModuleList()
        patches_resolution = [img_size // patch_size, img_size // patch_size]

        for i_layer in range(len(depths)):
            resolution = (patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer))
            layer_dim = int(embed_dim * 2 ** i_layer)

            # Create blocks for this layer
            blocks = nn.ModuleList([
                SafeSwinBlock(
                    dim=layer_dim,
                    input_resolution=resolution,
                    num_heads=num_heads[i_layer],
                    window_size=4,
                    shift_size=0 if (j % 2 == 0) else 2,
                    mlp_ratio=2.0
                ) for j in range(depths[i_layer])
            ])

            self.layers.append(nn.ModuleDict({
                'blocks': blocks,
                'downsample': nn.Conv2d(layer_dim, layer_dim * 2, kernel_size=2, stride=2) if i_layer < len(depths) - 1 else None
            }))

        self.final_dim = int(embed_dim * 2 ** (len(depths) - 1))
        self.final_res = patches_resolution[0] // (2 ** (len(depths) - 1))

    def forward(self, x):
        try:
            # Patch embedding
            x = self.patch_embed(x)  # B, C, H, W
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # B, H*W, C
            x = self.pos_drop(x)

            current_res = [H, W]

            for layer in self.layers:
                # Forward through blocks
                for block in layer['blocks']:
                    block.input_resolution = current_res
                    x = block(x)

                # Downsample if not last layer
                if layer['downsample'] is not None:
                    # Reshape to spatial format
                    x = x.transpose(1, 2).view(B, -1, current_res[0], current_res[1])
                    x = layer['downsample'](x)
                    current_res = [current_res[0] // 2, current_res[1] // 2]
                    x = x.flatten(2).transpose(1, 2)

            return x
        except RuntimeError as e:
            if "out of memory" in str(e):
                clear_memory()
                raise RuntimeError(f"GPU out of memory in encoder. Try reducing batch size or image size. Original error: {e}")
            raise e

class CompactSwinDecoder(nn.Module):
    """Compact Swin Transformer Decoder."""

    def __init__(self, embed_dim=192, depths=[1, 2, 1], num_heads=[8, 4, 2], img_size=32, patch_size=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.final_res = img_size // patch_size // (2 ** (len(depths) - 1))

        # Build upsampling layers
        self.layers = nn.ModuleList()

        for i_layer in range(len(depths)):
            current_dim = embed_dim // (2 ** i_layer)
            resolution = self.final_res * (2 ** i_layer)

            blocks = nn.ModuleList([
                SafeSwinBlock(
                    dim=current_dim,
                    input_resolution=(resolution, resolution),
                    num_heads=num_heads[i_layer],
                    window_size=4,
                    mlp_ratio=2.0
                ) for _ in range(depths[i_layer])
            ])

            upsample = nn.ConvTranspose2d(current_dim, current_dim // 2, kernel_size=2, stride=2) if i_layer < len(depths) - 1 else None

            self.layers.append(nn.ModuleDict({
                'blocks': blocks,
                'upsample': upsample
            }))

        # Final reconstruction
        final_dim = embed_dim // (2 ** (len(depths) - 1))
        self.final_layer = nn.ConvTranspose2d(final_dim, 3, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        try:
            B = x.shape[0]
            current_res = self.final_res

            # Reshape to spatial format if needed
            if x.dim() == 3:
                x = x.view(B, current_res * current_res, -1)

            for layer in self.layers:
                # Forward through blocks
                for block in layer['blocks']:
                    block.input_resolution = (current_res, current_res)
                    x = block(x)

                # Upsample if not last layer
                if layer['upsample'] is not None:
                    # Reshape to spatial format
                    x = x.transpose(1, 2).view(B, -1, current_res, current_res)
                    x = layer['upsample'](x)
                    current_res *= 2
                    x = x.flatten(2).transpose(1, 2)

            # Final reconstruction
            x = x.transpose(1, 2).view(B, -1, current_res, current_res)
            x = self.final_layer(x)

            return torch.sigmoid(x)  # Ensure output is in [0, 1]
        except RuntimeError as e:
            if "out of memory" in str(e):
                clear_memory()
                raise RuntimeError(f"GPU out of memory in decoder. Try reducing batch size. Original error: {e}")
            raise e

class ChannelCoder(nn.Module):
    """Simplified channel encoder/decoder."""

    def __init__(self, input_dim, output_dim, is_encoder=True):
        super().__init__()

        if is_encoder:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, output_dim)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Linear(input_dim * 2, output_dim)
            )

    def forward(self, x):
        # Flatten if input is not already a vector
        if x.dim() > 2:
            x = x.flatten(1)
        return self.layers(x)

class CompactSemComSystem(nn.Module):
    """Compact Semantic Communication System."""

    def __init__(self, img_size=32, compression_ratio=8):
        super().__init__()

        # --- Define Architectural Hyperparameters ---
        patch_size = 4
        num_stages = 4 # As per the paper
        embed_dim = 48

        num_downsamples = num_stages - 1
        final_dim = int(embed_dim * (2 ** num_downsamples))
        final_res = img_size // patch_size // (2 ** num_downsamples)

        # Encoder
        self.source_encoder = CompactSwinEncoder(img_size=img_size, embed_dim=48, depths=[2, 6, 2, 2], num_heads=[2, 4, 8, 16])

        # Calculate dimensions
        self.source_encoder.final_dim = final_dim
        self.source_encoder.final_res = final_res

        encoder_dim = self.source_encoder.final_dim * (self.source_encoder.final_res ** 2)
        compressed_dim = encoder_dim // compression_ratio

        # Channel coders
        self.channel_encoder = ChannelCoder(encoder_dim, compressed_dim, is_encoder=True)

        # Decoders for different computing capacities
        self.channel_decoder_high = ChannelCoder(compressed_dim, encoder_dim, is_encoder=False)
        self.channel_decoder_low = ChannelCoder(compressed_dim, encoder_dim, is_encoder=False)

        # Source decoders
        self.source_decoder_high = CompactSwinDecoder(embed_dim=self.source_encoder.final_dim, depths=[2, 6, 2, 2], num_heads=[16, 8, 4, 2], img_size=img_size, patch_size=patch_size)
        self.source_decoder_low = CompactSwinDecoder(embed_dim=self.source_encoder.final_dim, depths=[2, 2, 2, 2], num_heads=[16, 8, 4, 2], img_size=img_size, patch_size=patch_size)  # Simpler decoder

        self.encoder_dim = encoder_dim
        self.final_res = self.source_encoder.final_res

    def encode_and_add_noise(self, x, snr, add_noise=True):
        """Encodes image and adds channel noise."""
        try:
            semantic_features = self.source_encoder(x)
            encoded_signal = self.channel_encoder(semantic_features)

            # --- 🚀 FIX FOR BUG 1: NORMALIZE SIGNAL POWER ---
            # We must normalize the batch to have average power of 1
            # as assumed in the SNR noise calculation.
            avg_batch_power = torch.mean(encoded_signal ** 2, dim=(-1), keepdim=True)
            normalized_signal = encoded_signal / torch.sqrt(avg_batch_power + 1e-8)
            # --------------------------------------------------

            if add_noise:
                snr_reshaped = snr.view(-1, 1).expand_as(normalized_signal)
                # Now this formula is correct because signal power is 1
                noise_power = 10 ** (-snr_reshaped / 10.0)
                noise = torch.sqrt(noise_power) * torch.randn_like(normalized_signal)
                received_signal = normalized_signal + noise
            else:
                received_signal = normalized_signal
                
            # Return everything the decoder needs
            return received_signal, self.final_res, self.source_encoder.final_dim

        except RuntimeError as e:
            clear_memory()
            if "out of memory" in str(e):
                raise RuntimeError(f"OOM in encode_and_add_noise. {e}")
            raise e

    def decode(self, received_signal, final_res, final_dim, user_type='high'):
        """Decodes the received signal."""
        try:
            if user_type == 'high':
                decoded_features = self.channel_decoder_high(received_signal)
                decoded_features = decoded_features.view(-1, final_res * final_res, final_dim)
                reconstructed = self.source_decoder_high(decoded_features)
            else:  # low
                decoded_features = self.channel_decoder_low(received_signal)
                decoded_features = decoded_features.view(-1, final_res * final_res, final_dim)
                reconstructed = self.source_decoder_low(decoded_features)
            
            return reconstructed
            
        except RuntimeError as e:
            clear_memory()
            if "out of memory" in str(e):
                raise RuntimeError(f"OOM in decode. {e}")
            raise e    

    def forward(self, x, snr, user_type='high', add_noise=True):
        try:
            # Source encoding
            semantic_features = self.source_encoder(x)

            # Channel encoding
            encoded_signal = self.channel_encoder(semantic_features)

            # Add channel noise
            if add_noise:
                snr_reshaped = snr.view(-1, 1).expand_as(encoded_signal)
                noise_power = 10 ** (-snr_reshaped / 10.0)

                noise = torch.sqrt(noise_power) * torch.randn_like(encoded_signal)
                received_signal = encoded_signal + noise
            else:
                received_signal = encoded_signal

            # Channel decoding
            if user_type == 'high':
                decoded_features = self.channel_decoder_high(received_signal)
                # Reshape for decoder
                decoded_features = decoded_features.view(-1, self.final_res * self.final_res, self.source_encoder.final_dim)
                reconstructed = self.source_decoder_high(decoded_features)
            else:  # low
                decoded_features = self.channel_decoder_low(received_signal)
                # Reshape for decoder
                decoded_features = decoded_features.view(-1, self.final_res * self.final_res, self.source_encoder.final_dim)
                reconstructed = self.source_decoder_low(decoded_features)

            return reconstructed

        except RuntimeError as e:
            clear_memory()
            if "out of memory" in str(e):
                raise RuntimeError(f"GPU out of memory in forward pass. Reduce batch size. Original error: {e}")
            raise e

class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset that preloads a subset into RAM."""

    def __init__(self, train=True, size=10000, img_size=32):
        self.size = size
        self.img_size = img_size
        if train:
            # --- Augmentation Pipeline for TRAINING Data --- 🚀
            # We add random flips and color changes to make the model more robust.
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip image horizontally
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Randomly change colors
                transforms.Resize((img_size, img_size)), # Resize it back
                transforms.ToTensor(), # Convert to a tensor
            ])
            print("Loading 'train' dataset with data augmentation.")
        else:
            # --- Simple Pipeline for VALIDATION/TEST Data ---
            # No random changes here, just resizing.
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
            print("Loading 'test' dataset without data augmentation.")

        # Load only a subset and cache in memory
        print(f"Loading {'train' if train else 'test'} dataset...")
        full_dataset = CIFAR10(root='./data', train=train, download=True, transform=self.transform)

        # Cache subset in memory
        self.data = []
        num_samples = min(size, len(full_dataset))
        indices = np.random.choice(len(full_dataset), num_samples, replace=False)
        for idx in indices:
            self.data.append(full_dataset[idx][0])  # Only image, not label

        print(f"Loaded {len(self.data)} images into memory.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

from torch.utils.data import Dataset
from PIL import Image
import os

class DIV2KDataset(Dataset):
    """Custom PyTorch Dataset for the DIV2K dataset."""
    def __init__(self, data_dir, split='train', crop_size=128):
        """
        Args:
            data_dir (str): Path to the main DIV2K directory.
            split (str): 'train' or 'valid' to load the respective dataset.
            crop_size (int): The size of the random crop to apply.
        """
        if split == 'train':
            self.image_dir = os.path.join(data_dir, 'DIV2K_train_HR')
            # For training, we use random cropping as data augmentation
            self.transform = transforms.Compose([
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
            ])
        else: # validation
            self.image_dir = os.path.join(data_dir, 'DIV2K_valid_HR')
            # For validation, we use a deterministic center crop
            self.transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ])

        self.image_files = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the image using Pillow
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')

        # Apply transformations (cropping and converting to tensor)
        return self.transform(image)

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save training checkpoint."""
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'loss': loss,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint(filepath, model, optimizer=None):
    """Load training checkpoint."""
    try:
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            if model is not None:
                model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Checkpoint loaded: {filepath}")
            return checkpoint.get('epoch', 0), checkpoint.get('loss', float('inf'))
        else:
            print(f"No checkpoint found at {filepath}")
            return 0, float('inf')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, float('inf')

def safe_train_step(model, images, snr, criterion, user_type='high'):
    """Safe training step with error handling and recursive batch splitting."""
    try:
        reconstructed = model(images, snr, user_type=user_type)
        loss = criterion(reconstructed, images)
        return loss, reconstructed
    except RuntimeError as e:
        if "out of memory" in str(e):
            clear_memory()
            # Try with smaller batch
            if images.shape[0] > 1:
                print(f"OOM detected, reducing batch size from {images.shape[0]} to {images.shape[0]//2}")
                mid = images.shape[0] // 2
                loss1, _ = safe_train_step(model, images[:mid], snr[:mid], criterion, user_type)
                loss2, reconstructed = safe_train_step(model, images[mid:], snr[mid:], criterion, user_type)
                # Ensure losses require gradients before combining
                return (loss1.detach() + loss2) / 2 if not loss1.requires_grad else (loss1 + loss2) / 2, reconstructed
            else:
                raise RuntimeError("Cannot reduce batch size further. Try reducing image size or model complexity.")
        raise e



def train_with_paper_techniques(model, dataloader, epochs=500, lr=5e-4, use_transfer_learning=True, use_kd=True, alpha=0.5):
    """
    Implements the full training procedure from the paper, including the two-phase
    process, partial transfer learning, and knowledge distillation.
    """
    # ====================================================================================
    # PHASE 1: Train Encoder and High-Computing Decoder (HCD) as the "Teacher"
    # ====================================================================================
    print("--- 👨‍🏫 Phase 1: Training the Teacher (Encoder + HCD) ---")
    criterion = nn.MSELoss().to(device)

    # Parameters for the teacher model
    teacher_params = list(model.source_encoder.parameters()) + \
                     list(model.channel_encoder.parameters()) + \
                     list(model.channel_decoder_high.parameters()) + \
                     list(model.source_decoder_high.parameters())
    optimizer_teacher = optim.Adam(teacher_params, lr=lr)

    # <> Define path and load checkpoint for the teacher model
    latest_teacher_checkpoint_path = CHECKPOINT_DIR / "teacher_latest.pt"
    best_teacher_checkpoint_path = CHECKPOINT_DIR / "teacher_best.pt"

    print(f"Loading latest teacher state from: {latest_teacher_checkpoint_path}")
    start_epoch_teacher, _ = load_checkpoint(latest_teacher_checkpoint_path, model, optimizer_teacher)
    print(f"Loading best teacher loss from: {latest_teacher_checkpoint_path}")
    _, best_loss_teacher = load_checkpoint(best_teacher_checkpoint_path, None, None)

    # This loop trains the encoder and HCD together, as described in the paper[cite: 130].
    for epoch in range(start_epoch_teacher, epochs):
        model.train()
        total_loss_teacher = 0
        num_batches_teacher = 0
        for images in dataloader:
            images = images.to(device)
            snr = torch.FloatTensor(np.random.choice([1, 3, 5, 7], size=images.shape[0])).to(device)
            optimizer_teacher.zero_grad()

            reconstructed = model(images, snr, user_type='high')
            loss = criterion(reconstructed, images)

            loss.backward()
            optimizer_teacher.step()

            total_loss_teacher += loss.item()
            num_batches_teacher += 1
        avg_loss = total_loss_teacher / num_batches_teacher
        print(f'Phase 1 - Epoch {epoch+1}/{epochs}, Teacher Avg Loss: {avg_loss:.6f}')

    # <> Save checkpoint after each epoch
        print("   -> Saving latest teacher checkpoint...")
        save_checkpoint(model, optimizer_teacher, epoch + 1, avg_loss, latest_teacher_checkpoint_path)

        if avg_loss < best_loss_teacher:
            best_loss_teacher = avg_loss
            print(f"  -> New best loss for teacher: {best_loss_teacher:.6f}. Saving checkpoint.")
            save_checkpoint(model, optimizer_teacher, epoch + 1, best_loss_teacher, best_teacher_checkpoint_path)

    print("✅ Phase 1 Complete. Teacher model is trained.")
    
    print(f"\n--- Loading best teacher model from {best_teacher_checkpoint_path} before starting Phase 2 ---")
    load_checkpoint(best_teacher_checkpoint_path, model, optimizer=None)

    # ====================================================================================
    # PHASE 2: Train Low-Computing Decoder (LCD) as the "Student"
    # ====================================================================================
    print("\n--- 👨‍🎓 Phase 2: Training the Student (LCD) ---")

    # Freeze the encoder and HCD parameters as per the paper's procedure[cite: 40, 132].
    for param in model.source_encoder.parameters(): param.requires_grad = False
    for param in model.channel_encoder.parameters(): param.requires_grad = False
    for param in model.source_decoder_high.parameters(): param.requires_grad = False
    for param in model.channel_decoder_high.parameters(): param.requires_grad = False

    # Unfreeze only the LCD parameters
    for param in model.source_decoder_low.parameters(): param.requires_grad = True
    for param in model.channel_decoder_low.parameters(): param.requires_grad = True

    # --- 💡 Technique 1: Partial Transfer Learning ---
    if use_transfer_learning:
        student_checkpoint_path = CHECKPOINT_DIR / "student_checkpoint.pt"
        if not os.path.exists(student_checkpoint_path):
            print("🚀 Applying Partial Transfer Learning for the first time...")
            teacher_state_dict = model.source_decoder_high.state_dict()
            student_state_dict = model.source_decoder_low.state_dict()

            transferred_count = 0
            for name, param in teacher_state_dict.items():
                if name in student_state_dict and param.shape == student_state_dict[name].shape:
                    student_state_dict[name].copy_(param.data)
                    transferred_count += 1
            model.source_decoder_low.load_state_dict(student_state_dict)
            print(f"  -> Transferred weights for {transferred_count} layers.")

    # Parameters for the student model
    student_params = list(model.source_decoder_low.parameters()) + \
                     list(model.channel_decoder_low.parameters())
    optimizer_student = optim.Adam(student_params, lr=lr)

    # <> Define path and load checkpoint for the student model
    latest_student_checkpoint_path = CHECKPOINT_DIR / "student_latest.pt"
    best_student_checkpoint_path = CHECKPOINT_DIR / "student_best.pt"

    print(f"Loading latest student state from: {latest_student_checkpoint_path}")
    start_epoch_student, _ = load_checkpoint(latest_student_checkpoint_path, model, optimizer_student)

    print(f"Loading best student loss from: {best_student_checkpoint_path}")
    _, best_loss_student = load_checkpoint(best_student_checkpoint_path, None, None)

    # This loop trains the student (LCD) using knowledge distillation[cite: 43].
    for epoch in range(start_epoch_student, epochs):
        # Set teacher parts to eval mode
        model.source_encoder.eval()
        model.channel_encoder.eval()
        model.source_decoder_high.eval()
        # Set student parts to train mode
        model.source_decoder_low.train()

        total_loss_student = 0
        num_batches_student = 0

        for images in dataloader:
            images = images.to(device)
            snr = torch.FloatTensor(np.random.choice([1, 3, 5, 7], size=images.shape[0])).to(device)
            # 1. Encode and add noise ONCE (no gradients needed here)
            with torch.no_grad():
                received_signal, final_res, final_dim = model.encode_and_add_noise(
                    images, snr, add_noise=True
                )

            optimizer_student.zero_grad()

            # Get the student's output
            reconstructed_student = model.decode(
                received_signal, final_res, final_dim, user_type='low'
            )

            # Loss 1: The student's ability to reconstruct the original image[cite: 145].
            loss_main = criterion(reconstructed_student, images)
            total_loss = loss_main

            # --- 💡 Technique 2: Knowledge Distillation ---
            if use_kd:
                # The student mimics the teacher's output[cite: 148].
                with torch.no_grad(): # Teacher's forward pass requires no gradients
                    reconstructed_teacher = model.decode(
                        received_signal, final_res, final_dim, user_type='high'
                    )

                # Loss 2: The distillation loss between student and teacher outputs[cite: 144].
                loss_distill = criterion(reconstructed_student, reconstructed_teacher)

                # The final training loss is a combination of the two[cite: 145, 148].
                total_loss = loss_main + alpha * loss_distill

            total_loss.backward()
            optimizer_student.step()

            total_loss_student += total_loss.item()
            num_batches_student += 1

        avg_loss = total_loss_student / num_batches_student
        print(f'Phase 2 - Epoch {epoch+1}/{epochs}, Student Avg Loss: {avg_loss:.6f}') # Print avg loss
        
        print("   -> Saving latest student checkpoint...")
        save_checkpoint(model, optimizer_student, epoch + 1, avg_loss, latest_student_checkpoint_path)
        
        # <> Save checkpoint after each epoch
        if avg_loss < best_loss_student:
            best_loss_student = avg_loss
            print(f"  -> New best loss for student: {best_loss_student:.6f}. Saving checkpoint.")
            save_checkpoint(model, optimizer_student, epoch + 1, best_loss_student, best_student_checkpoint_path)

    print("✅ Phase 2 Complete. Student model is trained.")
    
    
    

def calculate_metrics_safe(img1, img2):
    """Safe metric calculation with error handling."""
    try:
        # PSNR
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * math.log10(1.0 / math.sqrt(mse))

        # Simplified SSIM
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        sigma1 = torch.std(img1)
        sigma2 = torch.std(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))

        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

        return psnr, ssim.item()
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 0, 0

def evaluate_safe(model, test_dataloader, snr_values=[1, 3, 5, 7]):
    """Safe evaluation with error handling."""
    model.eval()
    results = {'high': {}, 'low': {}}

    print("\n--- Evaluating model ---")

    with torch.no_grad():
        for user_type in ['high', 'low']:
            for snr_val in snr_values:
                psnr_values = []
                ssim_values = []

                try:
                    # Limit evaluation batches for speed and memory
                    for batch_idx, images in enumerate(test_dataloader):
                        if batch_idx >= 50:
                            break

                        images = images.to(device)
                        # Limit batch size during evaluation to prevent OOM on larger models
                        batch_size = min(images.shape[0], 4)
                        images = images[:batch_size]
                        snr = torch.FloatTensor([snr_val]).expand(batch_size).to(device)

                        try:
                            reconstructed = model(images, snr, user_type=user_type, add_noise=True)

                            for i in range(batch_size):
                                psnr, ssim = calculate_metrics_safe(images[i:i+1], reconstructed[i:i+1])
                                psnr_values.append(psnr)
                                ssim_values.append(ssim)

                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                clear_memory()
                                print(f"OOM during evaluation at SNR {snr_val}, skipping batch")
                                continue
                            raise e

                        if batch_idx % 10 == 0:
                            clear_memory()

                except Exception as e:
                    print(f"Error evaluating {user_type} at SNR {snr_val}: {e}")
                    psnr_values = [0]
                    ssim_values = [0]

                results[user_type][snr_val] = {
                    'psnr': np.mean(psnr_values) if psnr_values else 0,
                    'ssim': np.mean(ssim_values) if ssim_values else 0
                }

                print(f"{user_type.upper()} user at SNR={snr_val}dB: "
                      f"PSNR={results[user_type][snr_val]['psnr']:.2f}, "
                      f"SSIM={results[user_type][snr_val]['ssim']:.3f}")

    # Save results to a file
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_path}")
    return results

def visualize_safe(model, test_dataloader, snr_values=[1, 5, 7]):
    """Safe visualization of model reconstructions with error handling."""
    model.eval()
    print("\n--- Visualizing reconstructions ---")

    try:
        # Get a few test images
        test_images = next(iter(test_dataloader))[:2].to(device)  # Visualize 2 images

        num_images = test_images.shape[0]
        num_snrs = len(snr_values)

        fig, axes = plt.subplots(num_snrs, num_images * 3, figsize=(num_images * 6, num_snrs * 2.5))
        if num_snrs == 1:
            axes = axes.reshape(1, -1) # Ensure axes is 2D
        if num_images * 3 == 1:
            axes = axes.reshape(-1, 1)

        with torch.no_grad():
            for snr_idx, snr_val in enumerate(snr_values):
                snr = torch.FloatTensor([snr_val]).expand(num_images).to(device)

                # Get reconstructions for both user types
                reconstructed_high = model(test_images, snr, user_type='high', add_noise=True)
                reconstructed_low = model(test_images, snr, user_type='low', add_noise=True)

                for img_idx in range(num_images):
                    original_img = test_images[img_idx]
                    high_recon_img = reconstructed_high[img_idx]
                    low_recon_img = reconstructed_low[img_idx]

                    # Calculate metrics
                    psnr_high, ssim_high = calculate_metrics_safe(original_img, high_recon_img)
                    psnr_low, ssim_low = calculate_metrics_safe(original_img, low_recon_img)

                    # Denormalize and convert to numpy for plotting
                    orig_np = original_img.cpu().permute(1, 2, 0).numpy()
                    high_np = high_recon_img.cpu().permute(1, 2, 0).numpy()
                    low_np = low_recon_img.cpu().permute(1, 2, 0).numpy()

                    # Plot Original
                    ax = axes[snr_idx, img_idx * 3]
                    ax.imshow(np.clip(orig_np, 0, 1))
                    ax.set_title(f"Original (SNR: {snr_val}dB)")
                    ax.axis('off')

                    # Plot High-User Reconstruction
                    ax = axes[snr_idx, img_idx * 3 + 1]
                    ax.imshow(np.clip(high_np, 0, 1))
                    ax.set_title(f"High-User\nPSNR:{psnr_high:.2f} SSIM:{ssim_high:.3f}")
                    ax.axis('off')

                    # Plot Low-User Reconstruction
                    ax = axes[snr_idx, img_idx * 3 + 2]
                    ax.imshow(np.clip(low_np, 0, 1))
                    ax.set_title(f"Low-User\nPSNR:{psnr_low:.2f} SSIM:{ssim_low:.3f}")
                    ax.axis('off')

        plt.tight_layout()
        viz_path = RESULTS_DIR / "reconstruction_visualization.png"
        plt.savefig(viz_path)
        print(f"Visualization saved to {viz_path}")
        plt.show()

    except Exception as e:
        print(f"An error occurred during visualization: {e}")
        clear_memory()

def debug_model_behavior(model, test_images, snr_values=[1, 5, 7]):
    """Debug function to check if model responds to SNR changes."""
    print("Debugging model SNR response...")
    model.eval()

    with torch.no_grad():
        test_img = test_images[:1]  # Single image

        print("Testing HIGH user:")
        for snr_val in snr_values:
            snr = torch.FloatTensor([snr_val]).to(device)

            # Without noise
            recon_no_noise = model(test_img, snr, user_type='high', add_noise=False)
            mse_no_noise = F.mse_loss(recon_no_noise, test_img).item()

            # With noise
            recon_with_noise = model(test_img, snr, user_type='high', add_noise=True)
            mse_with_noise = F.mse_loss(recon_with_noise, test_img).item()

            print(f"  SNR {snr_val}dB: MSE no_noise={mse_no_noise:.4f}, MSE with_noise={mse_with_noise:.4f}")

        print("Testing LOW user:")
        for snr_val in snr_values:
            snr = torch.FloatTensor([snr_val]).to(device)

            # Without noise
            recon_no_noise = model(test_img, snr, user_type='low', add_noise=False)
            mse_no_noise = F.mse_loss(recon_no_noise, test_img).item()

            # With noise
            recon_with_noise = model(test_img, snr, user_type='low', add_noise=True)
            mse_with_noise = F.mse_loss(recon_with_noise, test_img).item()

            print(f"  SNR {snr_val}dB: MSE no_noise={mse_no_noise:.4f}, MSE with_noise={mse_with_noise:.4f}")

if __name__ == '__main__':
    # --- Hyperparameters ---
    # Using conservative values for memory safety
    CROP_SIZE = 128 # Define a reasonable crop size for DIV2K
    BATCH_SIZE = 8
    EPOCHS = 122
    LEARNING_RATE = (5e-4)
    COMPRESSION_RATIO = 16


    div2k_train_path = extract_path
    #div2k_test_path = '/content/drive/MyDrive/DIV2K_valid_HR.zip'

    # --- Data Loading ---
    train_dataset = DIV2KDataset(data_dir=div2k_train_path, split='train', crop_size=CROP_SIZE)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    test_dataset = DIV2KDataset(data_dir=div2k_train_path, split='valid', crop_size=CROP_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    model = CompactSemComSystem(img_size=CROP_SIZE, compression_ratio=COMPRESSION_RATIO).to(device)

    # --- Training ---
    try:
      train_with_paper_techniques(
            model,
            train_dataloader,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            use_transfer_learning=True,
            use_kd=True
        )
    except Exception as e:
        print(f"Fatal error during training: {e}")
        clear_memory()
        
    print("\n--- Loading best student model for final evaluation ---")
    best_student_checkpoint_path = CHECKPOINT_DIR / "student_best.pt"
    load_checkpoint(best_student_checkpoint_path, model, optimizer=None)
    
    '''
    print("\n--- Loading best teacher model for final evaluation ---")
    best_teacher_checkpoint_path = CHECKPOINT_DIR / "teacher_best.pt"
    load_checkpoint(best_teacher_checkpoint_path, model, optimizer=None)
    '''

    # Add this after your training
    #test_images = next(iter(test_dataloader))[:1].to(device)
    #debug_model_behavior(model, test_images)

    # --- Evaluation ---
    try:
        results = evaluate_safe(model, test_dataloader)
    except Exception as e:
        print(f"Fatal error during evaluation: {e}")
        clear_memory()

    # --- Visualization ---
    try:
        visualize_safe(model, test_dataloader)
    except Exception as e:
        print(f"Fatal error during visualization: {e}")

        clear_memory()
