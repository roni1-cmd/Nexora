"""
Nexora AI Model Architecture

This module defines the core neural network architecture for Nexora,
implementing a transformer-based model for sequence processing tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism as described in 'Attention is All You Need'"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V, and output
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.wo(context)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """Feed-forward layer in transformer architecture"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequence models"""
    
    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single layer of the transformer encoder"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention block with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward block with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class NexoraModel(nn.Module):
    """
    Nexora core model architecture based on the transformer encoder
    
    This implementation serves as the foundation for various NLP and sequence processing tasks.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Extract config values
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_layers = config["model"]["num_layers"]
        self.num_heads = config["model"]["num_heads"]
        self.ff_dim = config["model"]["ff_dim"]
        self.dropout_rate = config["model"]["dropout_rate"]
        self.max_seq_length = config["model"]["max_sequence_length"]
        self.vocab_size = config["data"]["vocab_size"]
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.embedding_dim, 
            self.max_seq_length, 
            self.dropout_rate
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                self.embedding_dim,
                self.num_heads,
                self.ff_dim,
                self.dropout_rate
            ) for _ in range(self.num_layers)
        ])
        
        # Final normalization layer
        self.final_norm = nn.LayerNorm(self.embedding_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the Nexora model
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_length]
            attention_mask: Optional mask tensor [batch_size, seq_length]
            
        Returns:
            output: Sequence of hidden states [batch_size, seq_length, embedding_dim]
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # Create attention mask for self-attention layers
        # Convert to binary mask and add sequence dimension for broadcasting
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Apply token embedding and positional encoding
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        
        # Pass through each encoder layer
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, extended_mask)
            
        # Apply final layer normalization
        output = self.final_norm(x)
        
        return output
        
    def get_embedding_output(self, input_ids):
        """Get embeddings for input tokens"""
        return self.token_embedding(input_ids)
    
    def export_to_onnx(self, save_path, input_sample=None):
        """Export model to ONNX format for optimized inference"""
        if input_sample is None:
            # Create dummy input for ONNX export
            input_sample = {
                'input_ids': torch.randint(0, self.vocab_size, (1, 64)),
                'attention_mask': torch.ones(1, 64)
            }
            
        torch.onnx.export(
            self,                                      # model being exported
            (input_sample['input_ids'], input_sample['attention_mask']),  # model input
            save_path,                                 # output file
            export_params=True,                        # store model weights inside the model file
            opset_version=12,                          # ONNX version to export to
            do_constant_folding=True,                  # optimize constants
            input_names=['input_ids', 'attention_mask'], # input names
            output_names=['output'],                   # output names
            dynamic_axes={                             # dynamic axes
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        print(f"Model exported to {save_path}")


class NexoraForSequenceClassification(nn.Module):
    """Nexora model adapted for sequence classification tasks"""
    
    def __init__(self, config, num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.nexora = NexoraModel(config)
        self.classifier = nn.Linear(config["model"]["embedding_dim"], num_labels)
        self.dropout = nn.Dropout(config["model"]["dropout_rate"])
        
    def forward(self, input_ids, attention_mask=None):
        sequence_output = self.nexora(input_ids, attention_mask)
        
        # Use [CLS] token representation (first token) for classification
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits


class NexoraForTokenClassification(nn.Module):
    """Nexora model adapted for token classification tasks like NER"""
    
    def __init__(self, config, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.nexora = NexoraModel(config)
        self.classifier = nn.Linear(config["model"]["embedding_dim"], num_labels)
        self.dropout = nn.Dropout(config["model"]["dropout_rate"])
        
    def forward(self, input_ids, attention_mask=None):
        sequence_output = self.nexora(input_ids, attention_mask)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        return logits


if __name__ == "__main__":
    # Test model instantiation with dummy config
    config = {
        "model": {
            "embedding_dim": 768,
            "num_layers": 6,
            "num_heads": 8,
            "ff_dim": 2048,
            "dropout_rate": 0.1,
            "max_sequence_length": 512
        },
        "data": {
            "vocab_size": 30000
        }
    }
    
    model = NexoraModel(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass with dummy inputs
    batch_size = 4
    seq_length = 128
    
    input_ids = torch.randint(0, config["data"]["vocab_size"], (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    outputs = model(input_ids, attention_mask)
    print(f"Output shape: {outputs.shape}")
