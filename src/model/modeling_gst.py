from typing import Optional, Tuple, Union
from transformers import PreTrainedModel, PretrainedConfig

import torch
import torch.utils.checkpoint
from torch import nn


class GSTConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_size = kwargs.get("input_size", 14)
        self.max_length = kwargs.get("max_length", 1024)
        self.hidden_size = kwargs.get("hidden_size", 512)
        self.num_heads = kwargs.get("num_heads", 8)
        self.num_layers = kwargs.get("num_layers", 12)
        self.dropout = kwargs.get("dropout", 0.1)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-12)

class GSTEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.input_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        input_vec: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        embeddings = self.linear(input_vec)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class GSTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_heads, dropout=config.dropout)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.linear1_act = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        attn_output = self.dropout1(attn_output)
        hidden_states = self.layer_norm1(hidden_states + attn_output)

        # Feed-forward
        feed_forward_output = self.linear1(hidden_states)
        feed_forward_output = self.linear1_act(feed_forward_output)
        feed_forward_output = self.dropout2(feed_forward_output)
        feed_forward_output = self.linear2(feed_forward_output)
        feed_forward_output = self.dropout3(feed_forward_output)
        hidden_states = self.layer_norm2(hidden_states + feed_forward_output)

        return hidden_states
    
class GSTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GSTConfig
    base_model_prefix = "gst"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.MultiheadAttention):
            # Initialize the attention weights
            module.in_proj_weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        


class GSTModel(GSTPreTrainedModel):


    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = GSTEmbeddings(config)
        self.hidden_layers = nn.ModuleList([GSTLayer(config) for _ in range(config.num_layers)])
        self.output_head = nn.Linear(config.hidden_size, config.input_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_vecs: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor]]:

        # Check if input_vecs is provided
        if input_vecs is None:
            raise ValueError("input_vecs must be provided")

        # Apply embeddings
        embedding_output = self.embeddings(input_vecs)

        # Apply hidden layers
        hidden_states = embedding_output
        for layer in self.hidden_layers:
            hidden_states = layer(hidden_states)

        # Apply output head
        output = self.output_head(hidden_states)

        return output
