import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import numpy as np


class MLP(nn.Module):
  """ Very simple multi-layer perceptron (also called FFN)"""

  def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
    super().__init__()
    self.num_layers = num_layers
    h = [hidden_dim] * (num_layers - 1)
    self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

  def forward(self, x):
    for i, layer in enumerate(self.layers):
      x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
    return x


class PartDecoder(nn.Module):
  def __init__(self,
               num_part_classes,
               input_dim,
               hidden_dim,
               mask_dim,
               ):
    super().__init__()

    self.num_part_classes = num_part_classes

    self.mask_head = MLP(input_dim, hidden_dim, mask_dim * num_part_classes, num_layers=3)

  def forward(self, queries, num_parts_per_query, mask_features, part_ids_per_query=None):
    # queries shape: [Nb, num_queries (padded to max), hidden_dim]

    mask_embeds = self.mask_head(queries)
    mask_embeds = torch.tensor_split(mask_embeds, self.num_part_classes, dim=2)

    # mask_embeds_total is [Nb, num_queries, num_partcls, num_channels]
    mask_embeds_total = torch.stack(mask_embeds, dim=2)
    embeds_shape = mask_embeds_total.shape

    # mask_embeds_total is [Nb, num_queries * num_partcls, num_channels]
    mask_embeds_total = mask_embeds_total.view(embeds_shape[0],
                                               embeds_shape[1] * embeds_shape[2],
                                               embeds_shape[3])

    # outputs_mask shape is [Nb, num_queries * num_partcls, height, width]
    outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embeds_total, mask_features)
    # outputs_mask shape is [Nb, num_queries, num_partcls, height, width]
    outputs_mask = outputs_mask.view(embeds_shape[0],
                                     embeds_shape[1],
                                     embeds_shape[2],
                                     outputs_mask.shape[2],
                                     outputs_mask.shape[3])

    # num_parts_per_query: list of length batch_size
    gather_batch_dim = []
    gather_num_queries = []
    gather_num_partcls = []

    for i, num_parts in enumerate(num_parts_per_query):
      if len(num_parts) != 0:
        if part_ids_per_query is None:
          idx_partcls = torch.cat([torch.arange(0, num_part) for num_part in num_parts], dim=0)
        else:
          idx_partcls = torch.cat([pt_idx for pt_idx in part_ids_per_query[i]])
        idx_queries = torch.cat([torch.full_like(torch.arange(0, num_part), e, dtype=torch.long)
                                 for e, num_part in enumerate(num_parts)], dim=0)
        idx_batch = torch.full_like(idx_partcls, fill_value=i, dtype=torch.long)

        gather_batch_dim.append(idx_batch)
        gather_num_partcls.append(idx_partcls)
        gather_num_queries.append(idx_queries)

    if len(gather_batch_dim) != 0:
      gather_batch_dim = torch.cat(gather_batch_dim, dim=0)
      gather_num_queries = torch.cat(gather_num_queries, dim=0)
      gather_num_partcls = torch.cat(gather_num_partcls, dim=0)

    else:
      gather_batch_dim = torch.zeros([0], dtype=torch.long, device=mask_features.device)
      gather_num_queries = torch.zeros([0], dtype=torch.long, device=mask_features.device)
      gather_num_partcls = torch.zeros([0], dtype=torch.long, device=mask_features.device)

    output_masks = outputs_mask[gather_batch_dim, gather_num_queries, gather_num_partcls]

    return output_masks