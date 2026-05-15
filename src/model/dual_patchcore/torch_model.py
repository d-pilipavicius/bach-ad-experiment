# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
This model is a changed version of the PatchCore anomaly detection model.

Paper: https://arxiv.org/abs/2106.08265

It adds an additional anomaly memory bank, but this implementation is just a prototype.
Actual implementation should determine better ways of filtering out feature maps for the anomaly memory bank.
"""

from enum import Enum
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin, KCenterGreedy, TimmFeatureExtractor

from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

if TYPE_CHECKING:
  from anomalib.data.utils.tiler import Tiler

DEFAULT_CHUNK_SIZE = 1024


class TrainType(Enum):
  GOOD = 0
  ANOMALOUS = 1

class DualPatchcoreModel(DynamicBufferMixin, nn.Module):

  def __init__(
    self,
    layers: Sequence[str],
    backbone: str = "wide_resnet50_2",
    pre_trained: bool = True,
    num_neighbors: int = 9,
  ) -> None:
    super().__init__()
    self.tiler: Tiler | None = None

    self.backbone = backbone
    self.layers = layers
    self.num_neighbors = num_neighbors

    self.feature_extractor = TimmFeatureExtractor(
      backbone=self.backbone,
      pre_trained=pre_trained,
      layers=self.layers,
    ).eval()
    self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
    self.anomaly_map_generator = AnomalyMapGenerator()
    self.memory_bank: torch.Tensor
    self.anomaly_memory_bank: torch.Tensor
    self.register_buffer("memory_bank", torch.empty(0))
    self.register_buffer("anomaly_memory_bank", torch.empty(0))
    self.embedding_store: list[torch.tensor] = []
    self.train_type = TrainType.GOOD

  def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
    """Process input tensor through the model.

    During training, returns embeddings extracted from the input. During
    inference, returns anomaly maps and scores computed by comparing input
    embeddings against the memory bank.

    Args:
      input_tensor (torch.Tensor): Input images of shape
        ``(batch_size, channels, height, width)``.

    Returns:
      torch.Tensor | InferenceBatch: During training, returns embeddings.
        During inference, returns ``InferenceBatch`` containing anomaly
        maps and scores.
    """

    input_tensor = input_tensor.type(self.memory_bank.dtype)
    output_size = input_tensor.shape[-2:]
    if self.tiler:
      input_tensor = self.tiler.tile(input_tensor)

    with torch.no_grad():
      features = self.feature_extractor(input_tensor)

    features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
    embedding = self.generate_embedding(features)

    if self.tiler:
      embedding = self.tiler.untile(embedding)

    batch_size, _, width, height = embedding.shape
    embedding = self.reshape_embedding(embedding)

    if self.training:
      self.embedding_store.append(embedding)
      return embedding

    # Ensure memory bank is not empty
    if self.memory_bank.size(0) == 0:
      msg = "Memory bank is empty. Cannot provide anomaly scores"
      raise ValueError(msg)

    # apply nearest neighbor search
    good_patch_scores, good_locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1, memory_bank=self.memory_bank)
    anom_patch_scores, anom_locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1, memory_bank=self.anomaly_memory_bank)
    
    # reshape to batch dimension
    good_patch_scores = good_patch_scores.reshape((batch_size, -1))
    good_locations = good_locations.reshape((batch_size, -1))
    anom_patch_scores = anom_patch_scores.reshape((batch_size, -1))
    anom_locations = anom_locations.reshape((batch_size, -1))

    # compute anomaly score
    pred_score = self.compute_anomaly_score(good_patch_scores, good_locations, embedding, self.memory_bank)

    # calculate best odds
    patch_scores = good_patch_scores.clone()
    mask = anom_patch_scores < good_patch_scores
    patch_scores[mask] = good_patch_scores[mask] + anom_patch_scores[mask]
    print(f"mask ratio: {mask.float().mean().item()}")

    # reshape to w, h
    patch_scores = patch_scores.reshape((batch_size, 1, width, height))

    # get anomaly map
    anomaly_map = self.anomaly_map_generator(patch_scores, output_size)


    return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

  def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
    """Generate embedding by concatenating multi-scale feature maps.

    Combines feature maps from different CNN layers by upsampling them to a
    common size and concatenating along the channel dimension.

    Args:
      features (dict[str, torch.Tensor]): Dictionary mapping layer names to
        feature tensors extracted from the backbone CNN.

    Returns:
      torch.Tensor: Concatenated feature embedding of shape
        ``(batch_size, num_features, height, width)``.
    """

    embeddings = features[self.layers[0]]
    for layer in self.layers[1:]:
      layer_embedding = features[layer]
      layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
      embeddings = torch.cat((embeddings, layer_embedding), 1)

    return embeddings

  @staticmethod
  def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Reshape embedding tensor for patch-wise processing.

    Converts a 4D embedding tensor into a 2D matrix where each row represents
    a patch embedding vector.

    Args:
      embedding (torch.Tensor): Input embedding tensor of shape
        ``(batch_size, embedding_dim, height, width)``.

    Returns:
      torch.Tensor: Reshaped embedding tensor of shape
        ``(batch_size * height * width, embedding_dim)``.
    """
    embedding_size = embedding.size(1)
    return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)


  def subsample_embedding(self, sampling_ratio: float) -> None:
    """Subsample the memory_banks embeddings using coreset selection.

    Uses k-center-greedy coreset subsampling to select a representative
    subset of patch embeddings to store in the memory bank.

    Args:
      sampling_ratio (float): Fraction of embeddings to keep, in range (0,1].
    """

    if len(self.embedding_store) == 0:
      msg = "Embedding store is empty. Cannot perform coreset selection."
      raise ValueError(msg)

    if self.train_type is TrainType.GOOD:
      # Coreset Subsampling
      self.memory_bank = torch.vstack(self.embedding_store)
      self.embedding_store.clear()

      sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
      self.memory_bank = sampler.sample_coreset()
    else:
      if self.memory_bank.size(0) == 0:
        msg = "Good memory bank is empty. Cannot fill anomalous memory bank."
        raise ValueError(msg)
      
      self.anomaly_memory_bank = torch.vstack(self.embedding_store)
      self.embedding_store.clear()

      self.anomaly_memory_bank = self.cos_similarity_embeddings_reduction(self.anomaly_memory_bank)
      self.filter_anomalous_with_nn()
  
  # Used for similar embedding reduction from the memory bank by a selected threshold
  def cos_similarity_embeddings_reduction(self, memory_bank: torch.Tensor, threshold: float = 0.97) -> torch.Tensor:
    temp_bank_n = F.normalize(memory_bank, dim=1)

    sim_matrix = temp_bank_n @ temp_bank_n.T
    sim_matrix.fill_diagonal_(-1.0)

    max_sim = sim_matrix.max(dim=1).values
    mask = max_sim < threshold

    return memory_bank[mask]
  
  def filter_anomalous_with_nn(self, assume_anomalous: float = 0.90) -> None:
    distances, _ = self.nearest_neighbors(
      embedding=self.anomaly_memory_bank,
      n_neighbors=1,
      memory_bank=self.memory_bank
    )

    threshold = torch.quantile(distances, assume_anomalous)
    mask = distances > threshold 
    self.anomaly_memory_bank = self.anomaly_memory_bank[mask]

  @staticmethod
  def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances between two sets of vectors.

    Implements an efficient matrix computation of Euclidean distances between
    all pairs of vectors in ``x`` and ``y`` without using ``torch.cdist()``.

    Args:
      x (torch.Tensor): First tensor of shape ``(n, d)``.
      y (torch.Tensor): Second tensor of shape ``(m, d)``.

    Returns:
      torch.Tensor: Distance matrix of shape ``(n, m)`` where element
        ``(i,j)`` is the distance between row ``i`` of ``x`` and row
        ``j`` of ``y``.

    Note:
      This implementation avoids using ``torch.cdist()`` for better
      compatibility with ONNX export and OpenVINO conversion.
    """
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    y_norm = y.pow(2).sum(dim=-1, keepdim=True)
    res = torch.matmul(x, y.transpose(-2, -1))
    res.mul_(-2)
    res.add_(x_norm)
    res.add_(y_norm.transpose(-2, -1))
    return res.clamp_min_(0).sqrt_()

  def nearest_neighbors(
    self, 
    embedding: torch.Tensor, 
    n_neighbors: int, 
    memory_bank: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Find nearest neighbors in memory bank for input embeddings.

    Uses brute force search with Euclidean distance to find the closest
    matches in the memory bank for each input embedding. Processes embeddings
    in chunks to reduce memory usage for large embedding sets.

    Args:
      embedding (torch.Tensor): Query embeddings to find neighbors for.
      n_neighbors (int): Number of nearest neighbors to return.
      memory_bank (torch.Tensor): Memory bank used in embedding search.

    Returns:
      tuple[torch.Tensor, torch.Tensor]: Tuple containing:
        - Distances to nearest neighbors (shape: ``(n, k)``)
        - Indices of nearest neighbors (shape: ``(n, k)``)
        where ``n`` is number of query embeddings and ``k`` is
        ``n_neighbors``.
    """
    n = embedding.shape[0]
    chunk_size = DEFAULT_CHUNK_SIZE

    if n <= chunk_size:
      # Small embedding set: process all at once
      distances = self.euclidean_dist(embedding, memory_bank)
      if n_neighbors == 1:
        # when n_neighbors is 1, speed up computation by using min instead of topk
        patch_scores, locations = distances.min(1)
      else:
        patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
    else:
      # Large embedding set: process in chunks
      all_scores = []
      all_locations = []

      for start_idx in range(0, n, chunk_size):
        end_idx = min(start_idx + chunk_size, n)
        embedding_chunk = embedding[start_idx:end_idx]

        # Compute distances for this chunk against full memory bank
        distances = self.euclidean_dist(embedding_chunk, memory_bank)

        # Find top-k neighbors immediately and discard full distance matrix
        if n_neighbors == 1:
          chunk_scores, chunk_locations = distances.min(1)
        else:
          chunk_scores, chunk_locations = distances.topk(k=n_neighbors, largest=False, dim=1)

        all_scores.append(chunk_scores)
        all_locations.append(chunk_locations)
        del distances  # Drop reference to allow garbage collection

      # Concatenate results from all chunks
      patch_scores = torch.cat(all_scores, dim=0)
      locations = torch.cat(all_locations, dim=0)

    return patch_scores, locations

  def compute_anomaly_score(
    self,
    patch_scores: torch.Tensor,
    locations: torch.Tensor,
    embedding: torch.Tensor,
    memory_bank: torch.Tensor,
  ) -> torch.Tensor:
    """Compute image-level anomaly scores.

    Implements the paper's weighted scoring mechanism that considers both
    the distance to nearest neighbors and the local neighborhood structure
    in the memory bank.

    Args:
      patch_scores (torch.Tensor): Patch-level anomaly scores.
      locations (torch.Tensor): Memory bank indices of nearest neighbors.
      embedding (torch.Tensor): Input embeddings that generated the scores.
      memory_bank (torch.Tensor): Used memory bank.

    Returns:
      torch.Tensor: Image-level anomaly scores.

    Note:
      When ``num_neighbors=1``, returns the maximum patch score directly.
      Otherwise, computes weighted scores using neighborhood information.
    """
    # Don't need to compute weights if num_neighbors is 1
    if self.num_neighbors == 1:
      return patch_scores.amax(1)
    batch_size, num_patches = patch_scores.shape
    # 1. Find the patch with the largest distance to it's nearest neighbor in each image
    max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
    # m^test,* in the paper
    max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
    # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
    score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
    nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
    # 3. Find the support samples of the nearest neighbor in the membank
    nn_sample = memory_bank[nn_index, :]  # m^* in the paper
    # indices of N_b(m^*) in the paper
    memory_bank_effective_size = memory_bank.shape[0]  # edge case when memory bank is too small
    _, support_samples = self.nearest_neighbors(
      nn_sample,
      n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
      memory_bank=memory_bank
    )
    # 4. Find the distance of the patch features to each of the support samples
    distances = self.euclidean_dist(max_patches_features.unsqueeze(1), memory_bank[support_samples])
    # 5. Apply softmax to find the weights
    weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
    # 6. Apply the weight factor to the score
    return weights * score  # s in the paper
