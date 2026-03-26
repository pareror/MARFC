# =============================================================================
# EVALUATION SCRIPT FOR CREATIVITY SCORE-BASED RECOMMENDATION RERANKING
# =============================================================================
# This script evaluates trained recommendation models and applies a reranking
# based on a creativity score formula that combines relevance, novelty, and
# unexpectedness. It integrates RecBole metrics with custom novelty calculations.
# =============================================================================

import os
import sys
import glob
import gc
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from recbole.data.dataloader.knowledge_dataloader import KnowledgeBasedDataLoader
from recbole.data.interaction import Interaction
from recbole.evaluator import Evaluator
from recbole.evaluator.collector import DataStruct

# =============================================================================
# FIX FOR RECBOLE COMPATIBILITY
# =============================================================================
# Patch for KnowledgeBasedDataLoader missing 'dataset' attribute.
# This adds a property to access the internal _dataset attribute.
if not hasattr(KnowledgeBasedDataLoader, 'dataset'):
    setattr(KnowledgeBasedDataLoader, 'dataset', property(lambda self: self._dataset))

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Recommendation list sizes: TOPKS are the final N items to evaluate on,
# CANDIDATE_KS are the pool sizes for the reranking algorithm
TOPKS = [5, 10]  # Final N to calculate metrics on
CANDIDATE_KS = [50, 100]  # Candidate K for reranking

# Batch sizes for different phases: evaluation with RecBole, generating recommendations
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 128))
GEN_BATCH_SIZE = int(os.environ.get("GEN_BATCH_SIZE", 64))

# Chunk size for processing large item catalogs in GPU
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 10000))

# Output file paths and checkpoint directory
RESULTS_FILE = "benchmark_creativity_score_reranking.csv"
LOG_FILE = "eval_creativity_score_reranking.log"
CKPT_ROOT = os.environ.get("CKPT_ROOT", os.path.join(SCRIPT_DIR, "saved"))

# =============================================================================
# CREATIVITY SCORE FORMULA
# =============================================================================
# Creativity Score = (0.33 * relevance) + (0.33 * novelty) + (0.33 * unexpectedness)
# Each component ranges from [0, 1] after normalization
WEIGHT_RELEVANCE = 0.33
WEIGHT_NOVELTY = 0.33
WEIGHT_UNEXPECTEDNESS = 0.33

# =============================================================================
# CHECKPOINTS TO EVALUATE
# =============================================================================
# Set of checkpoint filenames to evaluate. Leave empty to evaluate all checkpoints.
# Example format: "Pop-Dec-22-2025_12-31-19.pth"
TARGET_CKPTS = {
    #Example of checkpoint format: "DMF-Dec-22-2025_12-30-45.pth",
    "Pop-Dec-22-2025_12-31-19.pth",
    "ItemKNN-Dec-22-2025_12-33-14.pth",
}

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
# Automatically detect and use GPU if available, otherwise fall back to CPU
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

# =============================================================================
# SCIPY COMPATIBILITY FIX
# =============================================================================
# Add missing _update method to scipy's dok_matrix if needed
try:
    import scipy.sparse as _sp
    if not hasattr(_sp.dok_matrix, "_update"):
        def _dok_update(self, data_dict):
            for k, v in data_dict.items():
                self[k] = v
            return self
        _sp.dok_matrix._update = _dok_update
except Exception:
    pass


# =============================================================================
# LOGGING UTILITY
# =============================================================================
class TeeLogger:
    """Dual output logger: writes to both console and file simultaneously."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        # Write to both terminal and log file
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# =============================================================================
# DATA BUILDING UTILITIES
# =============================================================================
def build_pop_counter(inter_feat, iid_field):
    """Build item popularity counter from training interactions.
    
    Args:
        inter_feat: interaction features (DataFrame-like)
        iid_field: name of the item ID field
    
    Returns:
        Counter: dictionary mapping item_id -> popularity count
    """
    pop = Counter()
    for iid in inter_feat[iid_field].numpy():
        pop[int(iid)] += 1
    return pop


def build_ground_truth(inter_feat, uid_field, iid_field, label_field=None):
    """Build ground truth from test interactions.
    
    If label_field is present (from threshold), filter only positive label=1.
    Otherwise, use all interactions as positive.
    
    Args:
        inter_feat: interaction features from test set
        uid_field: name of the user ID field
        iid_field: name of the item ID field
        label_field: optional label field name for filtering by threshold
    
    Returns:
        defaultdict: mapping user_id -> set of relevant item_ids
    """
    truth = defaultdict(set)
    uids = inter_feat[uid_field].numpy()
    iids = inter_feat[iid_field].numpy()
    
    if label_field and label_field in inter_feat:
        # Filter for label=1 (only rating >= threshold)
        labels = inter_feat[label_field].numpy()
        for u, i, label in zip(uids, iids, labels):
            if label == 1:
                truth[int(u)].add(int(i))
    else:
        # Use all interactions (threshold=None)
        for u, i in zip(uids, iids):
            truth[int(u)].add(int(i))
    
    return truth


def build_user_history_cpu(dataset):
    """Build user interaction history from training data.
    
    Creates a mapping from user_id to list of items they interacted with.
    Used to filter out already-seen items during recommendation.
    
    Args:
        dataset: RecBole dataset object with training interactions
    
    Returns:
        defaultdict: mapping user_id -> torch.LongTensor of item_ids
    """
    inter_feat = dataset.inter_feat
    users = inter_feat[dataset.uid_field].numpy()
    items = inter_feat[dataset.iid_field].numpy()
    history = defaultdict(list)
    for u, i in zip(users, items):
        history[u].append(i)
    out = defaultdict(lambda: torch.LongTensor([]))
    for u, arr in history.items():
        out[u] = torch.LongTensor(arr)
    return out


def get_vectors(model, dataset):
    """Extract user and item embeddings from trained model.
    
    Different models store embeddings in different attributes.
    This function handles various RecBole model architectures.
    Special case: DMF requires batch-wise computation.
    
    Args:
        model: trained RecBole recommendation model
        dataset: RecBole dataset object
    
    Returns:
        tuple: (user_embeddings, item_embeddings) as detached tensors
               or (None, None) if extraction fails
    """
    u_emb, i_emb = None, None
    device = next(model.parameters()).device

    def extract(obj):
        """Helper to extract tensor from embedding or tensor object."""
        if isinstance(obj, torch.nn.Embedding):
            return obj.weight.detach()
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        return None

    name = model.__class__.__name__
    if name == "DMF":
        # DMF requires batch processing due to its special architecture
        try:
            batch_size = 256
            u_list, i_list = [], []
            for i in range(0, dataset.user_num, batch_size):
                end = min(i + batch_size, dataset.user_num)
                users = torch.arange(i, end, device=device)
                emb = model.get_user_embedding(users)
                emb = model.user_fc_layers(emb)
                u_list.append(emb.detach())
            for i in range(0, dataset.item_num, batch_size):
                end = min(i + batch_size, dataset.item_num)
                items = torch.arange(i, end, device=device)
                col_indices = model.history_user_id[items].flatten()
                row_indices = torch.arange(items.shape[0], device=device).repeat_interleave(model.history_user_id.shape[1], dim=0)
                matrix_01 = torch.zeros(items.shape[0], model.n_users, device=device)
                matrix_01.index_put_((row_indices, col_indices), model.history_user_value[items].flatten())
                emb = model.item_linear(matrix_01)
                emb = model.item_fc_layers(emb)
                i_list.append(emb.detach())
            u_emb = torch.cat(u_list, dim=0)
            i_emb = torch.cat(i_list, dim=0)
            return u_emb, i_emb
        except Exception:
            return None, None

    # Try standard embedding attribute names for user embeddings
    if hasattr(model, "user_embeddings_lookup"):
        u_emb = extract(model.user_embeddings_lookup)
    elif hasattr(model, "user_embedding"):
        u_emb = extract(model.user_embedding)
    elif hasattr(model, "user_embeddings"):
        u_emb = extract(model.user_embeddings)

    # Try standard embedding attribute names for item embeddings
    if hasattr(model, "item_embeddings_lookup"):
        i_emb = extract(model.item_embeddings_lookup)
    elif hasattr(model, "item_embedding"):
        i_emb = extract(model.item_embedding)
    elif hasattr(model, "item_embeddings"):
        i_emb = extract(model.item_embeddings)
    elif hasattr(model, "entity_embedding"):  # For Knowledge Graph models
        i_emb = extract(model.entity_embedding)

    return u_emb, i_emb


def _predict_full_set(model, batch_users, dataset, device):
    """Generate prediction scores for all items for a batch of users.
    
    Processes items in chunks to manage memory usage on GPU.
    Used as fallback when model doesn't implement full_sort_predict.
    
    Args:
        model: trained recommendation model
        batch_users: tensor of user IDs [batch_size]
        dataset: RecBole dataset with item_num
        device: torch device
    
    Returns:
        tensor: prediction scores [batch_size, num_items]
    """
    num_items = dataset.item_num
    batch_size = batch_users.size(0)
    all_items = torch.arange(num_items, device=device)
    scores_matrix = torch.zeros(batch_size, num_items, device=device)
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    
    for i, user_id in enumerate(batch_users):
        # Process items in chunks to avoid OOM
        item_chunk_size = 10000
        for start_idx in range(0, num_items, item_chunk_size):
            end_idx = min(start_idx + item_chunk_size, num_items)
            current_items = all_items[start_idx:end_idx]
            current_len = len(current_items)
            u_ids = user_id.repeat(current_len)
            interaction_dict = {uid_field: u_ids, iid_field: current_items}
            interaction = Interaction(interaction_dict).to(device)
            with torch.no_grad():
                pred = model.predict(interaction)
            scores_matrix[i, start_idx:end_idx] = pred
    return scores_matrix


def generate_recommendations_with_scores_gpu(model, test_data, dataset, topk, device, user_history):
    """Generate top-K recommendations with relevance scores for all test users.
    
    Handles various model architectures:
    - LightGCN: Uses precomputed embeddings (most efficient)
    - Graph models: Use fallback prediction methods
    - Dense models: Use matrix multiplication with chunking for large catalogs
    - Others: Use full_sort_predict or fallback
    
    Always filters out items from user's training history.
    
    Args:
        model: trained recommendation model
        test_data: test dataloader with user interactions
        dataset: RecBole dataset object
        topk: number of recommendations per user
        device: torch device
        user_history: dict mapping user_id -> tensor of seen item_ids
    
    Returns:
        dict: mapping user_id -> [(item_id, score), ...] sorted by score
    """
    model.eval()
    user_field = dataset.uid_field
    user_recs = {}
    model = model.to(device)
    model_name = model.__class__.__name__
    is_lightgcn = model_name == 'LightGCN'
    is_graph_model = model_name in ['LightGCN', 'NGCF', 'GCN', 'KGCN']
    cached_user_emb = None
    cached_item_emb = None

    # LightGCN: Compute aggregated embeddings once
    if is_lightgcn:
        with torch.no_grad():
            if hasattr(model, 'restore_user_e'):
                model.restore_user_e = None
            if hasattr(model, 'restore_item_e'):
                model.restore_item_e = None
            try:
                cached_user_emb, cached_item_emb = model.computer()
                cached_user_emb = cached_user_emb.detach()
                cached_item_emb = cached_item_emb.detach()
            except AttributeError:
                print("⚠️ model.computer() not found, trying model.forward()...")
                cached_user_emb, cached_item_emb = model.forward()
                cached_user_emb = cached_user_emb.detach()
                cached_item_emb = cached_item_emb.detach()

    # Check for direct item embeddings (for matrix multiplication method)
    has_embed = hasattr(model, 'user_embedding') and hasattr(model, 'item_embedding') and not is_graph_model
    if has_embed:
        item_embed = model.item_embedding.weight.to(device)
    else:
        item_embed = None

    # Generate recommendations for each batch in test set
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Gen recs with scores", leave=False):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            batch_users = batch[user_field]
            batch_size = batch_users.size(0)

            if is_lightgcn:
                # LightGCN: use cached embeddings with dot product
                batch_u_emb = cached_user_emb[batch_users]
                scores = torch.matmul(batch_u_emb, cached_item_emb.t())
                batch_users_list = batch_users.tolist()
                n_items = scores.size(1)
                # Mask out seen items
                for i, u_id in enumerate(batch_users_list):
                    seen = user_history[int(u_id)]
                    if len(seen) == 0:
                        continue
                    valid_mask = seen < n_items
                    valid_seen = seen[valid_mask]
                    if len(valid_seen) > 0:
                        scores[i, valid_seen.to(device)] = -float("inf")
                topk_scores, topk_idx = torch.topk(scores, topk, dim=1)
            elif has_embed and dataset.item_num > 50000:
                # Dense embeddings with large catalog: chunk processing
                user_e = model.user_embedding(batch_users).to(device)
                n_items = item_embed.size(0)
                vals = torch.empty((batch_size, 0), device=device)
                idxs = torch.empty((batch_size, 0), device=device, dtype=torch.long)
                # Process items in chunks
                for start in range(0, n_items, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, n_items)
                    scores_chunk = torch.matmul(user_e, item_embed[start:end].t())
                    batch_users_list = batch_users.tolist()
                    # Mask seen items
                    for i, u_id in enumerate(batch_users_list):
                        seen = user_history[int(u_id)]
                        if len(seen) == 0:
                            continue
                        mask_in_chunk = (seen >= start) & (seen < end)
                        if mask_in_chunk.any():
                            local_indices = (seen[mask_in_chunk] - start).to(device)
                            scores_chunk[i, local_indices] = -float("inf")
                    v, i_idx = torch.topk(scores_chunk, k=min(topk, end - start), dim=1)
                    i_idx = i_idx + start
                    vals = torch.cat([vals, v], dim=1)
                    idxs = torch.cat([idxs, i_idx], dim=1)
                    vals, top_idx = torch.topk(vals, k=topk, dim=1)
                    idxs = torch.gather(idxs, 1, top_idx)
                    del scores_chunk, v, i_idx
                topk_scores = vals
                topk_idx = idxs
            else:
                # Fallback: use model's full_sort_predict or chunk-based prediction
                try:
                    scores = model.full_sort_predict(batch)
                except NotImplementedError:
                    scores = _predict_full_set(model, batch_users, dataset, device)
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
                if scores.size(0) != batch_size:
                    scores = scores.view(batch_size, -1)
                n_items = scores.size(1)
                batch_users_list = batch_users.tolist()
                # Mask seen items
                for i, u_id in enumerate(batch_users_list):
                    seen = user_history[int(u_id)]
                    if len(seen) == 0:
                        continue
                    valid_mask = seen < n_items
                    valid_seen = seen[valid_mask]
                    if len(valid_seen) > 0:
                        scores[i, valid_seen.to(device)] = -float("inf")
                topk_scores, topk_idx = torch.topk(scores, topk, dim=1)

            # Convert to numpy and store results
            uids = batch[user_field].cpu().numpy()
            topk_idx = topk_idx.cpu().numpy()
            topk_scores = topk_scores.cpu().numpy()
            for i, uid in enumerate(uids):
                user_recs[int(uid)] = [(int(topk_idx[i][j]), float(topk_scores[i][j])) for j in range(topk)]
            if 'scores' in locals():
                del scores
            del batch, topk_idx, topk_scores

    # Clean up LightGCN cached embeddings
    if is_lightgcn:
        if hasattr(model, 'restore_user_e'):
            model.restore_user_e = None
        if hasattr(model, 'restore_item_e'):
            model.restore_item_e = None
        del cached_user_emb, cached_item_emb

    return user_recs


def calc_item_novelty(item_id, pop_counter):
    """Calculate novelty for a single item.
    
    Novelty is inversely proportional to item popularity:
    Novelty(item) = 1 / log(1 + pop(item))
    
    Less popular items receive higher novelty scores.
    Never-seen items get maximum novelty (1.0).
    
    Args:
        item_id: the item identifier
        pop_counter: Counter mapping item_id -> popularity count
    
    Returns:
        float: novelty score in range [0, 1] approximately
    """
    pop = pop_counter.get(int(item_id), 0)
    if pop > 0:
        return 1.0 / np.log1p(pop)
    else:
        return 1.0  # never seen item = maximum novelty


def calc_item_unexpectedness(item_id, user_history, i_embs_norm, device):
    """Calculate unexpectedness for a single item relative to user history.
    
    Measures how different the candidate item is from what the user has seen.
    Unexpectedness = 1 - avg_cosine_similarity(item, user_history_items)
    
    Items dissimilar to user history get high unexpectedness.
    If user has no history, returns neutral value (0.5).
    
    Args:
        item_id: the candidate item identifier
        user_history: torch.LongTensor of item IDs user has interacted with
        i_embs_norm: normalized item embeddings tensor [num_items, embedding_dim]
        device: torch device (cpu or cuda)
    
    Returns:
        float: unexpectedness score in [0, 1]
    """
    if len(user_history) == 0:
        return 0.5
    
    valid_history = user_history[user_history < i_embs_norm.shape[0]]
    if len(valid_history) == 0:
        return 0.5
    
    valid_history = valid_history.to(device).long()
    if item_id >= i_embs_norm.shape[0]:
        return 0.5
    
    # Get normalized embeddings
    item_emb = i_embs_norm[item_id].unsqueeze(0)
    history_embs = i_embs_norm[valid_history]
    
    # Calculate average cosine similarity between item and user history
    sims = F.cosine_similarity(item_emb, history_embs, dim=1)
    avg_sim = sims.mean().item()
    
    # Normalize similarity from [-1,1] to [0,1]
    avg_sim = (avg_sim + 1.0) / 2.0
    avg_sim = max(0.0, min(1.0, avg_sim))
    
    return 1.0 - avg_sim


def normalize_scores(scores):
    """Normalize a list of scores to [0, 1] using min-max normalization.
    
    Scales all scores so the minimum becomes 0 and the maximum becomes 1.
    Handles edge case where all scores are identical (returns 0.5).
    
    Args:
        scores: list of numeric scores to normalize
    
    Returns:
        list: normalized scores in [0, 1]
    """
    if len(scores) == 0:
        return []
    scores = np.array(scores)
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s > 1e-8:
        return ((scores - min_s) / (max_s - min_s)).tolist()
    else:
        return [0.5] * len(scores)


def rerank_creativity_score(user_recs_with_scores, user_history, i_embs, pop_counter, 
                            num_candidates, topk, device):
    """Rerank recommendations using creativity score formula.
    
    For each user, takes top-K candidates from the original ranking and reorders
    them by a weighted combination of three factors:
    
    Creativity Score = (0.33 * relevance) + (0.33 * novelty) + (0.33 * unexpectedness)
    
    Where:
    - relevance: normalized original model score
    - novelty: 1/log(1+popularity)
    - unexpectedness: 1 - similarity_to_user_history
    
    Args:
        user_recs_with_scores: dict[uid] = [(item_id, score), ...] original recommendations
        user_history: dict[uid] = torch.LongTensor([item_ids]) training interactions
        i_embs: item embeddings tensor [num_items, embedding_dim]
        pop_counter: Counter mapping item_id -> popularity count
        num_candidates: K - number of top candidates to rerank (pool size)
        topk: N - number of items to return after reranking
        device: torch device (cpu or cuda)
    
    Returns:
        dict[uid] = [item_id, ...] final reranked list of N items per user
    """
    if i_embs is None:
        print("⚠️ Item embeddings not available, using only relevance for reranking")
        # Fallback: use only relevance
        user_recs_reranked = {}
        for uid, candidates_with_scores in user_recs_with_scores.items():
            C = candidates_with_scores[:num_candidates]
            # Sort by score descending
            C_sorted = sorted(C, key=lambda x: x[1], reverse=True)
            user_recs_reranked[uid] = [item_id for item_id, _ in C_sorted[:topk]]
        return user_recs_reranked
    
    i_embs_norm = F.normalize(i_embs, p=2, dim=1)
    user_recs_reranked = {}
    
    for uid, candidates_with_scores in tqdm(user_recs_with_scores.items(), 
                                             desc=f"Reranking creativity (K={num_candidates})", 
                                             leave=False):
        C = candidates_with_scores[:num_candidates]
        if len(C) < 2:
            user_recs_reranked[uid] = [item_id for item_id, _ in C[:topk]]
            continue
        
        history = user_history.get(int(uid), torch.LongTensor([]))
        
        # Extract item_ids and scores
        item_ids = [item_id for item_id, _ in C]
        relevance_scores = [score for _, score in C]
        
        # Calculate novelty for each item
        novelty_scores = [calc_item_novelty(iid, pop_counter) for iid in item_ids]
        
        # Calculate unexpectedness for each item
        unexpectedness_scores = [
            calc_item_unexpectedness(iid, history, i_embs_norm, device) 
            for iid in item_ids
        ]
        
        # Normalize all scores to [0, 1]
        relevance_norm = normalize_scores(relevance_scores)
        novelty_norm = normalize_scores(novelty_scores)
        unexpectedness_norm = normalize_scores(unexpectedness_scores)
        
        # Calculate creativity score for each item
        creativity_scores = []
        for i in range(len(item_ids)):
            cs = (WEIGHT_RELEVANCE * relevance_norm[i] + 
                  WEIGHT_NOVELTY * novelty_norm[i] + 
                  WEIGHT_UNEXPECTEDNESS * unexpectedness_norm[i])
            creativity_scores.append(cs)
        
        # Sort by creativity score descending
        sorted_indices = np.argsort(creativity_scores)[::-1]
        
        user_recs_reranked[uid] = [item_ids[i] for i in sorted_indices[:topk]]
    
    return user_recs_reranked


def serendipity_ge_binary(user_recs, ground_truth, pop_counter, topk):
    """Compute Gérard-Evangelista binary serendipity metric.
    
    Measures fraction of serendipitous hits: recommendations that are both
    relevant (in ground truth) AND unpopular (not in top-topk popular items).
    
    Serendipity = (serendipitous_hits) / topk, averaged over users
    
    Args:
        user_recs: dict[uid] = [item_ids] recommendations
        ground_truth: dict[uid] = set(item_ids) test set positives
        pop_counter: Counter mapping item_id -> popularity
        topk: number of items to evaluate in recommendations
    
    Returns:
        float: average serendipity score [0, 1]
    """
    if not user_recs:
        return 0.0
    pm_set = set([item for item, _ in pop_counter.most_common(topk)])
    total = 0.0
    users = 0
    for uid, recs in user_recs.items():
        if uid not in ground_truth:
            continue
        l_u = recs[:topk]
        t_u = ground_truth[uid]
        hits = [i for i in l_u if i in t_u]
        ser_items = [i for i in hits if i not in pm_set]
        total += len(ser_items) / float(topk)
        users += 1
    return total / users if users else 0.0


def calc_serendipity_and_unexpectedness_yan_gpu(user_recs, ground_truth, u_embs, i_embs, topk):
    """Compute serendipity and unexpectedness using Yan et al. similarity formula.
    
    Measures unexpectedness as 1 - avg_cosine_similarity(item, user_history).
    Serendipity combines unexpectedness with relevance (hits only).
    
    Serendipity_Yan = avg_unexpectedness_of_hits / topk
    Unexpectedness_Yan = avg_unexpectedness_of_all / topk
    
    Args:
        user_recs: dict[uid] = [item_ids] recommendations
        ground_truth: dict[uid] = set(item_ids) test set positives
        u_embs: user embeddings tensor [num_users, embedding_dim]
        i_embs: item embeddings tensor [num_items, embedding_dim]
        topk: number of items to evaluate
    
    Returns:
        tuple: (serendipity_score, unexpectedness_score) averaged over users
    """
    if not user_recs or u_embs is None or i_embs is None:
        return 0.0, 0.0
    total_ser = 0.0
    total_unexp = 0.0
    users = 0
    u_norm = F.normalize(u_embs, p=2, dim=1).cpu()
    i_norm = F.normalize(i_embs, p=2, dim=1).cpu()
    for uid, recs in user_recs.items():
        if uid not in ground_truth:
            continue
        if uid >= len(u_norm):
            continue
        t_u = ground_truth[uid]
        l_u = recs[:topk]
        score_ser_u = 0.0
        score_unexp_u = 0.0
        user_vec = u_norm[uid]
        for iid in l_u:
            if iid >= len(i_norm):
                continue
            item_vec = i_norm[iid]
            sim = torch.dot(user_vec, item_vec).item()
            sim = (sim + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
            sim = max(0.0, min(1.0, sim))
            unexp = 1.0 - sim
            score_unexp_u += unexp
            if iid in t_u:
                score_ser_u += unexp
        total_ser += score_ser_u / float(topk)
        total_unexp += score_unexp_u / float(topk)
        users += 1
    return (total_ser / users if users else 0.0), (total_unexp / users if users else 0.0)


def build_recbole_datastruct_from_reranked(user_recs, ground_truth, train_dataset, topks):
    """Convert reranked recommendations to RecBole DataStruct for metrics calculation.
    
    Args:
        user_recs: dict[uid] = [item_ids] reranked recommendations
        ground_truth: dict[uid] = set(item_ids) test set positives
        train_dataset: RecBole dataset object
        topks: list of K values to evaluate at
    
    Returns:
        DataStruct: RecBole structure ready for evaluation
    """
    if not user_recs:
        return None

    users = [uid for uid in user_recs.keys() if uid in ground_truth and len(ground_truth[uid]) > 0]
    if not users:
        return None

    max_k = max(topks)
    item_matrix = torch.zeros((len(users), max_k), dtype=torch.long)
    pos_matrix = torch.zeros((len(users), max_k), dtype=torch.int)
    pos_len_list = torch.zeros((len(users), 1), dtype=torch.int)

    for row_idx, uid in enumerate(users):
        recs = user_recs[uid][:max_k]
        if len(recs) < max_k:
            recs = recs + ([recs[-1]] * (max_k - len(recs))) if recs else [0] * max_k
        item_matrix[row_idx] = torch.tensor(recs, dtype=torch.long)
        gt = ground_truth[uid]
        pos_len_list[row_idx, 0] = len(gt)
        hit_row = [1 if iid in gt else 0 for iid in recs[:max_k]]
        pos_matrix[row_idx] = torch.tensor(hit_row, dtype=torch.int)

    data_struct = DataStruct()
    data_struct.set("rec.items", item_matrix)
    data_struct.set("rec.topk", torch.cat([pos_matrix, pos_len_list], dim=1))
    data_struct.set("data.num_items", train_dataset.item_num)
    data_struct.set("data.num_users", train_dataset.user_num)
    data_struct.set("data.count_items", train_dataset.item_counter)
    data_struct.set("data.count_users", train_dataset.user_counter)
    return data_struct


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def evaluate_checkpoint(model_path):
    """Evaluate a single checkpoint with all metrics.
    
    Process:
    1. Load checkpoint and dataset
    2. Run RecBole evaluation (baseline metrics)
    3. Generate recommendations with scores
    4. For each reranking pool size K:
       a. Apply creativity score reranking
       b. Evaluate reranked recommendations
       c. Calculate novelty and creativity scores
       d. Compute serendipity/unexpectedness metrics
    
    Args:
        model_path: path to checkpoint .pth file
    
    Returns:
        dict: results dictionary with all computed metrics
    """
    print(f"\nProcessing: {model_path}")
    gc.collect()    
    if USE_GPU:
        torch.cuda.empty_cache()
    try:        
        # =================================================================
        # STEP 1: Load checkpoint and configure
        # =================================================================
        config, model, dataset, train_data, _, test_data = load_data_and_model(model_file=model_path)
        # Update configuration for current evaluation settings
        config['use_gpu'] = USE_GPU
        config['device'] = DEVICE
        config['gpu_id'] = 0 if USE_GPU else ''
        config['eval_batch_size'] = EVAL_BATCH_SIZE
        config['topk'] = TOPKS
        if hasattr(test_data, 'config'):
            test_data.config['eval_batch_size'] = EVAL_BATCH_SIZE
            test_data.config['device'] = DEVICE
        object.__setattr__(test_data, 'batch_size', EVAL_BATCH_SIZE)
        if hasattr(test_data, 'step'):
            object.__setattr__(test_data, 'step', EVAL_BATCH_SIZE)
        model = model.to(DEVICE)
        trainer = Trainer(config, model)
        trainer.save_model = False
        
        # =================================================================
        # STEP 2: Evaluate baseline recommendations with RecBole
        # =================================================================
        print("   RecBole metrics (original)...")
        trainer.eval_collector.data_collect(train_data)
        # base_metrics contains: ndcg@k, recall@k, hit@k, averagepopularity@k, etc.
        base_metrics = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

        uid_field = dataset.uid_field
        iid_field = dataset.iid_field
        label_field = dataset.label_field if hasattr(dataset, 'label_field') else None
        pop_counter = build_pop_counter(train_data.dataset.inter_feat, iid_field)
        test_truth = build_ground_truth(test_data.dataset.inter_feat, uid_field, iid_field, label_field=label_field)
        u_embs, i_embs = get_vectors(model, train_data.dataset)
        user_history = build_user_history_cpu(train_data.dataset)

        object.__setattr__(test_data, 'batch_size', GEN_BATCH_SIZE)
        if hasattr(test_data, 'step'):
            object.__setattr__(test_data, 'step', GEN_BATCH_SIZE)
        if hasattr(test_data, 'config'):
            test_data.config['eval_batch_size'] = GEN_BATCH_SIZE
            test_data.config['device'] = DEVICE

        try:
            dataset_name = config['dataset']
        except KeyError:
            dataset_name = dataset.name if hasattr(dataset, 'name') else ''
        try:
            model_name = config['model']
        except KeyError:
            model_name = model.__class__.__name__

        result = {
            'Checkpoint': os.path.basename(model_path),
            'Dataset': dataset_name,
            'Model': model_name,
        }

        # =================================================================
        # STEP 4: Store baseline RecBole metrics
        # =================================================================
        # RecBole original metrics (baseline values)
        for key, val in base_metrics.items():
            result[key] = val

        # =================================================================
        # STEP 5: Calculate novelty and creativity scores for original recs
        # =================================================================
        # Novelty is extracted from RecBole's averagepopularity metric
        # Creativity Score = 0.33*NDCG + 0.33*Novelty + 0.33*Unexpectedness
        # Calculate novelty and creativity score for original recommendations
        for k in TOPKS:
            # Extract avgpop from RecBole metrics
            avgpop_key = f"averagepopularity@{k}"
            if avgpop_key in base_metrics:
                avgpop = base_metrics[avgpop_key]
                novelty = 1.0 / np.log1p(avgpop) if avgpop > 0 else 0.0
                result[f'Novelty@{k}'] = novelty
            
            # Extract NDCG from RecBole metrics
            ndcg_key = f"ndcg@{k}"
            if ndcg_key in base_metrics:
                rel = base_metrics[ndcg_key]
            else:
                rel = 0.0
            
            # Get unexpectedness from serendipity calc
            unexp_key = f'Unexpectedness_Original@{k}'
            if unexp_key in result:
                unexp = result[unexp_key]
            else:
                unexp = 0.0
            
            # Get novelty
            nov_key = f'Novelty@{k}'
            if nov_key in result:
                novelty = result[nov_key]
            else:
                novelty = 0.0
            
            # Calculate creativity score
            creativity_score = (WEIGHT_RELEVANCE * rel + 
                               WEIGHT_NOVELTY * novelty + 
                               WEIGHT_UNEXPECTEDNESS * unexp)
            result[f'Creativity_Score_Original@{k}'] = creativity_score

        # =================================================================
        # STEP 6: Generate recommendations with scores
        # =================================================================
        # Generate recommendations with scores for maximum candidate K
        max_candidates = max(CANDIDATE_KS)
        print(f"   Generating recommendations (K={max_candidates} candidates)...")
        user_recs_with_scores = generate_recommendations_with_scores_gpu(
            model, test_data, dataset, max_candidates, DEVICE, user_history
        )

        # Convert to format without score for original metrics
        user_recs_original = {uid: [iid for iid, _ in recs] for uid, recs in user_recs_with_scores.items()}

        # =================================================================
        # STEP 7: For each reranking pool size K
        # =================================================================
        # For each combination of candidate K
        for num_candidates in CANDIDATE_KS:
            print(f"   Applying creativity score reranking (K={num_candidates})...")
            
            # ============================================================
            # 7.1 Apply creativity score reranking
            # ============================================================
            # Apply reranking with creativity score
            user_recs_reranked = rerank_creativity_score(
                user_recs_with_scores,
                user_history,
                i_embs,
                pop_counter,
                num_candidates=num_candidates,
                topk=max(TOPKS),
                device=DEVICE
            )

            # ============================================================
            # 7.2 Evaluate reranked recommendations with RecBole
            # ============================================================
            # RecBole metrics computed on reranked ranking
            rerank_data_struct = build_recbole_datastruct_from_reranked(
                user_recs_reranked, test_truth, train_data.dataset, TOPKS
            )
            if rerank_data_struct is not None:
                rerank_config = {
                    'topk': TOPKS,
                    'metrics': [m.lower() for m in config['metrics']],
                    'metric_decimal_place': config['metric_decimal_place'] if 'metric_decimal_place' in config else 4,
                }
                evaluator = Evaluator(rerank_config)
                rerank_metrics = evaluator.evaluate(rerank_data_struct)
                for key, val in rerank_metrics.items():
                    result[f'{key}_reranked_K{num_candidates}'] = val
                
                # Store delta values (improvement/degradation vs original)
                # Delta for all RecBole metrics
                for key, val in base_metrics.items():
                    if key in rerank_metrics:
                        result[f'Delta_{key}_K{num_candidates}'] = rerank_metrics[key] - val
                
                # ============================================================
                # 7.3 Calculate novelty and creativity for reranked recs
                # ============================================================
                # Calculate novelty and creativity score for reranked recommendations
                for k in TOPKS:
                    # Extract avgpop from reranked metrics
                    avgpop_key = f"averagepopularity@{k}"
                    if avgpop_key in rerank_metrics:
                        avgpop = rerank_metrics[avgpop_key]
                        novelty_reranked = 1.0 / np.log1p(avgpop) if avgpop > 0 else 0.0
                        result[f'Novelty_Reranked@{k}_K{num_candidates}'] = novelty_reranked
                    
                    # Extract NDCG from reranked metrics
                    ndcg_key = f"ndcg@{k}"
                    if ndcg_key in rerank_metrics:
                        rel_reranked = rerank_metrics[ndcg_key]
                    else:
                        rel_reranked = 0.0
                    
                    # Get unexpectedness from reranked metrics
                    unexp_key = f'Unexpectedness_Reranked@{k}_K{num_candidates}'
                    if unexp_key in result:
                        unexp_reranked = result[unexp_key]
                    else:
                        unexp_reranked = 0.0
                    
                    # Get novelty from reranked metrics
                    nov_key = f'Novelty_Reranked@{k}_K{num_candidates}'
                    if nov_key in result:
                        novelty_reranked = result[nov_key]
                    else:
                        novelty_reranked = 0.0
                    
                    # Calculate creativity score for reranked
                    creativity_score_reranked = (WEIGHT_RELEVANCE * rel_reranked + 
                                                WEIGHT_NOVELTY * novelty_reranked + 
                                                WEIGHT_UNEXPECTEDNESS * unexp_reranked)
                    result[f'Creativity_Score_Reranked@{k}_K{num_candidates}'] = creativity_score_reranked

            # ============================================================
            # 7.4 Compute serendipity and unexpectedness metrics
            # ============================================================
            # Serendipity/unexpectedness metrics
            # - Serendipity_Ge: fraction of hits that are unpopular items
            # - Serendipity_Yan: average unexpectedness of hits using embeddings
            # - Unexpectedness_Yan: average unexpectedness of all recommendations
            for k in TOPKS:
                # Original metrics (only for K=max_candidates to avoid duplicates)
                # Computed once to establish baseline values
                if num_candidates == max(CANDIDATE_KS):
                    k_recs_orig = {u: r[:k] for u, r in user_recs_original.items()}
                    ser_ge_orig = serendipity_ge_binary(k_recs_orig, test_truth, pop_counter, k)
                    ser_yan_orig, unexp_orig = calc_serendipity_and_unexpectedness_yan_gpu(
                        k_recs_orig, test_truth, u_embs, i_embs, k
                    )
                    result[f'Serendipity_Ge_Original@{k}'] = ser_ge_orig
                    result[f'Serendipity_Yan_Original@{k}'] = ser_yan_orig
                    result[f'Unexpectedness_Original@{k}'] = unexp_orig

                # Reranked metrics
                k_recs_reranked = {u: r[:k] for u, r in user_recs_reranked.items()}
                ser_ge_reranked = serendipity_ge_binary(k_recs_reranked, test_truth, pop_counter, k)
                ser_yan_reranked, unexp_reranked = calc_serendipity_and_unexpectedness_yan_gpu(
                    k_recs_reranked, test_truth, u_embs, i_embs, k
                )
                
                result[f'Serendipity_Ge_Reranked@{k}_K{num_candidates}'] = ser_ge_reranked
                result[f'Serendipity_Yan_Reranked@{k}_K{num_candidates}'] = ser_yan_reranked
                result[f'Unexpectedness_Reranked@{k}_K{num_candidates}'] = unexp_reranked

                # Delta (only if we have original values)
                if f'Serendipity_Ge_Original@{k}' in result:
                    result[f'Delta_Serendipity_Ge@{k}_K{num_candidates}'] = ser_ge_reranked - result[f'Serendipity_Ge_Original@{k}']
                    result[f'Delta_Serendipity_Yan@{k}_K{num_candidates}'] = ser_yan_reranked - result[f'Serendipity_Yan_Original@{k}']
                    result[f'Delta_Unexpectedness@{k}_K{num_candidates}'] = unexp_reranked - result[f'Unexpectedness_Original@{k}']

        # =================================================================
        # Return complete evaluation results
        # =================================================================
        # result contains all computed metrics:
        # - RecBole baseline metrics (ndcg, recall, hit, etc.)
        # - Novelty@k calculated from averagepopularity
        # - Creativity_Score_Original@k combining 3 weighted factors
        # - For each candidate K:
        #   * Reranked RecBole metrics
        #   * Delta values vs original
        #   * Novelty and Creativity for reranked
        #   * Serendipity and Unexpectedness metrics
        return result
    except Exception as e:
        print(f"Error on {model_path}: {e}")
        traceback.print_exc()
        return None


def filter_checkpoints(ckpts):
    """Filter checkpoint list by TARGET_CKPTS if specified.
    
    If TARGET_CKPTS is empty, returns all checkpoints.
    Otherwise, returns only checkpoints in TARGET_CKPTS and prints missing ones.
    
    Args:
        ckpts: list of checkpoint file paths
    
    Returns:
        list: filtered checkpoint paths
    """
    if not TARGET_CKPTS:
        return ckpts
    filtered = [p for p in ckpts if os.path.basename(p) in TARGET_CKPTS]
    missing = sorted(list(TARGET_CKPTS - {os.path.basename(p) for p in ckpts}))
    if missing:
        print("Required checkpoints missing:", ", ".join(missing))
    return filtered


def main():
    """Main evaluation pipeline.
    
    Process:
    1. Setup logging to file and console
    2. Print configuration
    3. Discover checkpoints from CKPT_ROOT
    4. Filter by TARGET_CKPTS if specified
    5. For each checkpoint:
       - Evaluate and compute all metrics
       - Append results to CSV file
       - Clean up GPU memory
    6. Print summary table
    """
    # Clean up previous results
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    
    # Setup dual logging (console + file)
    tee = TeeLogger(LOG_FILE)
    sys.stdout = tee
    sys.stderr = tee
    
    try:
        # Print header with configuration
        print("GPU:", USE_GPU, "Device:", DEVICE)
        print("=" * 80)
        print(f"EVALUATION: CREATIVITY SCORE RERANKING")
        print(f"  Weights: relevance={WEIGHT_RELEVANCE}, novelty={WEIGHT_NOVELTY}, unexpectedness={WEIGHT_UNEXPECTEDNESS}")
        print(f"  Candidate K: {CANDIDATE_KS}")
        print(f"  Final Top-N: {TOPKS}")
        print("=" * 80)
        
        # Discover and filter checkpoints
        ckpts = glob.glob(os.path.join(CKPT_ROOT, "**", "*.pth"), recursive=True)
        ckpts = filter_checkpoints(ckpts)
        if not ckpts:
            print("No checkpoints found.")
            return
        
        # Process each checkpoint
        results = []
        header_written = False
        for i, ck in enumerate(ckpts, 1):
            print(f"[{i}/{len(ckpts)}]")
            res = evaluate_checkpoint(ck)
            if res:
                results.append(res)
                # Append result to CSV (write header only on first row)
                pd.DataFrame([res]).to_csv(RESULTS_FILE, mode='a', header=not header_written, index=False, float_format='%.6f')
                header_written = True
            # Clean up memory after each checkpoint
            gc.collect()
            if USE_GPU:
                torch.cuda.empty_cache()
        
        # Print summary table
        if results:
            df = pd.DataFrame(results)
            # Prioritize metadata columns first
            cols = [c for c in ['Checkpoint', 'Dataset', 'Model'] if c in df.columns]
            other_cols = [c for c in df.columns if c not in cols]
            cols += other_cols
            try:
                print(df[cols].to_markdown(index=False, floatfmt=".4f"))
            except Exception:
                print(df[cols])
    finally:
        # Restore stdout/stderr
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Configure multiprocessing for compatibility
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    
    # Run main evaluation pipeline
    main()
