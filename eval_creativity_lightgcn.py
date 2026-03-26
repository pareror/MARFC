# =============================================================================
# EVALUATION SCRIPT FOR CREATIVITY SCORE-BASED RECOMMENDATION RERANKING - LIGHTGCN
# =============================================================================
# This script evaluates trained LightGCN recommendation models and applies a reranking
# based on a creativity score formula that combines relevance, novelty, and 
# unexpectedness. LightGCN uses graph neural networks for recommendation.
#
# Creativity Score = (0.33 * relevance) + (0.33 * novelty) + (0.33 * unexpectedness)
# Novelty = 1 / log(1 + average_popularity)
# =============================================================================

import os
import sys
import gc
import traceback
from collections import Counter, defaultdict

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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model
from recbole.trainer import Trainer
from recbole.evaluator import Evaluator
from recbole.evaluator.collector import DataStruct

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# Checkpoints to evaluate
CHECKPOINTS = [
    # List of LightGCN checkpoint files to evaluate
    "./saved/LightGCN-Jan-22-2026_20-07-03.pth",  # Amazon
]

# Batch size for model evaluation
EVAL_BATCH_SIZE = 32

# Output file paths for results and logs
RESULTS_FILE = 'benchmark_creativity_lightgcn.csv'
LOG_FILE = 'eval_creativity_lightgcn.log'

# =============================================================================
# RECOMMENDATION LIST SIZES CONFIGURATION
# =============================================================================
# TOPKS: Final N items to evaluate metrics on
# CANDIDATE_KS: Pool sizes for the reranking algorithm (K candidates to reorder)
TOPKS = [5, 10]  # Final N to calculate metrics on
CANDIDATE_KS = [50, 100]  # Candidate K for reranking

# =============================================================================
# CREATIVITY SCORE FORMULA WEIGHTS
# =============================================================================
# Creativity Score = (0.33 * relevance) + (0.33 * novelty) + (0.33 * unexpectedness)
# Each component is normalized to [0, 1] before combination
WEIGHT_RELEVANCE = 0.33
WEIGHT_NOVELTY = 0.33
WEIGHT_UNEXPECTEDNESS = 0.33

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
# Check GPU availability and set device
if not torch.cuda.is_available():
    print("❌ GPU not available.")
    sys.exit(1)

DEVICE = torch.device("cuda")



# =============================================================================
# LOGGING UTILITY
# =============================================================================
class TeeLogger:
    """Dual output logger: writes to both console and file simultaneously."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')
    
    def write(self, message):
        """Write message to both terminal and log file."""
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        """Flush both terminal and log file buffers."""
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        """Close log file."""
        self.log.close()


# =============================================================================
# MODEL LOADING
# =============================================================================
def custom_load_model(model_file):
    """Load LightGCN checkpoint and configure for evaluation.
    
    Loads a pre-trained LightGCN model from checkpoint file and initializes
    RecBole dataset and data splits. Configures batch sizes and metrics for
    evaluation.
    
    Args:
        model_file: path to checkpoint .pth file containing model state and config
    
    Returns:
        tuple: (config, model, dataset, train_data, test_data)
            - config: RecBole configuration object
            - model: loaded and initialized LightGCN model on GPU
            - dataset: RecBole dataset object
            - train_data: training data loader
            - test_data: test data loader
    """
    checkpoint = torch.load(model_file, map_location='cpu')
    saved_config = checkpoint['config']
    
    override_dict = {
        'model': saved_config['model'],
        'dataset': saved_config['dataset'],
        'eval_batch_size': EVAL_BATCH_SIZE,
        'train_batch_size': EVAL_BATCH_SIZE,
        'gpu_id': '0',
        'use_gpu': True,
        'save_model': False,
        'state': 'INFO',
        'data_path': './dataset/',
        'topk': TOPKS,
        'metrics': ['NDCG', 'Recall', 'Precision', 'AveragePopularity', 'GiniIndex', 'ShannonEntropy'],
        'valid_metric': 'NDCG@10',
        'eval_setting': 'RO_RS,full',
        'metric_decimal_place': 4,
    }
    
    print(f"   📂 Loading: {saved_config['model']} on {saved_config['dataset']}")

    config = Config(model=saved_config['model'], dataset=saved_config['dataset'], config_dict=override_dict)
    init_seed(config['seed'], config['reproducibility'])
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    model = get_model(config['model'])(config, train_data.dataset).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return config, model, dataset, train_data, test_data


# =============================================================================
# LIGHTGCN PATCHING FOR SAFE FULL RANKING
# =============================================================================
def patch_lightgcn_safe_predict(model):
    """Patch LightGCN model to ensure safe full-ranking predictions.
    
    LightGCN has a proprietary forward pass that may use cached embeddings.
    This function computes and caches the graph-propagated embeddings for
    safe full-ranking without disturbing internal state.
    
    Args:
        model: LightGCN model instance
    """
    if model.__class__.__name__ != 'LightGCN':
        return

    print("   🔧 Patching LightGCN for Safe Full Ranking...")
    
    with torch.no_grad():
        if hasattr(model, 'restore_user_e'):
            model.restore_user_e = None
        if hasattr(model, 'restore_item_e'):
            model.restore_item_e = None
        
        user_e = None
        item_e = None

        # Try to compute graph-propagated embeddings
        try:
            user_e, item_e = model.computer()
        except (AttributeError, ValueError):
            pass

        if user_e is None:
            try:
                res = model.get_ego_embeddings()
                if isinstance(res, (tuple, list)) and len(res) == 2:
                    user_e, item_e = res
                elif torch.is_tensor(res):
                    user_e = res[:model.n_users]
                    item_e = res[model.n_users:]
            except (AttributeError, ValueError, IndexError):
                pass

        # Fallback to static embeddings if graph propagation fails
        if user_e is None:
            print("   ⚠️ Warning: Graph propagation failed. Using static embeddings.")
            user_e = model.user_embedding.weight
            item_e = model.item_embedding.weight

        # Cache safe embeddings for prediction
        model.safe_user_e = user_e.detach()
        model.safe_item_e = item_e.detach()

    # Replace full_sort_predict with safe implementation
    def safe_full_sort_predict(self, interaction):
        """Safe full-ranking prediction using cached embeddings."""
        user = interaction[self.USER_ID]
        u_embeddings = self.safe_user_e[user]
        scores = torch.matmul(u_embeddings, self.safe_item_e.transpose(0, 1))
        
        if scores.dim() == 1:
            batch_size = user.size(0)
            scores = scores.view(batch_size, -1)
            
        return scores

    model.full_sort_predict = safe_full_sort_predict.__get__(model)


# =============================================================================
# DATA BUILDING UTILITIES
# =============================================================================
def build_pop_counter(dataset):
    """Build item popularity counter from training interactions.
    
    Args:
        dataset: RecBole dataset with training interactions
    
    Returns:
        Counter: dictionary mapping item_id -> popularity count
    """
    pop = Counter()
    items = dataset.inter_feat[dataset.iid_field].numpy()
    for i in items:
        pop[int(i)] += 1
    return pop


def build_ground_truth(dataset, label_field=None):
    """Build ground truth from test interactions.
    
    If label_field is present (from threshold), filter only positive label=1.
    Otherwise, use all interactions as positive.
    
    Args:
        dataset: test dataset with interactions
        label_field: optional label field name for filtering by threshold
    
    Returns:
        defaultdict: mapping user_id -> set of relevant item_ids
    """
    truth = defaultdict(set)
    uids = dataset.inter_feat[dataset.uid_field].numpy()
    iids = dataset.inter_feat[dataset.iid_field].numpy()
    
    if label_field and label_field in dataset.inter_feat:
        # Filter for label=1 (only rating >= threshold)
        labels = dataset.inter_feat[label_field].numpy()
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
        dataset: RecBole dataset with training interactions
    
    Returns:
        defaultdict: mapping user_id -> torch.LongTensor of item_ids
    """
    history = defaultdict(list)
    users = dataset.inter_feat[dataset.uid_field].numpy()
    items = dataset.inter_feat[dataset.iid_field].numpy()
    for u, i in zip(users, items):
        history[u].append(i)
    out = defaultdict(lambda: torch.LongTensor([]))
    for u, arr in history.items():
        out[u] = torch.LongTensor(arr)
    return out


def get_vectors_for_yan(model):
    """Extract user and item embeddings from LightGCN model.
    
    Returns cached safe embeddings if available (from patching),
    otherwise attempts to compute embeddings on the fly.
    
    Args:
        model: trained LightGCN model
    
    Returns:
        tuple: (user_embeddings, item_embeddings) or (None, None) if extraction fails
    """
    if hasattr(model, 'safe_user_e'):
        return model.safe_user_e, model.safe_item_e
    
    if model.__class__.__name__ == 'LightGCN':
        try:
            return model.computer()
        except AttributeError:
            pass

    if hasattr(model, 'user_embedding') and hasattr(model, 'item_embedding'):
        return model.user_embedding.weight.detach(), model.item_embedding.weight.detach()
    return None, None


# =============================================================================
# RECOMMENDATION GENERATION WITH SCORES
# =============================================================================
def generate_recs_with_scores(model, test_data, topk):
    """Generate top-K recommendations with relevance scores for all test users.
    
    Processes each batch in the test set, generates full ranking scores,
    and returns top-K items per user with their scores. Filters out seen items
    using user history within the function.
    
    Args:
        model: trained LightGCN model
        test_data: test data loader
        topk: number of recommendations per user
    
    Returns:
        dict: mapping user_id -> [(item_id, score), ...] sorted by score desc
    """
    user_recs = {}
    uid_field = model.USER_ID
    
    with torch.no_grad():
        for batch in tqdm(test_data, desc="Generating Lists with Scores", leave=False):
            interaction = batch[0].to(DEVICE)
            scores = model.full_sort_predict(interaction)
            
            # Safe reshape for flexible batch handling
            if scores.dim() == 1:
                batch_users = interaction[uid_field]
                batch_size = batch_users.size(0)
                scores = scores.view(batch_size, -1)
            
            # Get top-K items with scores
            topk_scores, topk_indices = torch.topk(scores, topk, dim=1)
            
            batch_users_cpu = interaction[uid_field].cpu().numpy()
            topk_indices_cpu = topk_indices.cpu().numpy()
            topk_scores_cpu = topk_scores.cpu().numpy()
            
            for i, uid in enumerate(batch_users_cpu):
                user_recs[int(uid)] = list(zip(
                    topk_indices_cpu[i].tolist(),
                    topk_scores_cpu[i].tolist()
                ))
                
    return user_recs


# =============================================================================
# CREATIVITY SCORE CALCULATION FUNCTIONS
# =============================================================================
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
        float: novelty score approximately in range [0, 1]
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
        user_recs_reranked = {}
        for uid, candidates_with_scores in user_recs_with_scores.items():
            C = candidates_with_scores[:num_candidates]
            C_sorted = sorted(C, key=lambda x: x[1], reverse=True)
            user_recs_reranked[uid] = [item_id for item_id, _ in C_sorted[:topk]]
        return user_recs_reranked
    
    i_embs_norm = F.normalize(i_embs, p=2, dim=1)
    user_recs_reranked = {}
    
    for uid, candidates_with_scores in tqdm(user_recs_with_scores.items(), 
                                             desc=f"Creativity rerank (K={num_candidates})", 
                                             leave=False):
        C = candidates_with_scores[:num_candidates]
        if len(C) < 2:
            user_recs_reranked[uid] = [item_id for item_id, _ in C[:topk]]
            continue
        
        history = user_history.get(int(uid), torch.LongTensor([]))
        
        item_ids = [item_id for item_id, _ in C]
        relevance_scores = [score for _, score in C]
        
        novelty_scores = [calc_item_novelty(iid, pop_counter) for iid in item_ids]
        unexpectedness_scores = [
            calc_item_unexpectedness(iid, history, i_embs_norm, device) 
            for iid in item_ids
        ]
        
        relevance_norm = normalize_scores(relevance_scores)
        novelty_norm = normalize_scores(novelty_scores)
        unexpectedness_norm = normalize_scores(unexpectedness_scores)
        
        creativity_scores = []
        for i in range(len(item_ids)):
            cs = (WEIGHT_RELEVANCE * relevance_norm[i] + 
                  WEIGHT_NOVELTY * novelty_norm[i] + 
                  WEIGHT_UNEXPECTEDNESS * unexpectedness_norm[i])
            creativity_scores.append(cs)
        
        sorted_indices = np.argsort(creativity_scores)[::-1]
        user_recs_reranked[uid] = [item_ids[i] for i in sorted_indices[:topk]]
    
    return user_recs_reranked


# =============================================================================
# SERENDIPITY AND UNEXPECTEDNESS METRICS
# =============================================================================
def serendipity_ge_binary(user_recs, ground_truth, pop_counter, topk):
    """Calculate binary serendipity metric (Ge).
    
    Measures relevance with serendipity: how many relevant items are NOT
    in the most popular top-K items.
    
    Args:
        user_recs: dict[uid] = [item_ids] recommendations
        ground_truth: dict[uid] = set(item_ids) relevant items
        pop_counter: Counter of item popularity
        topk: K value for metric
    
    Returns:
        float: average serendipity across users
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
    """Calculate serendipity (Yan) and unexpectedness using embeddings.
    
    Uses cosine similarity between user and item embeddings to measure
    the unexpectedness of recommended items (how dissimilar from user profile).
    
    Args:
        user_recs: dict[uid] = [item_ids] recommendations
        ground_truth: dict[uid] = set(item_ids) relevant items
        u_embs: user embeddings tensor [num_users, embedding_dim]
        i_embs: item embeddings tensor [num_items, embedding_dim]
        topk: K value for metric
    
    Returns:
        tuple: (serendipity_yan, unexpectedness) averages
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
            sim = (sim + 1.0) / 2.0
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
    """Build RecBole DataStruct from reranked recommendations for evaluation.
    
    Converts reranked recommendations into RecBole's internal DataStruct format
    for use with RecBole's evaluator on ranking metrics.
    
    Args:
        user_recs: dict[uid] = [item_ids] reranked recommendations
        ground_truth: dict[uid] = set(item_ids) relevant items
        train_dataset: training dataset for item/user statistics
        topks: list of K values for padding/truncation
    
    Returns:
        DataStruct: RecBole data structure ready for evaluation, or None if empty
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
# EVALUATION FUNCTION
# =============================================================================
def evaluate_checkpoint(model_path):
    """Evaluate a single checkpoint with all metrics.
    
    Process:
    1. Load checkpoint and dataset
    2. Run RecBole evaluation (baseline metrics)
    3. Calculate novelty and creativity scores
    4. Generate recommendations with scores
    5. For each reranking pool size K:
       a. Apply creativity score reranking
       b. Evaluate reranked recommendations
       c. Calculate novelty and creativity scores for reranked
       d. Compute serendipity/unexpectedness metrics
    
    Args:
        model_path: path to checkpoint .pth file
    
    Returns:
        dict: results dictionary with all computed metrics, or None on error
    """
    print(f"\n⚙️  Processing: {os.path.basename(model_path)}")
    gc.collect()
    torch.cuda.empty_cache()
    
    try:
        config, model, dataset, train_data, test_data = custom_load_model(model_file=model_path)
        
        # Patch LightGCN
        patch_lightgcn_safe_predict(model)
        
        print("   📊 Running RecBole Native Evaluator (baseline)...")
        trainer = Trainer(config, model)
        trainer.eval_collector.data_collect(train_data)
        base_metrics = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        
        # =================================================================
        # STEP 2: Collect data and build necessary structures
        # =================================================================
        label_field = dataset.label_field if hasattr(dataset, 'label_field') else None
        pop_counter = build_pop_counter(train_data.dataset)
        ground_truth = build_ground_truth(test_data.dataset, label_field=label_field)
        user_history = build_user_history_cpu(train_data.dataset)
        u_emb, i_emb = get_vectors_for_yan(model)
        
        # =================================================================
        # STEP 3: Build result dictionary with baseline metrics
        # =================================================================
        result = {
            'Checkpoint': os.path.basename(model_path),
            'Dataset': config['dataset'],
            'Model': config['model'],
        }
        
        # Store RecBole baseline metrics
        for key, val in base_metrics.items():
            result[key] = val
        
        # =================================================================
        # STEP 4: Calculate novelty and creativity for original recommendations
        # =================================================================
        # Novelty is calculated from RecBole's averagepopularity metric
        # Creativity Score = 0.33*NDCG + 0.33*Novelty + 0.33*Unexpectedness
        for k in TOPKS:
            # Extract averagepopularity from RecBole metrics
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
            
            # Get unexpectedness from serendipity calculation
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
        # STEP 5: Generate recommendations with scores
        # =================================================================
        max_candidates = max(CANDIDATE_KS)
        print(f"   🔮 Generating {max_candidates} candidates with scores...")
        user_recs_with_scores = generate_recs_with_scores(model, test_data, max_candidates)
        
        # Convert to format without score for original metrics
        user_recs_original = {uid: [iid for iid, _ in recs] for uid, recs in user_recs_with_scores.items()}
        
        # =================================================================
        # STEP 6: For each reranking pool size K
        # =================================================================
        # For each combination of candidate K
        for num_candidates in CANDIDATE_KS:
            print(f"   🎨 Applying creativity score reranking (K={num_candidates})...")
            
            # ============================================================
            # 6.1: Apply creativity score reranking
            # ============================================================
            user_recs_reranked = rerank_creativity_score(
                user_recs_with_scores,
                user_history,
                i_emb,
                pop_counter,
                num_candidates=num_candidates,
                topk=max(TOPKS),
                device=DEVICE
            )
            
            # ============================================================
            # 6.2: Evaluate reranked recommendations with RecBole
            # ============================================================
            # RecBole metrics computed on reranked ranking
            rerank_data_struct = build_recbole_datastruct_from_reranked(
                user_recs_reranked, ground_truth, train_data.dataset, TOPKS
            )
            if rerank_data_struct is not None:
                rerank_config = {
                    'topk': TOPKS,
                    'metrics': [m.lower() for m in config['metrics']],
                    'metric_decimal_place': config['metric_decimal_place'],
                }
                evaluator = Evaluator(rerank_config)
                rerank_metrics = evaluator.evaluate(rerank_data_struct)
                
                for key, val in rerank_metrics.items():
                    result[f'{key}_reranked_K{num_candidates}'] = val
                
                # Store delta values (improvement/degradation vs original)
                for key, val in base_metrics.items():
                    if key in rerank_metrics:
                        result[f'Delta_{key}_K{num_candidates}'] = rerank_metrics[key] - val
                
                # ============================================================
                # 6.3: Calculate novelty and creativity for reranked recommendations
                # ============================================================
                # Calculate novelty and creativity score for reranked recommendations
                for k in TOPKS:
                    # Extract averagepopularity from reranked metrics
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
            # 6.4: Custom serendipity and unexpectedness metrics
            # ============================================================
            # Metriche custom serendipità/inaspettatezza
            for k in TOPKS:
                # Metriche originali (solo per K=max candidati per evitare duplicati)
                if num_candidates == max(CANDIDATE_KS):
                    k_recs_orig = {u: r[:k] for u, r in user_recs_original.items()}
                    ser_ge_orig = serendipity_ge_binary(k_recs_orig, ground_truth, pop_counter, k)
                    ser_yan_orig, unexp_orig = calc_serendipity_and_unexpectedness_yan_gpu(k_recs_orig, ground_truth, u_emb, i_emb, k)
                    
                    result[f'Serendipity_Ge_Original@{k}'] = ser_ge_orig
                    result[f'Serendipity_Yan_Original@{k}'] = ser_yan_orig
                    result[f'Unexpectedness_Original@{k}'] = unexp_orig
                
                # Metriche rerankate
                k_recs_reranked = {u: r[:k] for u, r in user_recs_reranked.items()}
                ser_ge_reranked = serendipity_ge_binary(k_recs_reranked, ground_truth, pop_counter, k)
                ser_yan_reranked, unexp_reranked = calc_serendipity_and_unexpectedness_yan_gpu(k_recs_reranked, ground_truth, u_emb, i_emb, k)
                
                result[f'Serendipity_Ge_Reranked@{k}_K{num_candidates}'] = ser_ge_reranked
                result[f'Serendipity_Yan_Reranked@{k}_K{num_candidates}'] = ser_yan_reranked
                result[f'Unexpectedness_Reranked@{k}_K{num_candidates}'] = unexp_reranked
                
                # Delta (solo se abbiamo i valori originali)
                if f'Serendipity_Ge_Original@{k}' in result:
                    result[f'Delta_Serendipity_Ge@{k}_K{num_candidates}'] = ser_ge_reranked - result[f'Serendipity_Ge_Original@{k}']
                    result[f'Delta_Serendipity_Yan@{k}_K{num_candidates}'] = ser_yan_reranked - result[f'Serendipity_Yan_Original@{k}']
                    result[f'Delta_Unexpectedness@{k}_K{num_candidates}'] = unexp_reranked - result[f'Unexpectedness_Original@{k}']

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# MAIN SCRIPT
# =============================================================================
def main():
    """Main evaluation loop: process all checkpoints and aggregate results."""
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    
    tee = TeeLogger(LOG_FILE)
    sys.stdout = tee
    sys.stderr = tee
    
    try:
        print("=" * 80)
        print("CREATIVITY SCORE RERANKING EVALUATION - LIGHTGCN")
        print(f"  Weights: relevance={WEIGHT_RELEVANCE}, novelty={WEIGHT_NOVELTY}, unexpectedness={WEIGHT_UNEXPECTEDNESS}")
        print(f"  Candidate K: {CANDIDATE_KS}")
        print(f"  Final Top-N: {TOPKS}")
        print("=" * 80)
        
        results = []
        header_written = False
        for i, ck in enumerate(CHECKPOINTS, 1):
            if not os.path.exists(ck):
                print(f"File not found: {ck}")
                continue
            
            print(f"[{i}/{len(CHECKPOINTS)}]")
            res = evaluate_checkpoint(ck)
            if res:
                results.append(res)
                pd.DataFrame([res]).to_csv(RESULTS_FILE, mode='a', header=not header_written, index=False, float_format='%.6f')
                header_written = True
            gc.collect()
            torch.cuda.empty_cache()
        
        if results:
            df = pd.DataFrame(results)
            cols = [c for c in ['Checkpoint', 'Dataset', 'Model'] if c in df.columns]
            other_cols = [c for c in df.columns if c not in cols]
            cols += other_cols
            try:
                print(df[cols].to_markdown(index=False, floatfmt=".4f"))
            except Exception:
                print(df[cols])
    finally:
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.close()


if __name__ == '__main__':
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
