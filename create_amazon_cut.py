import os

# Reduces intermittent segfaults related to BLAS/OpenMP (numpy/pandas)
# Set BEFORE importing pandas/numpy.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import faulthandler
faulthandler.enable()

from collections import Counter

# Configuration
SOURCE_PATH = './dataset/Amazon_Books/Amazon_Books.inter'
OUTPUT_DIR = './dataset/Amazon_Books_cut500'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'Amazon_Books_cut500.inter')

# Thresholds
USER_THRESHOLD = 500
ITEM_THRESHOLD = 5  # Keep 5 as in the analysis to remove too rare items

# Fixed encoding required (Amazon file is often not clean UTF-8)
FILE_ENCODING = 'ISO-8859-1'

def create_cut_dataset():
    print(f"Processing dataset...")
    print(f"Source: {SOURCE_PATH}")
    print(f"Target: {OUTPUT_FILE}")
    print(f"Thresholds: Users >= {USER_THRESHOLD}, Items >= {ITEM_THRESHOLD}")

    # 1. Load Data
    # Read the original header to reuse it in saving (RecBole needs it)
    with open(SOURCE_PATH, 'r', encoding=FILE_ENCODING, errors='replace') as f:
        header_line = f.readline()
    
    # Identify column names from header (e.g. user_id:token -> user_id:token)
    headers = header_line.strip().split('\t')
    try:
        user_col = next(h for h in headers if h.startswith('user_id'))
        item_col = next(h for h in headers if h.startswith('item_id'))
    except StopIteration:
        print("Error: Could not identify user_id or item_id columns.")
        return

    encoding = FILE_ENCODING
    print(f"Using encoding: {encoding}")

    # Calculate column indexes (faster/more robust than using pandas)
    try:
        user_idx = headers.index(user_col)
        item_idx = headers.index(item_col)
    except ValueError:
        print("Error: Could not resolve user/item column indexes from header")
        return
    expected_cols = len(headers)

    def _iter_interactions():
        """Yield (user, item) for each valid line in the file, without loading everything in RAM."""
        with open(SOURCE_PATH, 'r', encoding=encoding, errors='replace') as f:
            _ = f.readline()  # skip header
            for line in f:
                if not line:
                    continue
                # Don't fully strip to not lose possible final tabs; removes only newline
                raw = line.rstrip('\n')
                parts = raw.split('\t')
                if len(parts) < expected_cols:
                    continue
                yield parts[user_idx], parts[item_idx]

    def _iter_filtered_lines(valid_users=None, valid_items=None):
        """Yield original lines (without header) filtered by user/item set."""
        with open(SOURCE_PATH, 'r', encoding=encoding, errors='replace') as f:
            _ = f.readline()  # skip header
            for line in f:
                if not line:
                    continue
                raw = line.rstrip('\n')
                parts = raw.split('\t')
                if len(parts) < expected_cols:
                    continue
                u = parts[user_idx]
                i = parts[item_idx]
                if valid_users is not None and u not in valid_users:
                    continue
                if valid_items is not None and i not in valid_items:
                    continue
                if line.endswith('\n'):
                    yield line
                else:
                    yield line + '\n'

    # 1) Count users (pass 1)
    print("Counting users (stream)...")
    user_counts = Counter()
    original_len = 0
    for u, _ in _iter_interactions():
        original_len += 1
        user_counts[u] += 1

    print(f"Original interactions: {original_len:,}")

    print(f"Filtering users with < {USER_THRESHOLD} interactions...")
    valid_users = {u for u, c in user_counts.items() if c >= USER_THRESHOLD}
    print(f"Valid users: {len(valid_users):,}")

    # 2) Count items on valid users only (pass 2)
    print(f"Counting items with valid users only (stream)...")
    item_counts = Counter()
    interactions_after_user_cut = 0
    for u, i in _iter_interactions():
        if u not in valid_users:
            continue
        interactions_after_user_cut += 1
        item_counts[i] += 1

    print(f"Interactions after user cut: {interactions_after_user_cut:,}")
    print(f"Filtering items with < {ITEM_THRESHOLD} interactions (in the remaining data)...")
    valid_items = {i for i, c in item_counts.items() if c >= ITEM_THRESHOLD}
    print(f"Valid items: {len(valid_items):,}")

    # 3) Final write (pass 3)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(header_line)

    print(f"\nSaving to {OUTPUT_FILE} (stream)...")
    final_len = 0
    final_users = set()
    final_items = set()
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out:
        for line in _iter_filtered_lines(valid_users=valid_users, valid_items=valid_items):
            # Update stats without re-splitting too much: extract u/i once
            raw = line.rstrip('\n')
            parts = raw.split('\t')
            if len(parts) < expected_cols:
                continue
            u = parts[user_idx]
            i = parts[item_idx]
            final_users.add(u)
            final_items.add(i)
            out.write(line)
            final_len += 1

    print(f"Final interactions: {final_len:,}")

    # 4. Calculate Stats
    n_users = len(final_users)
    n_items = len(final_items)
    sparsity = 1 - (final_len / (n_users * n_items)) if n_users * n_items > 0 else 0

    print("\nFinal Statistics:")
    print(f"Users: {n_users}")
    print(f"Items: {n_items}")
    print(f"Interactions: {final_len}")
    print(f"Sparsity: {sparsity:.6f}")

    print("Done.")

if __name__ == "__main__":
    create_cut_dataset()