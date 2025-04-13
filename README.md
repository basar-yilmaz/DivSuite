# **DivSuite**:  A Diversification Experiments Framework

This project implements various diversification algorithms for recommendation systems and provides tools to evaluate their performance by tracking NDCG and ILD metrics.

## Project Structure

```
diversification/
├── main.py                 # Main entry point
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── src/                  # Source code
│   ├── core/            # Core functionality
│   │   ├── algorithms/  # Diversification algorithms
│   │   ├── embedders/   # Text embedding models
│   │   └── pipeline.py  # Main pipeline
│   ├── data/            # Data loading and processing
│   ├── metrics/         # Evaluation metrics
│   ├── visualization/   # Plotting and visualization
│   ├── config/         # Configuration handling
│   └── utils/          # Helper utilities
├── tests/              # Test suite
└── configs/           # Configuration files
```

## Features

- Multiple diversification algorithms:
  - Motley (theta parameter)
  - MMR (lambda parameter)
  - BSwap (theta parameter)
  - CLT (lambda parameter)
  - MaxSum (lambda parameter)
  - Swap (theta parameter)
  - SY (lambda parameter)
  - GNE (lambda parameter, alpha parameter, imax integer)
  - GMC (lambda parameter)
- Configurable via YAML and command-line arguments
- Automatic parameter sweep with early stopping based on NDCG drop
- Metrics tracking and visualization
- Tab-separated results export
- Dual-axis performance plots

## Setup

The recommended way to set up the development environment is using the provided script:

1.  **Run the setup script:**
    ```bash
    source setup.sh
    ```
    This script performs the following steps:
    *   Checks if Python 3.10 or higher is installed.
    *   Installs `uv` (a fast Python package installer and resolver) if it's not found.
    *   Installs `pre-commit` (for code quality checks) if it's not found.
    *   Creates a virtual environment named `.venv` using `uv`.
    *   Installs all project dependencies (including development and optional extras) into the virtual environment using `uv sync`.
    *   Installs pre-commit hooks to ensure code quality before commits.
    *   **Activates the virtual environment.** You should see `(divsuite_env)` in your shell prompt.

2.  **Prepare your data:** (This step remains the same)
    Ensure your data is structured as described below in the `topk_data/` directory (e.g., `topk_data/movielens/` or `topk_data/amazon14/`). Update the `config.yaml` or use command-line arguments if your file names differ.
    ```
    topk_data/your_dataset_name/
    ├── target_item_mapping.csv # Maps internal item IDs to external IDs
    ├── test_samples.csv        # User interaction samples (user_id, pos_item, neg_items)
    ├── your-prefix_iid_list.pkl # Top-K item lists (internal IDs) per user
    ├── your-prefix_score.pkl    # Scores for the items in the top-K lists
    ├── your_categories.csv     # Category mapping file (necessary for Cat-ILD, otherwise unnecessary)
    └── can be expanded...

    ```
    *Note: File names like `target_item_mapping.csv`, `test_samples.csv`, `ml-topk_iid_list.pkl`, `ml-topk_score.pkl` are configurable in your YAML file or via CLI arguments.*

## Running Experiments

After running `source setup.sh`, the virtual environment (`divsuite_env`) will be active in your current shell session. You can then run the main script directly:

```bash
python main.py [arguments...]
```

Alternatively, if you open a new shell or deactivate the environment, you can use `uv run` which automatically executes commands within the project's virtual environment without needing to activate it manually first:

```bash
uv run python main.py [arguments...]
```

**Example:**

Run MMR diversification with custom parameters using the activated environment:
```bash
# Ensure (divsuite_env) is active (run 'source setup.sh' if not)
python main.py --diversifier mmr --param_start 0.1 --param_end 0.9 --param_step 0.1 --threshold_drop 0.02
```

Or using `uv run`:
```bash
uv run python main.py --diversifier mmr --param_start 0.1 --param_end 0.9 --param_step 0.1 --threshold_drop 0.02
```

## Configuration

The experiment can be configured through:
1. YAML configuration file (`config.yaml` or custom config files in `configs/`)
2. Command-line arguments (override YAML settings)

### YAML Configuration

The primary configuration is done via a YAML file (default: `config.yaml`). Here's a breakdown of the sections and parameters:

1.  **`data` Section:** Defines paths to your dataset files.
    ```yaml
    data:
      base_path: "topk_data/movielens"  # Base directory for dataset files
      item_mappings: "target_item_mapping.csv" # Maps internal item IDs to external IDs (e.g., product names, movie titles)
      test_samples: "test_samples.csv" # Contains test user interactions (user_id, positive_item_id, list_of_negative_item_ids)
      topk_list: "ml-topk_iid_list.pkl" # Pre-computed top-K item lists (internal IDs) for each test user
      topk_scores: "ml-topk_score.pkl" # Scores corresponding to the items in topk_list
      movie_categories: "movie_categories.csv" # Optional: Maps item IDs to categories (e.g., genres) for category-based ILD calculation
    ```
    *   **To use a new dataset:** Create a similar directory structure (e.g., `topk_data/my_new_dataset/`) with corresponding files and update the `base_path` and potentially other filenames in this section.

2.  **`embedder` Section:** Configures the sentence transformer model used for calculating item similarity (used in ILD).
    ```yaml
    embedder:
      model_name: "all-MiniLM-L6-v2" # Name of the Sentence Transformer model (from Hugging Face Hub)
      device: "cuda"                  # Device for embedding computation ("cuda" or "cpu")
      batch_size: 40960               # Batch size for embedding generation (adjust based on GPU memory)
    ```

3.  **`experiment` Section:** Controls the diversification process and evaluation.
    ```yaml
    experiment:
      diversifier: "sy"         # Algorithm to use: motley, mmr, bswap, clt, msd (MaxSum), swap, sy
      param_name: "threshold"   # Name of the primary parameter for the chosen diversifier (auto-detected if omitted, e.g., "lambda_" for MMR, "theta_" for Motley)
      param_start: 1.0          # Starting value for the parameter sweep
      param_end: 0.0            # Ending value for the parameter sweep
      param_step: -0.05         # Step size for the parameter sweep (can be negative for decreasing sweeps like SY)
      threshold_drop: 0.1       # Early stopping criterion: stop if NDCG drops by this fraction (e.g., 0.1 = 10%) compared to the initial (highest) NDCG
      top_k: 10                 # Size of the final diversified list to evaluate (e.g., NDCG@10)
      use_category_ild: true    # Optional (default: false): If true and `movie_categories` is provided, calculate ILD based on item categories instead of embeddings. Requires the `data.movie_categories` file.
      # For CLT algorithm specifically:
      # pick_strategy: "medoid" # or "highest_relevance" - How to pick representative items for clusters
    ```

4.  **`similarity` Section (Optional):** Allows using pre-computed item similarity scores instead of calculating them on-the-fly using the embedder. This can significantly speed up repeated experiments if the item set and embeddings are fixed.
    ```yaml
    similarity:
      use_similarity_scores: false  # Set to true to load similarities from the specified file
      similarity_scores_path: "/path/to/your/similarity_results.pkl" # Path to a .pkl file containing pre-computed similarities
    ```
    *   The `.pkl` file should typically contain a data structure (like a dictionary or a NumPy array) representing the similarity matrix between items.
    *   When `use_similarity_scores` is `true`, the `embedder` settings are ignored for ILD calculation.

### Algorithm Parameters

Each diversification algorithm has specific parameters:

- **Motley**: Uses `theta_` parameter (0.0 to 1.0)
  - Higher values prioritize diversity over relevance

- **MMR**: Uses `lambda_` parameter (0.0 to 1.0)
  - Higher values prioritize relevance over diversity

- **BSwap**: Uses `theta_` parameter (0.0 to 1.0)
  - Higher values increase diversity threshold for swaps

- **CLT**: Uses `lambda_` parameter (0.0 to 1.0)
  - Higher values prioritize relevance over diversity
  - Additional option: `pick_strategy` ("medoid" or "highest_relevance")

- **MaxSum (MSD)**: Uses `lambda_` parameter (0.0 to 1.0)
  - Higher values prioritize relevance over diversity

- **Swap**: Uses `theta_` parameter (0.0 to 1.0)
  - Higher values increase diversity threshold for swaps

- **SY**: Uses `threshold` parameter (1.0 to 0.0, decreasing)
  - Lower values increase diversity threshold

- **GNE**: Uses `lambda_` (0.0 to 1.0), `alpha` (0.0 to 1.0), and `imax` (integer > 0) parameters.
  - `lambda_`: Balances relevance (higher) and diversity (lower).
  - `alpha`: Controls randomness in candidate selection (0 = greedy, 1 = fully random within bounds).
  - `imax`: Number of GRASP iterations to run.

- **GMC**: Uses `lambda_` parameter (0.0 to 1.0).
  - Higher values prioritize relevance over diversity.

### Command-line Arguments

All configuration options can be overridden via command-line:

```bash
python main.py --config custom_config.yaml  # Use different config file
python main.py --diversifier mmr --param_start 0.2 --param_end 0.8  # Override specific parameters
```

Available arguments:
- Data paths: `--data_path`, `--item_mappings`, `--test_samples`, `--topk_list`, `--topk_scores`, `--movie_categories`
- Embedder: `--model_name`, `--device`, `--batch_size`
- Experiment: `--diversifier`, `--param_name`, `--param_start`, `--param_end`, `--param_step`, `--threshold_drop`, `--top_k`, `--use_category_ild`

### When to Use What

1. Use `config.yaml` when:
   - Running experiments with a single algorithm
   - Quick testing or parameter tuning
   - Need to enable/disable category-based ILD
   - Storing default dataset paths

2. Use command-line arguments when:
   - Quick parameter overrides without editing config files
   - Running experiments in scripts/loops (e.g., iterating through algorithms)
   - CI/CD pipelines
   - Temporarily pointing to a different dataset (`--data_path`)

## Module Overview

### Core (`src/core/`)
- `algorithms/`: Implementation of diversification algorithms
- `embedders/`: Text embedding models (HuggingFace, SentenceTransformers)
- `pipeline.py`: Main experiment pipeline orchestration

### Data (`src/data/`)
- `data_loader.py`: Data loading and preparation
- `data_utils.py`: Data processing utilities
- `category_utils.py`: Category/genre handling

### Metrics (`src/metrics/`)
- `metrics_handler.py`: Metrics computation and tracking
- `metrics_utils.py`: Metric calculation utilities

### Visualization (`src/visualization/`)
- `visualization.py`: Plotting and results visualization

### Config (`src/config/`)
- `config_parser.py`: Configuration parsing and validation

### Utils (`src/utils/`)
- `experiment_utils.py`: Experiment setup utilities
- `logger.py`: Logging configuration
- `utils.py`: General utilities

## Output

The experiment produces:

1. Real-time logging of metrics for each parameter value to the console.
2. A tab-separated values (TSV) file with detailed metrics for each parameter step, saved in the results directory. The columns typically include:
   - Parameter value (e.g., `lambda_`, `theta_`)
   - NDCG@k (Normalized Discounted Cumulative Gain)
   - ILD (Intra-List Diversity, either embedding-based or category-based)
   - Optionally other metrics like Hit Rate, Recall, Precision.
3. A visualization plot (`.png`) showing NDCG and ILD against the parameter values, saved in the results directory.

Results are saved in a timestamped directory structure like `results/{algorithm_name}_{timestamp}/`. For example: `results/mmr_20240726_103000/`.

## Example Usage (Revisited)

**Basic run (uses defaults from `config.yaml`):**
```bash
# Activate env first: source setup.sh
python main.py
# OR using uv:
uv run python main.py
```

**Run MMR diversification with custom parameters and specify the dataset:**
```bash
# Activate env first: source setup.sh
python main.py --data_path topk_data/amazon14 --diversifier mmr --param_start 0.1 --param_end 0.9 --param_step 0.1
# OR using uv:
uv run python main.py --data_path topk_data/amazon14 --diversifier mmr --param_start 0.1 --param_end 0.9 --param_step 0.1
```

**Run multiple algorithms sequentially using a loop (example for bash/zsh):**
```bash
# Activate env first: source setup.sh
for alg in motley mmr bswap clt msd swap sy; do
    echo "Running experiment for: $alg"
    python main.py --diversifier $alg --data_path topk_data/movielens # Add other params as needed
    echo "--------------------------------------"
done

# OR using uv run inside the loop:
for alg in motley mmr bswap clt msd swap sy; do
    echo "Running experiment for: $alg"
    uv run python main.py --diversifier $alg --data_path topk_data/movielens # Add other params as needed
    echo "--------------------------------------"
done
```
