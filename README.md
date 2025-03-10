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
- Configurable via YAML and command-line arguments
- Automatic parameter sweep with early stopping based on NDCG drop
- Metrics tracking and visualization
- Tab-separated results export
- Dual-axis performance plots

## Setup

1. Install dependencies:
```bash
pip install numpy pandas matplotlib pyyaml sentence-transformers
```

2. Prepare your data in the following structure:
```
topk_data/amazon14/
├── target_item_mapping.csv
├── amz_14_final_samples.csv
├── amz-topk_iid_list.pkl
└── amz-topk_score.pkl

OR

topk_data/movielens/
├── target_item_mapping.csv (This is a mapping file from internal IDs to externals)
├── test_samples.csv        (user_id, pos_item, neg_items samples file)
├── ml-topk_iid_list.pkl    (internal top-k item IDs for each test user)
└── ml-topk_score.pkl       (scores of the items in the top-k list of each user)
```

This file names can be set in the config file.

## Configuration

The experiment can be configured through:
1. YAML configuration file (`config.yaml` or custom config files in `configs/`)
2. Command-line arguments (override YAML settings)

### YAML Configuration

There are two ways to configure the experiments:

1. **Single Algorithm Configuration** (`config.yaml`):
```yaml
# Data paths
data:
  base_path: "topk_data/movielens"
  item_mappings: "target_item_mapping.csv"
  test_samples: "test_samples.csv"
  topk_list: "ml-topk_iid_list.pkl"
  topk_scores: "ml-topk_score.pkl"
  movie_categories: "movie_categories.csv"  # Optional: only for category-based ILD

# Embedder settings
embedder:
  model_name: "all-MiniLM-L6-v2" # pick an embedding model
  device: "cuda"  # or "cpu"
  batch_size: 40960

# Experiment parameters
experiment:
  diversifier: "sy"  # Options: motley, mmr, bswap, clt, msd, swap, sy
  param_name: "threshold"   # Auto-set based on diversifier if not specified
  param_start: 1.0
  param_end: 0.0
  param_step: -0.05
  threshold_drop: 0.1   # Stop when NDCG drops by this percentage
  top_k: 10
  use_category_ild: true  # Optional: enable category-based ILD metric ( this is valid only if we have item_id to category mapping)

```

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

2. Use command-line arguments when:
   - Quick parameter overrides without editing config files
   - Running experiments in scripts/loops
   - CI/CD pipelines

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

1. Real-time logging of metrics for each parameter value
2. Tab-separated CSV file with metrics:
   - Parameter value
   - NDCG
   - NDCG drop percentage
   - ILD
   - Hit Rate
   - Recall
   - Precision

3. Visualization plot showing:
   - NDCG values on left y-axis (blue)
   - ILD values on right y-axis (orange)
   - Parameter values on x-axis

Results are saved in `results_{algorithm}/` directory with timestamp-based filenames.

## Example Usage

Basic run with default settings:
```bash
python main.py
```

Run MMR diversification with custom parameters:
```bash
python main.py --diversifier mmr --param_start 0.1 --param_end 0.9 --param_step 0.1 --threshold_drop 0.02
```

Run multiple algorithms sequentially (not working):
```bash
for alg in motley mmr bswap clt msd swap sy; do
    python main.py --diversifier $alg
done
```