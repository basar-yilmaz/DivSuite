# Diversification Experiments

This project implements various diversification algorithms for recommendation systems and provides tools to evaluate their performance by tracking NDCG and ILD metrics.

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
```

## Configuration

The experiment can be configured through:
1. YAML configuration file (`config.yaml`)
2. Command-line arguments (override YAML settings)

### YAML Configuration

```yaml
# Data paths
data:
  base_path: "topk_data/amazon14"
  item_mappings: "target_item_mapping.csv"
  test_samples: "amz_14_final_samples.csv"
  topk_list: "amz-topk_iid_list.pkl"
  topk_scores: "amz-topk_score.pkl"

# Embedder settings
embedder:
  model_name: "all-MiniLM-L6-v2"
  device: "cuda"
  batch_size: 40960

# Experiment parameters
experiment:
  diversifier: "motley"  # Options: motley, mmr, bswap, clt, msd, swap, sy
  param_name: "theta_"   # Auto-set based on diversifier if not specified
  param_start: 0.0
  param_end: 1.0
  param_step: 0.05
  threshold_drop: 0.01   # Stop when NDCG drops by this percentage
  top_k: 10
```

### Command-line Arguments

All configuration options can be overridden via command-line:

```bash
python main.py --config custom_config.yaml  # Use different config file
python main.py --diversifier mmr --param_start 0.2 --param_end 0.8  # Override specific parameters
```

Available arguments:
- Data paths: `--data_path`, `--item_mappings`, `--test_samples`, `--topk_list`, `--topk_scores`
- Embedder: `--model_name`, `--device`, `--batch_size`
- Experiment: `--diversifier`, `--param_name`, `--param_start`, `--param_end`, `--param_step`, `--threshold_drop`, `--top_k`

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

Run multiple algorithms sequentially:
```bash
for alg in motley mmr bswap clt msd swap sy; do
    python main.py --diversifier $alg
done
```