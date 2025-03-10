"""Main entry point for running diversification experiments."""

from src.config.config_parser import get_config
from src.core.pipeline import run_diversification_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run the diversification experiment pipeline."""
    config = get_config()
    results = run_diversification_pipeline(config)
    logger.info(
        f"Experiment completed. Results saved to {results['csv_path']} "
        f"and plots saved to {results['results_folder']}"
    )


if __name__ == "__main__":
    main()
