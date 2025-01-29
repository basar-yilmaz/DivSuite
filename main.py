from algorithms.swap import SwapDiversifier
from algorithms.mmr import MMRDiversifier
from algorithms.clt import CLTDiversifier
from algorithms.sy import SYDiversifier
from algorithms.bswap import BSwapDiversifier
from algorithms.motley import MotleyDiversifier
from algorithms.msd import MaxSumDiversifier
import numpy as np

from logger import get_logger

logger = get_logger(__name__, level="INFO")


items = np.array(
    [
        ["1", "Inception", 0.95],
        ["2", "The Matrix", 0.82],
        ["3", "The Godfather", 0.80],
        ["4", "Toy Story", 0.68],
        ["5", "Jurassic Park", 0.65],
        ["6", "The Shawshank Redemption", 0.57],
        ["7", "The Dark Knight", 0.55],
        ["8", "Pulp Fiction", 0.50],
        ["9", "The Lion King", 0.45],
        ["10", "The Lord of the Rings", 0.40],
    ],
    dtype=object,
)

# Swap Diversifier
logger.info("Starting Swap Diversification")
swapper = SwapDiversifier(model_name="all-MiniLM-L6-v2", device="cuda")
result = swapper.diversify(items, top_k=3, lambda_=0.5)
logger.info(f"Swap Diversification completed. Results:\n{result}")

# MMR Diversifier
logger.info("Starting MMR Diversification")
mmr = MMRDiversifier(model_name="all-MiniLM-L6-v2", device="cuda")
result = mmr.diversify(items, top_k=3, lambda_=0.5)
logger.info(f"MMR Diversification completed. Results:\n{result}")

# CLT Diversifier
logger.info("Starting CLT Diversification")
clt = CLTDiversifier(model_name="all-MiniLM-L6-v2", device="cuda")
result = clt.diversify(items, top_k=3, pick_strategy="medoid")
logger.info(f"CLT Diversification completed. Results:\n{result}")

# SY Diversifier
logger.info("Starting SY Diversification")
sy_div = SYDiversifier(model_name="all-MiniLM-L6-v2")
diverse_items = sy_div.diversify(items, top_k=3, threshold=0.5)
logger.info(f"SY Diversification completed. Results:\n{diverse_items}")

# BSwap Diversifier
logger.info("Starting BSwap Diversification")
bswap = BSwapDiversifier(model_name="all-MiniLM-L6-v2", device="cuda")
result = bswap.diversify(items, top_k=3, theta=0.6)
logger.info(f"BSwap Diversification completed. Results:\n{result}")

# Motley Diversifier
logger.info("Starting Motley Diversification")
motley = MotleyDiversifier(model_name="all-MiniLM-L6-v2", device="cuda")
result = motley.diversify(items, top_k=3, div_threshold=0.5)
logger.info(f"Motley Diversification completed. Results:\n{result}")

# MaxSum Diversifier
logger.info("Starting MaxSum Diversification")
msd = MaxSumDiversifier(model_name="all-MiniLM-L6-v2", device="cuda")
result = msd.diversify(items, top_k=3, lambda_=0.67)
logger.info(f"MaxSum Diversification completed. Results:\n{result}")
