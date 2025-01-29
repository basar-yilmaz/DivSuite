from algorithms.swap import SwapDiversifier
import numpy as np

from algorithms.mmr import MMRDiversifier
from algorithms.clt import CLTDiversifier

swapper = SwapDiversifier(model_name="Alibaba-NLP/gte-large-en-v1.5", device="cuda")

items = np.array(
        [
            ["1", "Inception - Sci-fi thriller about dreams", 0.95],
            ["2", "The Matrix - Cyberpunk action movie", 0.92],
            ["3", "The Godfather - Crime drama about family", 0.90],
            ["4", "Toy Story - Animated family movie", 0.88],
            ["5", "The Shawshank Redemption - Prison drama", 0.87],
            ["6", "Jurassic Park - Adventure with dinosaurs", 0.85],
        ],
        dtype=object,
    )

result = swapper.diversify(items, top_k=3, lambda_=0.5)

print(result)

mmr = MMRDiversifier(model_name="Alibaba-NLP/gte-large-en-v1.5", device="cuda")

result = mmr.diversify(items, top_k=3, lambda_=0.5)

print(result)

clt = CLTDiversifier(model_name="Alibaba-NLP/gte-large-en-v1.5", device="cuda")

result = clt.diversify(items, top_k=3, pick_strategy="medoid")

print(result)