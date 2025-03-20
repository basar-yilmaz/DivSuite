"""Registry for mapping algorithm names to their implementations."""

from typing import Dict, Optional

# Main registry with canonical names
ALGORITHM_REGISTRY = {
    # Standard algorithms
    "SY": {"module": "src.core.algorithms.sy", "class": "SYDiversifier"},
    "MMR": {"module": "src.core.algorithms.mmr", "class": "MMRDiversifier"},
    "CLT": {"module": "src.core.algorithms.clt", "class": "CLTDiversifier"},
    "Motley": {"module": "src.core.algorithms.motley", "class": "MotleyDiversifier"},
    "MSD": {"module": "src.core.algorithms.msd", "class": "MaxSumDiversifier"},
    "BSWAP": {"module": "src.core.algorithms.bswap", "class": "BSwapDiversifier"},
    "Swap": {"module": "src.core.algorithms.swap", "class": "SwapDiversifier"},
    "GMC": {"module": "src.core.algorithms.gmc", "class": "GMCDiversifier"},
    "GNE": {"module": "src.core.algorithms.gne", "class": "GNEDiversifier"},
}

# Create a case-insensitive lookup map
_NORMALIZED_REGISTRY: Dict[str, str] = {
    name.lower(): name for name in ALGORITHM_REGISTRY.keys()
}
# Add common variations
_NORMALIZED_REGISTRY.update(
    {
        "sy": "SY",
        "mmr": "MMR",
        "clt": "CLT",
        "motley": "Motley",
        "msd": "MSD",
        "sydiversifier": "SY",
        "mmrdiversifier": "MMR",
        "cltdiversifier": "CLT",
        "motleydiversifier": "Motley",
        "maxsumdiversifier": "MSD",
        "bswapdiversifier": "BSWAP",
        "swapdiversifier": "Swap",
        "gmcdiversifier": "GMC",
        "gmc": "GMC",
        "gnediversifier": "GNE",
        "gne": "GNE",
    }
)


def normalize_algorithm_name(name: str) -> Optional[str]:
    """
    Convert algorithm name to its canonical form.

    Args:
        name: Input algorithm name (case insensitive)

    Returns:
        Canonical name if found, None otherwise
    """
    return _NORMALIZED_REGISTRY.get(name.lower())


def get_registered_algorithms():
    """Get list of all registered algorithm names."""
    return list(ALGORITHM_REGISTRY.keys())


def get_algorithm_info(name: str) -> Optional[Dict]:
    """
    Get algorithm info from registry using case-insensitive lookup.

    Args:
        name: Algorithm name (case insensitive)

    Returns:
        Algorithm info dict if found, None otherwise
    """
    canonical_name = normalize_algorithm_name(name)
    return ALGORITHM_REGISTRY.get(canonical_name) if canonical_name else None
