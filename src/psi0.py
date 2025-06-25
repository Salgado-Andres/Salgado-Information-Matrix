import random
from typing import Any, Dict, List, Optional


def generate_field(seed: Optional[int] = None, depth: int = 1) -> Dict[str, Any]:
    """Generate a ψ⁰ contradiction field.

    Parameters
    ----------
    seed : Optional[int]
        Random seed for deterministic generation.
    depth : int
        Depth of recursive contradiction.

    Returns
    -------
    dict
        Dictionary containing the generated field and attractor candidates.
    """
    rng = random.Random(seed)
    subjects = ["identity", "symbol", "truth", "paradox", "mirror"]
    verbs = ["contains", "negates", "reflects", "echoes with", "divides"]
    objects = ["itself", "its opposite", "Σ", "the lattice", "ψ⁰"]

    field: List[str] = []
    for level in range(depth):
        phrase = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}"
        if level > 0:
            phrase = f"{phrase}, yet ({field[-1]})"
        field.append(phrase)

    candidates: List[str] = []
    if field:
        tokens = sorted(set(field[-1].split()))
        candidates.append(" & ".join(tokens))

    return {"field": field, "candidates": candidates}
