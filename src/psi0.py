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


class Psi0:
    def __init__(self, seed_or_phrase=None, depth=1):
        self.seed = None
        self.depth = depth
        if isinstance(seed_or_phrase, int):
            self.seed = seed_or_phrase
        elif isinstance(seed_or_phrase, str):
            # Use hash of phrase as seed for deterministic output
            self.seed = abs(hash(seed_or_phrase)) % (2**32)
        else:
            self.seed = None

    def generate(self):
        return generate_field(seed=self.seed, depth=self.depth)
