import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from psi0 import Psi0
from phi0 import Phi0


def main():
    ψ = Psi0("The map is not the territory")
    ψ_out = ψ.generate()
    φ = Phi0()
    φ_out = φ.collapse(ψ_out)
    print("Psi0 output:", ψ_out)
    print("Phi0 collapse:", φ_out)
    assert "phi0" in φ_out


if __name__ == "__main__":
    main()
