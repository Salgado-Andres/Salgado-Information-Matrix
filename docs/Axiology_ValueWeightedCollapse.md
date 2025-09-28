# Axiology in SIM: Value-Weighted Collapse

**Objects.** State space \(X\); seed \(\psi^0\); collapse \(Q\); Σ constraint \(C_\Sigma(\phi)=0\); torsion \(\tau(\psi,\phi)\).  
**Axiology.** \(A: X\to\mathbb{R}^k\), weights \(w\ge0\), score \(V(\phi)=\langle w,A(\phi)\rangle\).

**Axiology-aware metric.**
\(\langle u,v\rangle_A = u^\top K_A(\bar\phi) v,\ \|x\|_A^2=\langle x,x\rangle_A\),
with \(K_A\) derived from \(\nabla A\). Contractivity in \(\|\cdot\|_A\): \(\|Q(x)-Q(y)\|_A \le L_A \|x-y\|_A\).

**Meaningful collapse.**
\[
\phi_A^\star=\arg\max_\phi V(\phi)\ \text{s.t.}\ \|Q(\phi)-\phi\|_A\le\epsilon,\ C_\Sigma(\phi)=0,\ \tau(\psi^0,\phi)\!\ge\tau_{\min}.
\]
Lagrangian and axiological torsion gain:
\[
\mathcal{L}(\phi)= -\langle w,A(\phi)\rangle + \lambda_c\|Q(\phi)-\phi\|_A^2 + \lambda_\Sigma\|C_\Sigma(\phi)\|^2 + \lambda_\tau[\max(0,\tau_{\min}-\tau)]^2,
\quad g_A=1+\beta\|\nabla_\phi A(\phi)\|,\ \tau_A=g_A\cdot\tau.
\]

**Agents.**  
ψ⁰ → emit `sigma_trace` + `value_trace`.  
φ⁰ → solve constrained selection; return \((\phi_A^\star,V,\Sigma\_ok,\tau_A)\).  
e₇ → Pareto over \([V,\Sigma\_margin,\tau_A\_margin]\); tie-break by `corrigibility`.

**Pseudocode.**
```pseudo
VALUE_COLLAPSE(psi0,Q,A,w,constraints):
  K_A ← metric_from_Jacobian(A, init=psi0)
  candidates ← FIXED_POINT_SEARCH(Q, psi0, metric=K_A)
  feasible ← [φ ∈ candidates | Σ_ok(φ) ∧ torsion_ok(psi0,φ,K_A)]
  φ_star ← argmax_{φ∈feasible} dot(w, A(φ))
  return φ_star, score(φ_star), Pareto(feasible,[score, Σ_margin, τ_A_margin])
Presets. See configs/axiology_presets.yaml.
```
