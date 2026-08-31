# Thirteen Guesses Each: Bayesian Optimisation of Eight Black-Box Functions

**Imperial College AI/ML Programme — BBO Capstone**
Eight unknown objectives, 2-D to 8-D, one query per function per week for thirteen weeks. Scored on the final submission only.

---

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT

Eight hidden formulas. Each one takes a handful of numbers between 0 and 1 and hands back a single score, and the only way to learn anything about them is to submit a guess and wait a week for the answer. Thirteen guesses each, then it stops.

This repository is the complete record of those thirteen weeks — every guess, every score, and the reasoning that connected one to the next. It is not a success story. Six of my eight final answers were worse than values I had already found earlier and failed to keep track of. The most useful thing in here is the diagnosis of why that happened.

---

## DATA

**Seed data.** The course portal supplied `initial_inputs.npy` and `initial_outputs.npy` — a small set of pre-evaluated points to anchor the first surrogate fits. Provided by the Imperial College AI/ML programme (delivered via Emeritus); not publicly redistributable, so referenced rather than committed.

**Generated data.** Everything after that is self-generated: 13 rounds x 8 functions = 104 query–response pairs. One input vector submitted per function per round, one scalar returned. No gradients, no closed form, no noise model, no information about modality. Maximisation in all eight cases. Feedback is batched and delayed — all eight queries commit before any result is seen, so a round cannot be corrected mid-flight.

**Two gaps in the record.** Outputs for Weeks 3 and 7 were not retained. Both rounds were sub-10^-3 perturbations of an adjacent round, so their values are interpolated rather than measured; Week 7's F8 input is byte-identical to Week 6's, making that one value exact by construction. The gaps are flagged in the ledger below and excluded from all best-value claims.

**What each surface turned out to look like**, reconstructed from the full 104-point history:

| Fn | d | Observed range | Character |
|---|---|---|---|
| F1 | 2 | 10^-187 to 10^-3, alternating sign | Oscillatory with nodal lines; magnitude grows sharply toward x1 ~ 0.42–0.53 |
| F2 | 2 | -0.107 to 0.580 | Smooth ridge running roughly along the diagonal |
| F3 | 3 | -0.323 to -0.0154 | Broad and shallow near the top — see Results |
| F4 | 4 | -36.09 to -2.87 | Optimum near the domain centre; punishing toward the corners |
| F5 | 4 | 0.31 to 566.34 | Multimodal, extreme dynamic range, high values at widely separated points |
| F6 | 5 | -2.47 to -0.398 | Mild; optimum interior but not central |
| F7 | 6 | 0.080 to 2.424 | Optimum in a low corner, well away from the centre |
| F8 | 8 | 7.43 to 9.644 | Slow, shallow gradients; hardest to move |

**References**

- Jones, D.R., Schonlau, M. and Welch, W.J. (1998). *Efficient Global Optimization of Expensive Black-Box Functions.* Journal of Global Optimization, 13(4), 455–492.
- Rasmussen, C.E. and Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning.* MIT Press.
- Snoek, J., Larochelle, H. and Adams, R.P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms.* NeurIPS 25.
- Shahriari, B. et al. (2016). *Taking the Human Out of the Loop: A Review of Bayesian Optimization.* Proceedings of the IEEE, 104(1), 148–175.
- Frazier, P.I. (2018). *A Tutorial on Bayesian Optimization.* arXiv:1807.02811.

---

## MODEL

A **Gaussian Process surrogate per function**, built on scikit-learn's `GaussianProcessRegressor`:

- **Kernel** — Matérn-5/2 plus a `WhiteKernel` noise term. Matérn-5/2 assumes twice-differentiable sample paths, a weaker and more honest smoothness assumption than RBF for a surface I know nothing about.
- **Acquisition** — Expected Improvement, maximised over a 5,000-point grid.

**Why a GP and not a regression or SVM surrogate.** Both alternatives return a point estimate with no usable uncertainty. Under this budget the question is never *how well does the model fit the observed points* but *where should the next query go*, and only a calibrated posterior variance answers that. The fitting cost is irrelevant at a dozen observations.

**Where the model stopped carrying the work — and this is the honest core of the project.** A GP fitted to ~12 points in 5, 6 or 8 dimensions is severely under-determined. The posterior is dominated by the prior across almost the whole domain, so Expected Improvement over it returns points selected on essentially no evidence. From roughly Week 8 onward the GP became advisory and the real decisions came from a **directional heuristic layer** sitting on top of it: finite differences, one-dimensional line searches through the domain interior, and quadratic fits along those lines.

That layer produced real, repeatable gains. It also had a defect the GP would not have had: it anchored every move to the *current* incumbent rather than the *best-ever* incumbent, and nothing in the loop ever cross-checked the two. The Results section is mostly about what that cost.

---

## HYPERPARAMETER OPTIMISATION

The hyperparameters split into two tiers. The second tier mattered far more, and was tuned far less rigorously.

### Tier 1 — surrogate hyperparameters

| Hyperparameter | How it was set |
|---|---|
| Kernel length-scales (per dimension) | Maximised log marginal likelihood, with restarts to blunt initialisation sensitivity |
| Signal variance | Marginal likelihood |
| `WhiteKernel` noise level | Marginal likelihood; doubles as a numerical stabiliser |
| Matérn smoothness, nu | Fixed at 5/2 by choice, never searched — with ~12 points there is no basis to select it from data |
| EI exploration parameter, xi | 0.05, held constant across all functions and rounds |
| Acquisition grid | 5,000 points, resampled each round |

Marginal-likelihood fitting is the textbook answer and it is close to meaningless here. Fitting eight length-scales from twelve observations is badly over-parameterised, and the fitted values swung noticeably round to round — itself a diagnostic I should have logged and acted on rather than noticed in passing.

### Tier 2 — search hyperparameters

Each round was effectively a single evaluation of one hyperparameter setting, which is a brutal tuning regime.

**Perturbation magnitude.** The most expensive parameter in the project. Weeks 5–8 used successively smaller perturbations (±0.003, ±0.02, ≤0.001) around a rolling incumbent. Week 6 returned all eight values within 2% of Week 5. Week 8 returned near-exact repeats of Week 7. Four rounds — 31% of the total budget — bought almost no new spatial information.

The one thing it did buy: F8's Week 7 → Week 8 step moved the output from 8.6894 to 8.7507 over an input distance of ~0.057, a directional derivative of about 1.08 per unit distance. A usable finite difference, obtained by accident, and the trigger for everything that followed.

**Step size along an inferred direction.** Tuned by hand per function — continue at full step while improving, halve and reverse on a regression.

**Line parameter lambda.** From Week 11 the search was reparameterised from a d-dimensional point search to a 1-D line search, `x(λ) = x₈ + λ(c − x₈)` with `c` the domain centre. Probing λ ∈ {0.3, 0.5, 0.7, 1.0} improved five functions at once. Week 12 fitted a parabola through three collinear points per function and solved for the vertex:

| Fn | Fitted vertex λ* | Value at the vertex |
|---|---|---|
| F4 | 0.883 | -2.874597 (queried at λ = 0.881) |
| F6 | 0.548 | -0.730692 |
| F7 | 0.866 | 0.534120 |
| F8 | 0.518 | 9.094693 |

Once a vertex is located the line is spent, and further gains need perpendicular moves — which is how Week 13's budget went.

---

## RESULTS

### The round ledger

| Round | What was submitted | What it bought |
|---|---|---|
| W1 | One generic vector reused across all eight functions | **F5 = 566.34** — the best value found anywhere in the project, never beaten |
| W2 | Broad, unstructured exploration across the cube | **F6 = -0.398, F7 = 2.424, F8 = 9.644** — all three best-ever. The most productive round by a wide margin |
| W3 | ~10^-3 perturbation of W2 *(outputs not retained)* | Confirmation of W2 at best |
| W4 | Coordinate-aligned probe (near-identical values within each vector) | F5 = 81.80. One genuinely new region |
| W5 | New mid-cube region | F4 = -4.083 |
| W6 | ±0.003 perturbation of W5 | Nothing. All eight outputs within 2% of W5 |
| W7 | ±0.02 perturbation of W6 *(outputs not retained; F8 input identical to W6)* | Nothing |
| W8 | ≤0.001 perturbation of W7 | **F3 = -0.015429**, best-ever, found essentially by accident. Plus the F8 finite difference |
| W9 | Mirror probes — incumbents reflected through the centre | Every mirror failed (F3 -0.035, F4 -36.09, F6 -2.470). Correctly killed the symmetry hypothesis. **F1 = 1.04x10^-8** best-ever; F2 0.4233 |
| W10 | Direction continuation at reduced step | F5 1.327, F7 0.499. Modest |
| W11 | Line-search reparameterisation, λ ∈ {0.3, 0.5, 0.7, 1.0} | Five simultaneous improvements: F4 -23.62 → -4.06, F5 7.134, F6 -0.731, F7 0.505, F8 9.041 |
| W12 | Quadratic vertex fits along those lines | F4 -2.875, F5 24.903, F7 0.534, F8 9.095 |
| **W13** | Perpendicular probes off the spent lines — **the scored round** | **F2 = 0.5799**, the only final-round best. F5 48.008, F6 -0.579 |

![Best-value trajectory per function across the thirteen rounds](results_trajectory.png)

### Final submission versus best-ever

Scoring is on the Week 13 submission alone. This is what that cost:

| Fn | Scored value (W13) | Best-ever found | Round | Gap |
|---|---|---|---|---|
| F1 | 3.14x10^-12 | 1.04x10^-8 | W9 | ~3,300x worse |
| F2 | **0.579904** | 0.579904 | **W13** | — |
| F3 | -0.015521 | -0.015429 | W8 | -0.6% |
| F4 | -4.658694 | -2.874597 | W12 | -1.78 |
| F5 | 48.007751 | 566.341997 | W1 | ~11.8x worse |
| F6 | -0.578977 | -0.398257 | W2 | -0.18 |
| F7 | 0.334575 | 2.423761 | W2 | ~7.2x worse |
| F8 | 9.275988 | 9.644440 | W2 | -0.37 |

**One of eight.** Only F2 finished on its best value. Under a final-round-only scoring rule the correct terminal move is to resubmit each function's best-known point and spend nothing on exploration. I spent the last round on perpendicular probes instead.

### What actually went wrong

F5 sat at 0.522 in Week 8; Week 1 had returned 566.34. The celebrated "48x improvement" across Weeks 11–13 was a climb back to roughly a twelfth of a value found on the very first submission. The same pattern holds for F6, F7 and F8. Nine rounds of disciplined local refinement went on re-climbing hills I had already stood on.

**Broad exploration beat everything else in high dimensions.** Week 2 was a single unstructured exploratory round and it produced the best-ever value for F6 (5-D), F7 (6-D) and F8 (8-D) — the three hardest functions. Nothing in the following eleven rounds beat any of them. At ~13 queries in 5 to 8 dimensions, one well-spread probe outperformed every refinement strategy I could construct.

**The centre-ward hypothesis was locally right and globally wrong.** After Week 11 I noticed F4, F6, F7 and F8 all improved by moving toward the domain centre, and promoted that into a structural claim about the function family. It reliably improved on the Week 8 incumbents. But F7's true best sits at (0.152, 0.149, 0.072, 0.258, 0.285, 0.741) — a low corner, nowhere near centre, and 4.5x better than anything the line search produced. The hypothesis described a local gradient; I read it as a statement about where optima live.

**F3 was misdiagnosed.** I concluded it sat in a narrow local vertex because every perturbation from ±0.05 to ±0.09 came back worse. Week 13's λ = 0.7 centre-ward jump — a completely different part of the domain — returned -0.015521 against the incumbent's -0.015429, a difference of 9x10^-5. Two widely separated points scoring within 0.6% of each other is far better evidence of a broad shallow plateau than of a narrow vertex. The correct reading was available and I took the opposite one.

**Perpendicular probes confirmed the line vertices were real.** F4 and F7 both regressed on their first off-line moves (-2.875 → -4.659 and 0.534 → 0.335). Negative results, but clean ones: the vertices were genuine local optima in more than just the line direction.

### What I would do differently

1. **Keep a best-ever ledger per function from round one**, and make every proposal justify itself against that value rather than against last round's. This alone would have changed the score more than any modelling change.
2. **Confirm the scoring rule before the final round.** Final-round-only scoring makes the last query a pure exploitation move; I treated it as another exploration step.
3. **Spend the first four or five rounds on a Sobol or Latin hypercube design** scaled per dimension, rather than letting a five-point GP choose queries it has no basis to choose. Week 2 accidentally proved the point and I did not act on it.
4. **Never spend consecutive rounds on sub-0.001 perturbations.** The budget is measured in rounds, not evaluations. Weeks 5–8 donated four rounds back.
5. **Use trust-region BO (TuRBO-style)** for the 5-D to 8-D functions, where a single global GP is hopeless at this sample size.
6. **Add input warping and per-dimension length-scale diagnostics** — F1's oscillatory, sign-flipping surface would have been flagged as pathological for a stationary kernel long before Week 11.
