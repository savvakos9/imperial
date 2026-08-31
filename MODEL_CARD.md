# Model Card — GP Surrogate with a Directional Heuristic Layer

*Follows the model-card framework (Mitchell et al., 2019). Describes the optimisation system used across all thirteen rounds of the Imperial College AI/ML BBO capstone.*

---

## 1. Overview

**Name.** Gaussian Process surrogate with a directional heuristic layer.

**Type.** Sequential black-box optimiser. A predictive model per function, plus a rule layer that converts model output and observed history into the next query.

**Version.** Final. Thirteen rounds complete, budget exhausted, no further updates possible.

**Purpose.** Find inputs that maximise eight unknown scalar objectives of dimension 2 to 8, given one evaluation per function per week for thirteen weeks and no gradients, closed form, or structural information.

**Scoring context.** The capstone is scored on the Week 13 submission alone rather than on the best value found across the campaign. This distinction matters for reading Section 5 and is discussed in Section 6.

---

## 2. Architecture

Two components, and the boundary between them is where the interesting behaviour lives.

**Component A — GP surrogate (per function).**
`scikit-learn` `GaussianProcessRegressor`. Matérn-5/2 kernel plus a `WhiteKernel` noise term. Kernel length-scales, signal variance and noise level fitted by maximising the log marginal likelihood with restarts. Matérn smoothness `nu` fixed at 5/2 by choice rather than search. Expected Improvement acquisition, `xi = 0.05`, maximised over a 5,000-point grid.

Matérn-5/2 was chosen over RBF because it assumes twice-differentiable sample paths rather than infinitely smooth ones — a weaker and more defensible assumption for a surface with no known structure. A GP was chosen over regression or SVM surrogates because both alternatives return point estimates with no usable uncertainty, and under this budget the operative question is where to query next, not how well the model fits what is already observed.

**Component B — directional heuristic layer.**
From roughly Week 8 onward this layer drove most query selection:

- **Finite differences.** Paired near-identical queries yield a signed direction for the cost of one query.
- **Line reparameterisation.** From Week 11, the d-dimensional search was replaced by a 1-D search along `x(λ) = x₈ + λ(c − x₈)`, with `c` the domain centre. Probing λ ∈ {0.3, 0.5, 0.7, 1.0} improved five functions simultaneously.
- **Quadratic vertex fitting.** Three collinear points define a parabola; solve for the vertex.

| Fn | Fitted vertex λ* | Value returned at the vertex |
|---|---|---|
| F4 | 0.883 | -2.874597 (queried at λ = 0.881) |
| F6 | 0.548 | -0.730692 |
| F7 | 0.866 | 0.534120 |
| F8 | 0.518 | 9.094693 |

**Why the layer exists.** A GP fitted to roughly a dozen observations in 5, 6 or 8 dimensions is severely under-determined. The posterior is dominated by the prior across almost the entire domain, so Expected Improvement over it selects points on very little evidence. Component B is a response to that constraint, and its design choices are the subject of Section 6.

---

## 3. Intended Use

**Appropriate for.** Expensive, opaque, continuous objectives where each evaluation costs real time or money and the total budget is in the tens rather than the thousands. Experimental design, process tuning, formulation work, supplier or specification trade-offs where each trial is a physical run.

**Not appropriate for.**
- Problems with plentiful evaluations — simpler search wins, and the GP overhead buys nothing.
- Objectives that drift over time. Nothing here handles non-stationarity in time.
- Discrete or combinatorial spaces. The whole approach assumes a continuous box.
- Settings where a poor query is dangerous rather than merely wasteful. This system deliberately spends queries on probes expected to fail informatively (Week 9's mirrors), which is only acceptable when failure is cheap.
- Oscillatory or highly non-stationary surfaces. F1 demonstrates the failure mode directly.

**Primary audience for this card.** A practitioner deciding whether to reuse this approach, and an assessor checking whether the reasoning was sound. It is deliberately not written for a non-technical stakeholder — Section 8 addresses what such a reader would still need.

---

## 4. Assumptions

Stated explicitly because several proved incorrect.

1. **Local smoothness.** Nearby inputs give nearby outputs. Holds for F2, F4, F5, F6, F8; fails for F1.
2. **Stationarity.** One length-scale per dimension describes the whole domain. Violated by F1 and probably by F5, whose high values appear at widely separated points.
3. **Point observations.** Each returned value was treated as the objective at that coordinate, with observation variance absorbed into the fitted `WhiteKernel` term rather than measured independently.
4. **The current incumbent is the best-known point.** This held early and did not hold from Week 8 onward. See Section 6.
5. **Improvement direction generalises across functions.** The centre-ward hypothesis, formed after Week 11. Locally true, globally false.
6. **The budget is measured in evaluations.** It is measured in *rounds*, and eight parallel queries per round make a round spent on confirmation more expensive than it appears.

---

## 5. Performance

**Metrics.** Best-so-far per function; gain per round; and — added retrospectively — the gap between the scored submission and the best value the system had already found. The third metric is the one that matters and it was not tracked during the campaign.

**Final scored submission versus best found**

| Fn | Scored value (W13) | Best found | Round found | Gap |
|---|---|---|---|---|
| F1 | 3.14x10^-12 | 1.04x10^-8 | W9 | ~3,300x lower |
| F2 | **0.579904** | 0.579904 | **W13** | — |
| F3 | -0.015521 | -0.015429 | W8 | -0.6% |
| F4 | -4.658694 | -2.874597 | W12 | -1.78 |
| F5 | 48.007751 | 566.341997 | W1 | ~11.8x lower |
| F6 | -0.578977 | -0.398257 | W2 | -0.18 |
| F7 | 0.334575 | 2.423761 | W2 | ~7.2x lower |
| F8 | 9.275988 | 9.644440 | W2 | -0.37 |

One of eight final submissions matched the best value the system had found.

**Where the system performed well.** Week 11's reparameterisation from an 8-D point search to a 1-D line search produced five simultaneous improvements — more gain in two rounds than the preceding eight combined. The quadratic vertex fits in Week 12 were accurate: F4's predicted vertex at λ* = 0.883, queried at 0.881, returned the best F4 value of the campaign. Week 9's mirror probes cheaply falsified a symmetry hypothesis that would otherwise have consumed later rounds.

---

## 6. Limitations of the Approach

**L1 — Anchoring to the current round rather than the best round.** The system carried the previous round's result forward as its reference point and did not maintain a running best across the full history. From Week 8 onward, proposals were generated relative to the Week 8 position, which for several functions sat well below values already returned in Weeks 1–2. F5 stood at 0.522 in Week 8; Week 1 had returned 566.34. The improvement from Weeks 11–13 was therefore a recovery toward a level reached earlier rather than a new high. The local reasoning in those rounds was sound; the reference point it operated from was not.

**L2 — GP-EI underperforms broad sampling at this budget in high dimensions.** Week 2 was a single unstructured exploratory round and it produced the best value obtained for F6 (5-D), F7 (6-D) and F8 (8-D) — the three hardest functions. Nothing in the following eleven rounds exceeded them. At roughly a dozen queries in 5 to 8 dimensions, one well-spread probe outperformed every refinement strategy available. This is a property of the method at this sample size rather than an implementation defect.

**L3 — Structural hypotheses were generalised beyond their evidence.** After Week 11, four functions improved by moving toward the domain centre, and this was treated as a claim about where optima lie rather than an observation about a local gradient. F7's best point sits at (0.152, 0.149, 0.072, 0.258, 0.285, 0.741) — a low corner, 4.5x above anything the centre-ward line search reached.

**L4 — Landscape inference from insufficient geometry.** F3 was read as a narrow local vertex because perturbations from ±0.05 to ±0.09 all returned worse values. Week 13's probe into a distant region returned -0.015521 against the incumbent's -0.015429, a difference of 9x10^-5. Two widely separated points scoring within 0.6% of each other indicates a broad shallow plateau rather than a narrow vertex.

**L5 — Terminal strategy was not matched to the scoring rule.** Under final-round-only scoring, the optimal last move is to resubmit each function's best-known point and spend nothing on exploration. Week 13 was allocated to perpendicular probes instead.

**L6 — Budget allocation across the middle rounds.** Weeks 5–8 used successively smaller perturbations around a small number of locations, and Week 6 returned all eight values within 2% of Week 5. Under a round-based budget, closely spaced consecutive rounds return little that the previous round had not already established.

---

## 7. Ethical Considerations and Risk

The objectives are synthetic and contain no personal, demographic or sensitive information, so the conventional fairness questions — disparate performance across groups, representational harm, biased training data — do not arise in their usual form. Two related concerns do apply.

**Sampling bias as a validity risk.** The system's own sampling policy produced a clustered, partly collinear dataset. Any conclusion about these functions is conditioned on where this optimiser chose to look. In a transferred setting — screening formulations or suppliers, say — the equivalent failure is a search that converges confidently on a region because that is where it happened to start, while a better region sits unexamined. The mechanism is identical and the consequences are not synthetic.

**Confident output on limited evidence.** Expected Improvement over an under-determined posterior returns a specific, precise-looking coordinate regardless of how much the model actually knows. Nothing in the system flags when a proposal rests largely on the prior. A user seeing six decimal places may reasonably infer a confidence the model does not hold. Any deployment should surface posterior variance and effective sample size alongside every proposal, and should decline to propose when leave-one-out performance is no better than predicting the mean.

**Transparency as the operative mitigation.** The value of this work to anyone else depends on their ability to check it. Every query, every returned value, every rule change and the reasoning behind it is in the repository, including the reasoning that proved wrong.

---

## 8. What This Card Does Not Tell You

Stated plainly, because a card that omits its own gaps is worse than no card.

- **Sensitivity to `xi`, grid size, or `nu`.** These were fixed by judgement and never varied. There is no ablation.
- **Whether Component A or Component B deserves credit for a given improvement.** The two were never run separately against the same history. A counterfactual replay is possible from the stored data and has not been performed.
- **Performance outside 2–8 dimensions or outside a ~13-query budget.** No evidence either way.
- **Calibration.** Predicted-versus-actual was checked informally per round and never logged systematically. Several predictions were close, including F4's vertex; others were not.
- **Objective noise behaviour.** The `WhiteKernel` term was fitted, not independently characterised, so no variance estimate at a given coordinate should be taken from this system.

**If adopting this approach, establish first:** the noise behaviour of your objective through repeated evaluation; a leave-one-out trust score per surrogate per round; and an ablation separating the surrogate from the heuristic layer.

**Monitoring in deployment.** Track, per round: best-so-far versus the current proposal's reference point (the check that addresses L1); leave-one-out R² per surrogate, with a refusal threshold; predicted versus actual for every proposal; and the fraction of budget spent within some minimum distance of an existing observation, with a cap.

---

## 9. Reflection

**Most valuable property of this system.** The line reparameterisation. Replacing a d-dimensional search with a 1-D search along a justifiable direction converted an intractable problem into a tractable one and worked immediately across five functions. It generalises well beyond this project.

**Greatest weakness.** Not in the surrogate. The system had no persistent record of its own best results, and no amount of modelling sophistication compensates for optimising from the wrong reference point. The highest-value improvement available was a ledger, not a kernel.

**What I would change.** In order of expected value: maintain a running best per function and require every proposal to justify itself against it; confirm the scoring rule before the terminal round; seed with a Sobol or Latin hypercube design scaled per dimension for the first four or five rounds; establish objective noise behaviour early; add a leave-one-out trust gate that blocks proposals from surrogates performing no better than the mean; and adopt trust-region BO (TuRBO-style) for the 5-D to 8-D functions where a single global GP is ineffective at this sample size.

**Compared with other documentation practices.** A model card is stronger than a technical report at forcing disclosure of limitations, because the limitations have their own mandatory heading and cannot be quietly omitted. It is weaker than an audit at establishing whether the stated limitations are complete — nothing in the format compels the author to look for problems they have not already noticed. Here, the most significant limitation was not visible until the full round history was reassembled and compared against the final submission, which no section of the standard template asks for. Section 8 is an attempt to compensate for that.

---

## Appendix — Coverage of the assessment categories

| Category | Where addressed |
|---|---|
| Purpose and intended use | §1, §3 |
| Stated limitations, biases, risks | §4, §6, §7, §8 |
| Intended audience | §3 |
| Assumptions about data and context | §4 |
| Sufficiency for use / not-use decisions | §3, §8 |
| Where information is vague or missing | §8 |
| Misuse and harmful outcomes | §7 |
| Disadvantaged scenarios and edge cases | §6 (L2, L3), §7 |
| Error cases | §5, §6 |
| Information an adopter should request | §8 |
| Deployment monitoring | §8 |
| Reflection and proposed improvements | §9 |
