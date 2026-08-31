# Datasheet — BBO Capstone Query History

*Follows the datasheets-for-datasets framework (Gebru et al., 2021). Describes the complete query history of the Imperial College AI/ML BBO capstone: 13 rounds against eight unknown objective functions, now frozen as the project record.*

---

## 1. Motivation

**Why was this dataset created?**
To support a black-box optimisation challenge. Eight unknown functions, each accepting between 2 and 8 continuous inputs in approximately [0.001, 1], each returning a single scalar. The objective is maximisation under a hard budget of one query per function per week for thirteen weeks. The dataset *is* the search history — there is no separate corpus, and every row cost one irreplaceable query.

**What task does it support?**
Fitting the surrogate models that proposed each week's queries; reconstructing the reasoning behind each proposal; and, retrospectively, auditing where the best-known point of each function lay relative to where the strategy believed it lay. The third use proved the most informative.

**Who created it and who funded it?**
Created by the project author as coursework for the Imperial College AI/ML programme (delivered via Emeritus). No external funding. The underlying objective functions belong to the course provider.

---

## 2. Composition

**What does an instance represent?**
One query. Each row is an input vector submitted to one function in one round, paired with the scalar the portal returned.

**Data dictionary**

| Field | Type | Description |
|---|---|---|
| `function_id` | categorical, F1–F8 | Which objective was queried |
| `round` | integer, 1–13 | Submission week |
| `x1 … xd` | float, ~[0.001, 1] | Input vector; `d` varies by function |
| `y` | float | Returned objective value; unbounded, maximisation |

**How many instances?**

| Fn | d | Rounds | Output range observed | Best observed | Round found |
|---|---|---|---|---|---|
| F1 | 2 | 13 | 10^-187 to 10^-3, sign-alternating | 1.04x10^-8 | W9 |
| F2 | 2 | 13 | -0.107 to 0.580 | 0.579904 | W13 |
| F3 | 3 | 13 | -0.323 to -0.0154 | -0.015429 | W8 |
| F4 | 4 | 13 | -36.09 to -2.87 | -2.874597 | W12 |
| F5 | 4 | 13 | 0.31 to 566.34 | 566.341997 | W1 |
| F6 | 5 | 13 | -2.47 to -0.398 | -0.398257 | W2 |
| F7 | 6 | 13 | 0.080 to 2.424 | 2.423761 | W2 |
| F8 | 8 | 13 | 7.43 to 9.644 | 9.644440 | W2 |

104 queries submitted across thirteen rounds. Seed observations were additionally supplied by the portal as `initial_inputs.npy` and `initial_outputs.npy`; see L2 regarding their per-function decomposition.

**Does it contain confidential or sensitive data?**
No. The functions are synthetic and the inputs are arbitrary coordinates. No personal, demographic, or proprietary information of any kind.

**Is it self-contained?**
The generated query history is. The seed arrays are course property and are referenced rather than redistributed.

---

## 3. Collection Process

**How was the data acquired?**
Each round, an input vector per function was proposed by the optimisation loop, submitted to the course portal, and the returned scalar appended to the record. Feedback is batched and delayed: all eight queries commit before any result is seen.

**Sampling strategy, by phase**

| Rounds | Strategy | Sampling character |
|---|---|---|
| W1 | One generic vector reused across all eight functions | Arbitrary single point |
| W2 | Broad unstructured exploration | Well-spread; the most informative round |
| W3–W4 | Perturbation of W2, then a coordinate-aligned probe | One new region |
| W5–W8 | Successively smaller perturbations (±0.003, ±0.02, ≤0.001) around a rolling incumbent | Tightly clustered around a small number of locations |
| W9 | Mirror probes — incumbents reflected through the domain centre | Deliberate falsification; all mirrors returned worse |
| W10–W12 | Directional continuation, then a 1-D line search toward the centre with quadratic vertex fitting | Collinear by construction |
| W13 | Perpendicular probes off the spent lines | First off-line information for several functions |

**Over what timeframe?**
Thirteen weekly rounds, one query per function per round.

**Was anyone paid, or were ethical review processes applied?**
Not applicable. No human subjects, no third-party data collection.

---

## 4. Characteristics and Limitations

The properties below condition every use of this dataset and should be read before the sections that follow.

**L1 — Effective sample size is smaller than the raw count.**
Several rounds are close perturbations of an adjacent round, returning values within a few percent of the previous week. The raw record contains 104 queries; the number of *distinct* regions probed is materially lower, and this should be accounted for in any analysis that treats rows as independent observations.

**L2 — Seed data decomposition is ambiguous.**
The portal supplied `initial_inputs.npy` (40x8) and `initial_outputs.npy` (20,). Early in the project these were read as eight independent 1-D functions on [0,1]; later rounds established that the functions are 2-D through 8-D. The seed arrays were therefore interpreted under one scheme at the start and a different one later, and the per-function seed allocation should be treated as uncertain.

**L3 — Noise structure is not characterised.**
Values are recorded as returned and treated as point observations. The surrogate carried a fitted `WhiteKernel` noise term throughout, but this dataset does not establish the noise behaviour of the underlying functions. Any analysis requiring a variance estimate at a given coordinate will need evidence beyond what is recorded here.

**L4 — Sampling is clustered, and collinear for several functions.**
Coverage of the domain interior is thin and non-uniform for all eight functions. From W8 through W12, every point submitted for F7 was collinear, so five of its six dimensions received no new information across those rounds. The same pattern holds to a lesser degree for F4, F6 and F8.

**L5 — F1's values span roughly 190 orders of magnitude with sign changes.**
Recorded values range from 10^-187 to 10^-3 and alternate sign across a nodal structure. Conventional summary statistics on the F1 column are not meaningful, and any model assuming stationarity over it will be misled.

**L6 — The record is ordered, not annotated.**
The dataset stores what was submitted and what came back, in submission order, and nothing else. Incumbent status is not flagged on any row. Reconstructing which observation held the best value at any point in the project requires a pass over the full history — a step worth building into any reuse of this data.

---

## 5. Preprocessing, Cleaning and Labelling

**What preprocessing was done?**
None on the stored values. The record holds exactly what was submitted and exactly what came back, to full returned precision.

Inside the modelling code, inputs are treated as already scaled to the unit box and outputs are standardised before GP fitting (`normalize_y` behaviour). These transformations are applied on the fly and never written back.

---

## 6. Uses

**What has it been used for?**
Fitting a per-function Gaussian Process surrogate; deriving finite-difference directions; fitting quadratic vertices along a parameterised line; and the retrospective performance audit reported in the README.

**What else could it be used for?**
- Counterfactual strategy benchmarking — "what would UCB, or Thompson sampling, or a Sobol design have proposed in Week 5, given only the data available then?" The record preserves submission order, so any point-in-time data state is reconstructible by truncation.
- A worked example of budget allocation in sequential optimisation under a round-based rather than evaluation-based constraint.

**What should it NOT be used for?**
- **Inferring the true shape of the eight functions.** Coverage is too thin and too biased (L4). F1 in particular should not be characterised from these observations.
- **Treating any single value as a variance-bounded estimate.** See L3.
- **Benchmarking surrogate model accuracy.** The sampling is adaptive and heavily clustered, so held-out error on this data measures the sampling policy at least as much as the model.
- **Any real-world scientific or engineering conclusion.** The functions are synthetic course artefacts.

---

## 7. Distribution and Maintenance

**How is it distributed?**
Within the capstone GitHub repository, alongside the code that generated it. The generated query history is small enough to host directly.

**Terms of use.**
Generated query data: free for study and research with attribution. Seed arrays: property of the course provider, referenced not redistributed. The underlying objective functions belong to the course; this dataset only records where they were probed and what came back.

**Who maintains it, and will it be updated?**
Maintained by the project author. The final submission has been made and the query budget is exhausted, so the dataset is frozen. No further rounds are possible; any correction to the record would be issued as a versioned commit with the original preserved.
