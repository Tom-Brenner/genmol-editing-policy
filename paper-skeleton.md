# Paper skeleton: frozen-GenMol remasking site selection

Handoff for a short methods / negative-result note. Fill `[TODO]` items from JSONL before drafting. Do not retune thresholds, policies, or the value model against the locked test set.

**Pre-registered conclusion.** The primary success criterion was **not met**. On 200 core-disjoint held-out parents, learned site selection did not outperform the largest-fragment heuristic. Parent-level 16-sample success: largest-fragment **0.565**, learned **0.530**, random **0.460**. Paired difference learned − largest = **−0.035** (bootstrap 95% CI **[−0.075, 0.005]**). Candidate validity was matched (both **0.98875**). Point estimate favors the heuristic; the interval includes zero, so the study reports a **null / negative-leaning** result, not a statistically significant harm.

---

## Title (working)

Learning remasking sites for a frozen GenMol sampler does not beat a fragment-size heuristic on constrained QED optimization

## Venue shape

Short empirical note or workshop paper: one frozen generator, one property (RDKit QED), three policies, pre-registered paired test. Not a new generator, not a docking/ADMET claim.

## Abstract (draft)

We test whether a cheap value model can choose which peripheral fragment a frozen molecular language model should remask. Using GenMol V1 SAFE remasking with a protected Bemis–Murcko core, we label each eligible pendant fragment by the mean of eight constrained QED-rollout rewards, train a two-layer MLP on 1,000 ChEMBL 37 parents, and compare learned selection to random choice and a largest-fragment heuristic on 200 held-out, core-disjoint parents (16 completions each). Generator, decoding, fingerprint, and reward are identical across arms; only the selected span differs. Learned selection slightly ranked better on validation mean reward but lost on the pre-registered test estimand: 16-sample parent success was 53.0% (learned) vs 56.5% (largest-fragment) vs 46.0% (random). The paired bootstrap 95% CI for learned minus largest is [−7.5, +0.5] percentage points and does not exclude zero. Candidate validity remained ~99% in all arms. Under this protocol, site choice is not a free lunch: a size heuristic is at least as good as an 8-sample rollout-value MLP.

**Keywords.** fragment remasking; site selection; GenMol; QED; frozen generator; negative result

---

## 1. Introduction

**Claim to argue, not a new model.** Learned molecular edit-site selection is prior work. The question is whether it helps **this** frozen GenMol remasking loop under a budgeted, pre-registered protocol.

**Motivation.** Generators such as GenMol remask one SAFE span at a time. Which pendant to replace is a discrete policy that is usually random or heuristic. If expected constrained-QED return varies across sites, a value model trained on stochastic rollouts should pick better sites without touching generator weights.

**What is held fixed.** Checkpoint, SAFE remask implementation, `num_insert_mask`, softmax temperature, randomness, fingerprint, feasibility constraints, and number of completions per selected action. The intervention is **one chosen fragment per parent**.

**What this paper is not.** Binding, synthesizability, ADMET, clinical value, GenMol fine-tuning, online RL, or multi-objective optimization.

**Result in one sentence.** Pre-registered success (CI lower bound of learned − largest > 0, validity drop ≤ 3 pp) failed; the size heuristic is the stronger baseline on test.

---

## 2. Related work (skeleton)

- GenMol / SAFE fragment remasking and lead-optimization usage (Lee et al., ICML 2025; NVIDIA GenMol repo).
- Learned vs heuristic edit-site or attachment-point selection in generative chemistry (cite 3–5 site-selection / fragment-replacement papers; this work does not claim novelty of the idea).
- Constrained property optimization with QED and similarity (GuacaMol / PMO-style caveats: QED is cheap and gameable).
- Frozen-generator + policy head vs generator fine-tuning (position as an ablation of the former).

Keep this section short. The contribution is evidence, not a method name.

---

## 3. Methods

### 3.1 Task

Given parent \(x\) with a protected Bemis–Murcko core and 2–4 eligible pendant fragments, choose one fragment \(a\), remask its SAFE span with frozen GenMol, and score each completion \(y\).

### 3.2 Feasibility and rollout reward (pre-registered)

\[
\begin{aligned}
\mathrm{feasible}(y) &= \mathrm{RDKit\text{-}sanitized}(y) \\
&\quad \wedge \text{exact mapped-core preservation} \\
&\quad \wedge \mathrm{Tanimoto}_{\mathrm{ECFP4}}(x,y) \ge 0.50 \\
&\quad \wedge |\mathrm{HAC}(y)-\mathrm{HAC}(x)| \le \mathrm{round}(\tfrac13 \mathrm{HAC}(x)) \\[0.5em]
r(x,a,y) &= \mathrm{feasible}(y) \times \mathrm{clip}\!\left(\frac{\mathrm{QED}(y)-\mathrm{QED}(x)}{0.10},\,0,\,1\right)
\end{aligned}
\]

Training labels use the **mean of eight** independent rewards, not best-of-eight:

\[
\hat R(x,a) = \tfrac18 \sum_{k=1}^{8} r(x,a,y_k).
\]

Exact core means the atom-mapped Murcko core of the parent is recovered on the candidate (element, aromaticity, charge, internal bonds). A second unmapped copy of the core SMARTS does not count. Mapping failure is infeasible. Raw Tanimoto, HAC, and uncapped \(\Delta\)QED are stored so thresholds can be re-analyzed without re-generation.

### 3.3 Parent-level success (primary test metric)

A parent–policy pair succeeds if **any** of 16 completions is feasible and \(\Delta\mathrm{QED} \ge 0.05\). This is a different functional of the same generator than \(\hat R\) (mean clipped reward). Do not conflate them in the discussion.

### 3.4 Data

- Source: ChEMBL 37 chemreps (`chembl_37_chemreps.txt.gz`, SHA-256 `ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7`), CC BY-SA 3.0.
- Filters: 15–45 heavy atoms; QED in \([0.30, 0.85]\); uniquely verifiable Murcko core; 2–4 pendant fragments joined by one cuttable core–pendant bond; unique RDKit→SAFE span alignment via a pendant-bond slicer.
- Yield: scanned 17,108 standardized molecules; eligible pool 5,600; sampled 1,400.
- Split **before** any policy rollouts, grouping by canonical Murcko core (seed `20260919`): 1,000 / 200 / 200 train / val / test (470 / 87 / 128 distinct cores). Test cores are disjoint from train. This does not prove GenMol never saw related structures in pretraining.

### 3.5 Generator (frozen)

- GenMol V1 remasking checkpoint (HF revision `cdaf0c1b45c281315ab10389836bc46093cdbb65`); vendor source `add09fc83b7255bd09c797e527c0f4b51f5fb7c1`.
- Native SAFE remask of one `.`-delimited fragment; `num_insert_mask = 10` (pinned midpoint of GenMol’s Uniform{5,…,15}); `softmax_temp=1.2`, `randomness=2.0`, `gamma=0`, `w=2`, `fix=true`, `min_add_len=18`.
- Per-completion `torch.Generator` seeds; same parent+action rollouts batched without a shared global seed.
- Precision: bf16 on Ampere+, else fp16 on CUDA, else fp32. Record actual precision in provenance.

### 3.6 Policies

Only the selected eligible fragment differs.

1. **Random.** Uniform among 2–4 actions; per-parent seed derived from `random_policy_seed`.
2. **Largest fragment.** Max fragment heavy-atom count; ties by canonical order.
3. **Learned.** Argmax of MLP \(\hat R(x,a)\); ties by canonical order.

Hard eligibility mask at train and test. No GNN, no categorical policy gradient, no online updates.

### 3.7 Value model

- 13 fixed RDKit features: parent MolWt, LogP, HBD, HBA, HAC; core HAC; fragment HAC and HAC fraction; one-hot attachment bond (single/double/aromatic); ring-fragment and terminal-fragment flags.
- StandardScaler fit on **train actions only**.
- MLP: Linear(13, 64)–ReLU–Linear(64, 32)–ReLU–Linear(32, 1).
- Weighted Huber (\(\delta=1\)); weight \(1/(1+\mathrm{Var}[r])\); Adam \(10^{-3}\); batch 64; max 200 epochs; early stop patience 20 on validation MSE.
- Selected checkpoint: epoch **51**, val MSE **0.069** (RMSE ≈ 0.26 on a \([0,1]\) target).

### 3.8 Compute and gate

- One GCP `g2-standard-4` (NVIDIA L4); budget cap 180 GPU-h / $200 study ceiling.
- Phase 0: 100 probe parents × 10 completions, not in the 1,400-task manifest. Gate: ≥25% valid, ≤20 s/completion, 50-record core/mapping audit.
- Observed probe (fill exact JSONL stats if they differ): validity **0.992**, mean wall **0.137 s/completion**. `[TODO: peak VRAM, CUDA/PyTorch versions, GPU-hours, $ spend]`

---

## 4. Experiment protocol

| Phase | What was frozen / measured | Status |
| --- | --- | --- |
| 0 | Implementation, checkpoint, decoding, probe throughput | Passed gate |
| 1 | Core-disjoint manifest; val labels (8 / action) | Done |
| 2 | Train labels (8 / action); one MLP selected on val | Done |
| 3 | Test: one action / policy / parent; 16 completions; metrics before example inspection | **Locked primary result** |
| 4 | Independent seeds, learned vs largest only, 16 new samples | `[TODO: run or state not run]` |

Primary estimand: paired difference in parent success, learned − largest-fragment, 10,000-replicate paired bootstrap over 200 parents (seed `20260925`).

**Success criterion (pre-registered).** CI lower bound > 0 **and** learned candidate validity no more than 3 percentage points below the stronger baseline.

**No peeking.** Test manifest and `results/test` are write-protected after evaluation. Do not change policies or hyperparameters after Phase 3.

---

## 5. Results

### 5.1 Validation (model selection, not the claim)

Mean \(\hat R\) of the selected action on 200 validation parents:

| Policy | Mean selected-action \(\hat R\) |
| --- | ---: |
| Random | 0.232 |
| Largest fragment | 0.240 |
| Learned | 0.249 |

Small ranking signal (~+0.010 vs largest). Weak regression (val MSE 0.069). A validation gain was **not** required to proceed.

### 5.2 Held-out test (primary)

200 parents × 16 rollouts × 3 policies = 9,600 completions.

| Policy | Parent success (any of 16) | Candidate validity |
| --- | ---: | ---: |
| Random | 0.460 | 0.9825 |
| Largest fragment | **0.565** | 0.98875 |
| Learned | 0.530 | 0.98875 |

- Learned − largest: **−0.035**, 95% CI **[−0.075, 0.005]**, \(n_{\mathrm{boot}}=10{,}000\).
- Validity difference learned − largest: **0.0 pp**.
- `success_criterion_met`: **false**.

**How to state this in the paper.** Fail to reject no-difference at 95% while the point estimate favors largest-fragment by 3.5 pp. Random is clearly worse than both structured policies (~10 pp below largest). The generator remains chemically valid (~98–99%) regardless of site policy; the gap is **which parents get a +0.05 QED hit**, not sanitization.

### 5.3 Secondary metrics `[TODO: compute from locked JSONL, no re-generation]`

Plan-required descriptives, all policies, failures retained:

- Exact core-preservation rate
- Median / mean \(\Delta\)QED (raw and feasible-only)
- Tanimoto distribution
- Heavy-atom \(\Delta\) vs per-parent budget
- Unique canonical SMILES per parent
- Predicted vs observed action value on val (calibration scatter)
- Cost per successful parent; completions per GPU-hour
- Agreement: how often learned and largest pick the **same** fragment `[TODO: from results/test/selections.json]`

### 5.4 Figures (planned)

1. Protocol diagram: parent → 2–4 pendants → policy → frozen remask → constrained QED.
2. Bar chart of test parent success with bootstrap CI on the paired difference (primary figure).
3. Validation \(\hat R\) vs test success (show metric mismatch).
4. Optional: \(\Delta\)QED histograms; site-overlap Venn (learned ∩ largest).

No cherry-picked molecules in the main text until metrics are locked. If examples are shown, include failures.

---

## 6. Discussion

**Negative result, usefully bounded.** With validity matched and the CI upper bound only +0.5 pp, there is no support for deploying this MLP over “remask the heaviest pendant.” There is support that **unstructured random** is inferior to a size heuristic (~10 pp).

**Why validation and test can disagree.**

1. Different functionals: val compared mean 8-sample clipped reward of the chosen action; test used any-of-16 binary success at \(\Delta\)QED ≥ 0.05.
2. Eight rollouts are a noisy \(\hat R\); Huber-on-MSE can prefer calibrated averages over the tail that drives any-of-16 success.
3. Largest-fragment is already a strong, chemically plausible prior (more atoms to rewrite → more QED headroom under a 1/3 HAC budget). The MLP includes fragment HAC, so it may be a noisy approximation of the same heuristic.
4. Core-disjoint test cores (128 vs 87 val) can shift which sites matter.

**Interpretation limits (pre-registered).** Null effect can mean (i) site choice barely matters relative to GenMol sampling noise, (ii) eight labels are too few, (iii) 13 RDKit scalars underfit the site, or (iv) QED+constraints make many pendants equivalent. The design cannot separate these without new, pre-registered runs.

**Do not upgrade the claim.** Do not call this “QED optimization failed” — random already succeeds on 46% of parents. Do not call the MLP harmful — the CI includes 0. Do not generalize to docking, synthesizability, or other generators.

---

## 7. Limitations

- QED is a toy drug-likeness score; constrained \(\Delta\)QED is not bioactivity.
- GenMol pretraining leakage of ChEMBL-like structures is untestable here.
- Eligibility is 2–4 single-cut pendants only; fused or multi-cut edits are out of scope.
- One MLP width, one label budget (8), one test budget (16); no architecture search after seeing test.
- Phase 4 independent-seed confirmation `[TODO]`.
- Probe audit `[TODO: exact core/mapping error count; earlier 50-record pass was validity-clean with possible non-zero exact-core misses — state the audit table honestly]`.

---

## 8. Reproducibility

- Configs: `configs/experiment.yaml`, `configs/genmol_sampling.yaml`.
- Frozen split: `manifests/{train,validation,test}.jsonl`, `manifests/hashes.json`, `manifests/seeds.json`.
- Locked test: `results/test/{random,largest_fragment,learned}.jsonl`, `results/test/metrics.json` — `chmod -R a-w results/test`.
- Code: this repository; do not re-implement reward in the paper without matching `src/rollout_policy/scoring.py`.
- ChEMBL attribution: CC BY-SA 3.0.

---

## 9. Outline to paste into a draft

1. Introduction (frozen remask; site policy; pre-registration)
2. Related work (short)
3. Protocol (reward, success, split, generator, MLP, three policies)
4. Results (val table; test table + CI; secondary appendix)
5. Discussion (null; heuristic strength; metric mismatch)
6. Conclusion: **Under a frozen GenMol V1 remasking protocol, an 8-sample rollout-value MLP does not improve 16-sample constrained-QED parent success over choosing the largest eligible fragment.**

**One-line takeaway for collaborators.** Keep the largest-fragment baseline; do not spend further GPU on this MLP configuration unless Phase 4 or a pre-registered feature/label change is justified independently of the locked test numbers.
