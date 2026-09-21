# Budget-capped experiment: rollout-value learning for GenMol remasking

## Decision

Run one focused experiment: keep GenMol frozen and learn the expected reward of remasking each eligible peripheral fragment.

> Given a parent molecule and protected core, select the eligible fragment with the highest predicted expected GenMol rollout reward. The generator checkpoint, remasking implementation, decoding parameters, and candidate budget are identical across all arms.

This is a narrow, reproducible search-policy ablation. Learned molecular edit-site selection is established prior work; the question here is whether it improves GenMol remasking. It is not a new molecular generator and does not establish binding, synthesis, ADMET, or clinical value.

Use one property only: RDKit QED increase while preserving a molecular core. QED is deterministic and cheap, making it suitable for a $200 policy experiment. Do not add cLogP, docking, biological activity, MoCo, online reinforcement learning, or a second objective.

## Hypothesis

For a frozen GenMol sampler, a value model trained on stochastic per-fragment rollouts selects remasking sites that yield more acceptable QED-improved candidates than random selection and a fixed, size-based heuristic.

The intervention is the selected fragment only. Each policy chooses one fragment, then receives 16 GenMol completions per held-out parent.

## Pre-registered rollout reward

For parent molecule `x`, eligible fragment `a`, and GenMol completion `y`, define:

```text
feasible(y) =
    RDKit-sanitized(y)
    ∧ exact preservation of the protected core
    ∧ Morgan-Tanimoto(x, y) >= 0.50
    ∧ abs(heavy_atoms(y) - heavy_atoms(x)) <= 8

r(x, a, y) = feasible(y) × clip((QED(y) - QED(x)) / 0.10, 0, 1)
```

The protected core is the atom-mapped Bemis–Murcko core of the parent. Its atoms and internal bonds must be recovered through the SAFE-to-RDKit conversion and match exactly. A newly generated copy of the same SMARTS elsewhere does not count. A mapping failure is infeasible.

The training target is the mean of eight independent samples:

```text
Rhat(x, a) = mean(r(x, a, y_1), ..., r(x, a, y_8))
```

Store all eight rewards, validity and core-preservation indicators, output structures, and seeds. Do not use the best observed completion as a label: best-of-eight labels systematically favor high-variance actions.

## Data and split

Create exactly 500 parent tasks from one pinned ChEMBL release. ChEMBL supplies a public, chemically diverse parent distribution; it is not used as experimental supervision because the reward is calculated.

After standardized RDKit processing, retain neutralizable small molecules with:

- 15–45 heavy atoms;
- QED in `[0.30, 0.85]`;
- a non-empty Bemis–Murcko core;
- 2–4 eligible pendant fragments, each joined to the protected core by one cuttable bond; and
- a verified mapping from the RDKit fragment to a GenMol SAFE remasking span.

Exclude salts, mixtures, failed sanitizations, multiple disconnections, and parents whose protected core cannot be uniquely verified after conversion. If more than four fragments are eligible, retain the first four using a deterministic canonical ordering recorded in the manifest.

Split before generating rollouts, grouping by Bemis–Murcko core: 350 training, 75 validation, and 75 held-out test parents. Make the test manifest, seed schedule, and output directory read-only until final evaluation. This prevents policy-label leakage across related cores, although it cannot prove the GenMol checkpoint never encountered related structures during pretraining.

Source: [ChEMBL](https://www.ebi.ac.uk/chembl/), available under CC BY-SA 3.0.

## Generator, value model, and baselines

Use the released GenMol V1 remasking checkpoint and its native SAFE workflow. V2 is excluded because the repository reports slightly worse lead-optimization performance; switching checkpoints would confound the policy comparison. Record the repository commit, checkpoint hash, CUDA/PyTorch versions, and every decoding parameter in Phase 0.

Compare exactly three policies:

1. **Random:** uniformly choose one eligible fragment.
2. **Largest fragment:** choose the eligible pendant fragment with the most heavy atoms; break ties by canonical order.
3. **Learned rollout value:** choose the fragment with the largest predicted `Rhat(x, a)`.

The learned model is a two-layer MLP with a scalar predicted value. Inputs are fixed RDKit descriptors for the parent, protected core, and candidate fragment; candidate heavy-atom fraction; attachment-bond type; and ring/terminal-fragment flags. Train it only on training action labels using weighted Huber loss; select the sole model configuration using validation mean squared error and top-1 action value.

The eligibility mask is hard-enforced during training and inference. No GNN, categorical policy, historical-edit supervision, property predictor, or online policy updates are in scope.

## Budget and throughput gate

Use one on-demand GCP `g2-standard-4` with an NVIDIA L4. At the planning rate of approximately $0.71/hour, impose a 180 GPU-hour cap (about $128). Reserve the remaining $72 of the $200 ceiling for disk, checkpoint/image transfer, regional price variation, and setup failures. Confirm the regional all-in price before launch; the dollar cap overrides this estimate.

Set billing alerts at $150 and $175. Terminate GPU instances before accrued spend reaches $185, and write JSONL/checkpoints after every batch. Do not use an A100 or train GenMol.

Phase 0 includes 1,000 batched completions from 100 probe molecules that are not in the 500-task manifest. Measure effective seconds per completion, including conversion and scoring.

- **Proceed:** at least 25% valid completions, correct core checking in a 50-example manual audit, and at most 20 seconds per completion.
- **Reduce:** at 20–30 seconds per completion, use 250 training parents while retaining the 75-parent validation and test sets.
- **Stop:** above 30 seconds per completion, validity below 25%, or unreliable core mapping. Deliver the compatibility/baseline result rather than an underpowered learned-policy result.

The planned workload is about 14,000 training-and-validation label completions, 3,600 primary-test completions, 2,400 independent-confirmation completions, and 1,000 probes: roughly 21,000 completions. At 20 seconds each, this is about 117 GPU-hours, leaving setup and retry capacity within the cap.

Sources: [GCP pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized), [GenMol](https://proceedings.mlr.press/v267/lee25o.html), and the [GenMol repository](https://github.com/NVIDIA-Digital-Bio/GenMol).

## Execution

### Phase 0 — compatibility and throughput (maximum 8 GPU-hours)

1. Provision the L4 VM and persistent disk; configure the alerts and termination guard.
2. Install the pinned GenMol V1 environment and run its native remasking example.
3. Validate RDKit/SAFE mapping, protected-core preservation, and the 1,000-completion throughput gate.
4. Write a versioned probe JSONL containing parent, core map, selected span, seed, output, reward components, wall time, and peak VRAM.

**Exit criterion:** satisfy the throughput gate and freeze the implementation, checkpoint, and decoding settings before opening the 500-task manifest.

### Phase 1 — manifest and baseline validation (maximum 25 GPU-hours)

1. Build and freeze the 350/75/75 core-disjoint ChEMBL manifest.
2. Implement the random and largest-fragment policies and reward computation.
3. On validation only, produce eight rollouts for every eligible action and audit 50 parent/action/output records.
4. Confirm that no validation or test core occurs in the training manifest.

**Exit criterion:** every record is traceable to input, selected SAFE span, output, seed, and reward components. The manual audit has zero core-preservation or fragment-mapping errors.

### Phase 2 — rollout labels and value-model training (maximum 60 GPU-hours)

1. For every training action, generate exactly eight independent completions using fixed, recorded seed schedules.
2. Compute and store `Rhat`, preserving every per-rollout outcome.
3. Fit the MLP, with early stopping and all tuning confined to validation.
4. On validation, compare the learned policy's selected-action value with random and largest-fragment selection. Select only one MLP configuration.

**Exit criterion:** a frozen value-model checkpoint and validation report. A validation gain is not required to continue: the held-out result is reported even for a null policy.

### Phase 3 — single held-out evaluation (maximum 55 GPU-hours)

1. For every test parent, select one fragment per policy without generating test action labels.
2. Generate exactly 16 completions for every selected action. Use the same top-level seed schedule across policies where supported, retaining actual seeds in the output records.
3. Compute all metrics before inspecting examples.
4. Estimate paired differences in parent-level success with a paired bootstrap over the 75 parents.

**Exit criterion:** a locked result file with metrics, 95% confidence intervals, and raw outputs. Do not change a policy or hyperparameter after this point.

### Phase 4 — independent-seed confirmation (maximum 20 GPU-hours)

Repeat Phase 3 for learned rollout value and the stronger pre-specified baseline only, using 16 new samples per selected action and no further fitting. Report this as seed-sensitivity confirmation, not a second independent test set.

## Metrics and decision rule

For each parent and policy, success means at least one of its 16 samples is:

```text
RDKit-sanitized
∧ protected core exactly preserved
∧ Tanimoto >= 0.50
∧ absolute heavy-atom change <= 8
∧ QED(candidate) - QED(parent) >= 0.05
```

The primary estimand is the paired difference in 16-sample parent success rate between learned rollout value and largest-fragment selection. Report its paired-bootstrap 95% confidence interval.

Secondary metrics are candidate-level validity, exact protected-core preservation, median QED delta, molecular-similarity distribution, heavy-atom change, unique canonical-SMILES count, mean predicted versus observed action value, cost per successful parent, and effective completions per GPU-hour. Report all generation attempts and sanitized-only outcomes separately; never discard failures before calculating a rate.

**Success criterion:** the lower bound of the paired 95% confidence interval for the primary difference is greater than zero, with no more than a 3-percentage-point absolute decrease in candidate validity versus the stronger baseline.

**Stop criterion:** trigger the Phase 0 stop gate, reach $185 accrued spend, or lack enough remaining budget for the pre-registered 16-sample test. Preserve and report the reproducible compatibility or negative result.

## Risks and limits

- **Null effect:** random remasking may already be adequate, or site choice may matter less than GenMol sampling. The fixed baselines and held-out evaluation make this a useful result.
- **Noisy values:** eight rollouts are an imperfect estimate of expected reward. Mean labels, retained outcomes, and independent seed confirmation limit but do not remove noise.
- **Representation mismatch:** SAFE spans may not map to local graph fragments. The mapping gate and manual audit are mandatory; a loose SMARTS-presence check is not acceptable.
- **QED optimization:** constraints limit trivial changes, but outputs are not thereby synthesizable or biologically active.
- **Limited novelty:** learned edit-site selection predates this project. The contribution is evidence about this frozen GenMol workflow.

## Deliverables

1. Pinned environment, cost log, and throughput report.
2. Core-disjoint ChEMBL manifest and eligibility/mapping specification.
3. Per-rollout JSONL and action-level rollout-value labels.
4. Frozen MLP checkpoint and validation selection report.
5. Locked held-out and independent-seed confirmation reports with a clear gain, null, or inconclusive conclusion.
