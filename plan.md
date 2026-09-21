# How to read this plan

Two protocols live in this file.

The locked protocol below is the completed 1,400-parent experiment. Its held-out result is recorded in `paper-skeleton.md` and is not amended here. Experiment V2 was not part of that preregistration.

Experiment V2, specified 21 September 2026, is the active next study. It uses fresh manifests and does not reopen `manifests/test.jsonl` or `results/test`.

**Planning assumption:** the pinned ChEMBL 37 chemreps file already used by the locked protocol contains at least **40,000** parents that pass V2 eligibility (30,000 train / 5,000 validation / 5,000 test). That count is assumed, not re-measured in this revision, and the chemreps file is not downloaded again to check it. On the GPU machine, `scripts/audit_action_counts.py` still has to show that those parents pack into exact core-disjoint quotas before any V2 labels are generated. Forty thousand eligible parents do not by themselves guarantee that packing.

---

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
    ∧ abs(heavy_atoms(y) - heavy_atoms(x)) <= round((1/3) * heavy_atoms(x))

r(x, a, y) = feasible(y) × clip((QED(y) - QED(x)) / 0.10, 0, 1)
```

The protected core is the atom-mapped Bemis–Murcko core of the parent. Its atoms and internal bonds must be recovered through the SAFE-to-RDKit conversion and match exactly. A newly generated copy of the same SMARTS elsewhere does not count. A mapping failure is infeasible.

The training target is the mean of eight independent samples:

```text
Rhat(x, a) = mean(r(x, a, y_1), ..., r(x, a, y_8))
```

Store all eight rewards, validity and core-preservation indicators, output structures, and seeds. Do not use the best observed completion as a label: best-of-eight labels systematically favor high-variance actions.

## Data and split

Create exactly 1,400 parent tasks from one pinned ChEMBL release. ChEMBL supplies a public, chemically diverse parent distribution; it is not used as experimental supervision because the reward is calculated.

After standardized RDKit processing, retain neutralizable small molecules with:

- 15–45 heavy atoms;
- QED in `[0.30, 0.85]`;
- a non-empty Bemis–Murcko core;
- 2–4 eligible pendant fragments, each joined to the protected core by one cuttable bond; and
- a verified mapping from the RDKit fragment to a GenMol SAFE remasking span.

Exclude salts, mixtures, failed sanitizations, multiple disconnections, and parents whose protected core cannot be uniquely verified after conversion. If more than four fragments are eligible, retain the first four using a deterministic canonical ordering recorded in the manifest.

Split before generating rollouts, grouping by Bemis–Murcko core: 1,000 training, 200 validation, and 200 held-out test parents. Make the test manifest, seed schedule, and output directory read-only until final evaluation. This prevents policy-label leakage across related cores, although it cannot prove the GenMol checkpoint never encountered related structures during pretraining.

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

Phase 0 includes 1,000 batched completions from 100 probe molecules that are not in the 1,400-task manifest. Measure effective seconds per completion, including conversion and scoring.

- **Proceed:** at least 25% valid completions, correct core checking in a 50-example manual audit, and at most 20 seconds per completion.
- **Reduce:** at 20–30 seconds per completion, use 250 training parents while retaining the 200-parent validation and test sets.
- **Stop:** above 30 seconds per completion, validity below 25%, or unreliable core mapping. Deliver the compatibility/baseline result rather than an underpowered learned-policy result.

The planned workload is about 28,800 training-and-validation label completions (at 3 actions/parent), 9,600 primary-test completions, 6,400 independent-confirmation completions, and 1,000 probes: roughly 46,000 completions. At 0.14 seconds each this is well under the GPU-hour cap.

Sources: [GCP pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized), [GenMol](https://proceedings.mlr.press/v267/lee25o.html), and the [GenMol repository](https://github.com/NVIDIA-Digital-Bio/GenMol).

## Execution

### Phase 0 — compatibility and throughput (maximum 8 GPU-hours)

1. Provision the L4 VM and persistent disk; configure the alerts and termination guard.
2. Install the pinned GenMol V1 environment and run its native remasking example.
3. Validate RDKit/SAFE mapping, protected-core preservation, and the 1,000-completion throughput gate.
4. Write a versioned probe JSONL containing parent, core map, selected span, seed, output, reward components, wall time, and peak VRAM.

**Exit criterion:** satisfy the throughput gate and freeze the implementation, checkpoint, and decoding settings before opening the 1,400-task manifest.

### Phase 1 — manifest and baseline validation (maximum 25 GPU-hours)

1. Build and freeze the 1000/200/200 core-disjoint ChEMBL manifest.
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
4. Estimate paired differences in parent-level success with a paired bootstrap over the 200 parents.

**Exit criterion:** a locked result file with metrics, 95% confidence intervals, and raw outputs. Do not change a policy or hyperparameter after this point.

### Phase 4 — independent-seed confirmation (maximum 20 GPU-hours)

Repeat Phase 3 for learned rollout value and the stronger pre-specified baseline only, using 16 new samples per selected action and no further fitting. Report this as seed-sensitivity confirmation, not a second independent test set.

## Metrics and decision rule

For each parent and policy, success means at least one of its 16 samples is:

```text
RDKit-sanitized
∧ protected core exactly preserved
∧ Tanimoto >= 0.50
∧ absolute heavy-atom change <= round((1/3) * heavy_atoms(parent))
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

---

# Experiment V2: endpoint-aligned edit-site models

Specified 21 September 2026. Settings live in `configs/experiment_v2.yaml`. Outputs go to `manifests_v2/`, `artifacts_v2/`, `results_v2/`, and `data/processed/parents_v2.parquet`. The locked manifests, labels, checkpoint, and `results/test` stay read-only.

Implementation notes with function-level sketches are in `experiment-v2-handoff.md`. Where this section and that note disagree, this section is the experiment specification.

## Question

For a frozen GenMol V1 remasker, does an edit-site model trained on the same per-completion success event used at test select a better fragment than the largest-fragment heuristic?

The generator checkpoint, SAFE remasking implementation, decoding parameters, and completion budget stay identical across arms. Only the selected fragment changes. QED remains the only property. There is still no docking, activity model, cLogP objective, or online reinforcement learning.

## Why the training target changes

The locked model predicts mean clipped reward

```text
mean(feasible * clip(delta_QED / 0.10, 0, 1))
```

The primary test event is different: a completion succeeds when it is valid, keeps the exact protected core, has Tanimoto at least 0.50, stays inside the heavy-atom budget, and has delta QED at least 0.05. Parent success is that event on at least one of 16 completions.

If an action has per-completion success probability `p`, the probability of at least one success in 16 independent completions is `1 - (1 - p)^16`. That quantity increases strictly with `p`, so the action with the largest predicted `p` is also the right choice for the any-of-16 endpoint. V2 models predict `p`.

Keep `reward`, `rhat`, and reward variance in the label files. Add schema-v2 fields beside them: `success_flags`, `n_trials`, `n_successes`, `p_success_observed`, `label_schema_version`. `parent_success` is the OR of a shared `completion_success` predicate, so training and evaluation cannot drift apart. The preregistered objective is ordinary binomial negative log likelihood on `(n_successes, n_trials)`, normalized by the trial count. Do not reuse inverse reward-variance weights. A beta-binomial model is a later sensitivity analysis, not the primary objective.

Eight trials make the observed proportion noisy. That is accepted. Do not replace the label with the best completion.

## Data

Source, filters, and chemistry rules stay those of the locked protocol: ChEMBL 37 chemreps, SHA-256 `ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7`, 15–45 heavy atoms, QED in `[0.30, 0.85]`, one verifiable Bemis–Murcko core, and pendants joined by one non-ring single bond with a verified SAFE span.

The locked scan stopped at 5,600 eligible parents after 17,108 rows because that pool filled a 1,400-parent manifest. It is not a census. This plan assumes the same file contains at least 40,000 V2-eligible parents.

Splits, drawn before any V2 rollouts and grouped by Murcko core:

| Split | Parents |
| --- | ---: |
| Train | 30,000 |
| Validation | 5,000 |
| Test | 5,000 |

Manifest seed `20261001` is frozen before the scan. Core sets must be pairwise disjoint, and each split must hit its quota exactly. The configured eligible-pool target is 80,000, which is a packing margin, not a second sample size. Replace that target from the action-count audit if the 30,000/5,000/5,000 packing fails. Do not shrink the split after seeing validation or test metrics.

The original 1,400 parents may overlap this draw. That is acceptable because V2 never reads the locked test labels and reports a new test. Do not copy the locked checkpoint into V2 selection.

## Action cap

Raising `max_eligible_actions` does not create pendants. A molecule with two cuttable core attachments still has two actions. Many parents will have fewer than four. Molecules with more than the cap are truncated in the existing canonical order.

Do not switch to “exactly four” unless the audit is already frozen and the scientific claim is changed on purpose. Requiring four shifts the parent distribution toward heavily substituted molecules and drops two- and three-action parents only to simplify tensors. Tabular models score one action row at a time. The graph model batches variable-size graphs. Pad and mask if a constant width is needed.

Eligibility enumeration must report the pre-truncation count. The production manifest keeps today’s all-or-nothing SAFE round-trip: one failed action rejects the parent. Diagnostics may record per-action reasons (`ambiguous_element_multiset`, `missing_safe_fragment`, `missing_cut_piece`, `safe_decode_failed`, `roundtrip_failed`) without changing that inclusion rule.

**Cap rule, fixed before label generation and not tuned on policy performance:**

```text
Keep min_actions = 2.
Choose the smallest cap in {4, 6, 8} that retains at least 95% of verified
actions among parents that already have at least two, provided the resulting
label workload fits the GPU-hour cap below.
If none do, use 8 and report the truncation rate.
```

Broadening eligibility to internal side chains, ring edits, or multi-bond cuts is a different method. It is out of scope.

`scripts/audit_action_counts.py --config configs/experiment_v2.yaml` stops after 40,000 parents with at least `min_eligible_actions` verified pendants. It writes those parents to `artifacts_v2/eligible_parents.jsonl` and the cap summary to `artifacts_v2/action_count_audit.json`. Run it with real SAFE alignment. `--offline-no-safe` is not a basis for the cap. The script does not download chemreps; it reads the file already on that machine.

## Features

The locked 13 descriptors leave little that varies across actions of one parent. V2 tabular inputs add, in a fixed order stored in the checkpoint:

- fragment MolWt, LogP, TPSA, HBD, HBA, heteroatom count, formal charge, rotatable bonds, ring count, aromatic atom count, and heavy-atom count;
- fragment/parent fractions for heavy atoms, TPSA, and MolWt;
- core-side and fragment-side attachment atom: element, degree, total valence, formal charge, aromaticity, ring membership, hybridization, and hydrogen count;
- radius-2 Morgan bits centered on each attachment atom (128 bits each);
- the existing parent and core descriptors.

`bond_double` and `bond_aromatic` are normally zero because every eligible cut is a non-ring single bond. Keep them only for schema compatibility. Audit every column for zero variance on the training actions; drop nothing silently after test is open. `is_terminal` may also be constant under the current pendant definition.

Build the fragment from the persisted atom indices, without the dummy atom stored in `fragment_smiles`. Reject stale or impossible indices. Checkpoints store `feature_schema_version`, ordered `feature_names`, and `n_features`. Loading fails on a mismatch. `fragment_heavy_atoms` is a named field; largest-fragment selection must not index a hard-coded column.

## Models

One action table and one predictor interface serve every candidate. Each `predict_actions` call returns one score per manifest action, in that order. Evaluation uses the actions frozen in the manifest and does not rediscover them.

The candidate set is frozen before the new test is opened:

1. **Shallow MLP.** One hidden layer of width 64, dropout 0.15, weight decay `1e-4`, binomial NLL. Logits in the module; `sigmoid` only for probabilities. The locked two-hidden-layer `ValueMLP` stays available for the original protocol and is not the V2 candidate.
2. **LightGBM regressor.** Pointwise regression on `n_successes / n_trials` with sample weight `n_trials`. Clip outputs to `[0, 1]` for reporting. Selection uses rank order.
3. **LightGBM ranker.** LambdaRank with linear gain `0 .. n_trials`. Rows of one parent must be contiguous. Reject the fit if the rollout count is not shared by every train and validation action. Ranker scores are not probabilities; do not label them as such. Calibrate only on a held-out slice of training data if a probability is required.
4. **Action-conditioned GATv2.** A parent-only graph cannot tell which fragment is proposed. Each example is one parent/action graph: chemistry node features, both directed bond edges, a selected-cut flag on the attachment edge in both directions, nonempty core and fragment masks, and the two attachment atoms marked. Three GATv2 layers, four heads, 64 channels, residual LayerNorm, then mean pools of the whole molecule, the core, and the fragment, concatenated with the tabular vector, then a 64-unit head emitting one binomial logit. Record the trainable parameter count. The attachment masks and the selected-edge flag must actually enter the forward pass.

No training script may open `manifests_v2/test.jsonl`.

## Selection and the single test

On validation, group predictions by parent, choose the highest-scoring action, and record that action’s observed `n_successes / n_trials`. This `validation_top1_success_rate` is the primary selection metric. Also report binomial NLL and Brier score where the output is a probability, calibration, agreement with largest-fragment, success by number of actions, parameter count, and inference time.

Pick one learned policy with that rule. Freeze the checkpoint, the dependency lock, the test seed schedule, and the metric code. Then run the new test once: 5,000 parents, the learned policy plus random and largest-fragment, 16 completions each, same eligible action sets and the same completion budget.

The primary estimand remains the paired difference in 16-sample parent success between the chosen learned policy and largest-fragment, with a paired bootstrap 95% interval over parents. The success criterion is unchanged: the lower bound is greater than zero, and candidate validity does not fall by more than 3 percentage points versus the stronger baseline.

If every candidate is reported on the test set, declare all four as preregistered comparisons before the test opens and correct for four comparisons. Do not promote the best test result after looking.

## Cost

At the locked probe rate of 0.137 seconds per completion, 35,000 train and validation parents with four actions and eight labels each are 1.12 million completions, about 43 L4 GPU-hours before overhead. The test is `5,000 × 3 policies × 16 = 240,000` completions, about 9 GPU-hours, and does not grow with the number of unselected actions. Recompute both figures from the audit’s mean retained action count before launching label generation.

The study ceiling remains 180 GPU-hours and a hard stop at $185. If the audit’s label workload plus the 16-sample test and a confirmation pass does not fit, lower the action cap or the pool under the rule above. Do not switch to an A100 or train GenMol.

Phase 0’s throughput gate still applies to the frozen sampler before the V2 manifest is opened: at least 25% valid, at most 20 seconds per completion, and a 50-record core-mapping audit. The locked probe already passed that gate at 0.137 seconds and validity 0.992; repeat it only if the VM image, checkpoint, or decoding config changes.

## Tests on a GPU machine

There is no GPU on the machine where this plan was written. The tests below are part of the specification and are run on the GPU machine, which already has the chemreps file and the GenMol environment. Install optional model libraries there:

```bash
python -m pip install -e '.[dev,models]'
```

CPU checks, including graph construction and a GATv2 backward pass on CPU:

```bash
python -m pytest tests/test_labels.py tests/test_eligibility_counts.py tests/test_core_check.py tests/test_graph_model.py -q
```

CUDA checks (`tests/test_gpu_models.py` skips itself when CUDA is absent):

```bash
python -m pytest tests/test_gpu_models.py -q
```

That GPU file runs a binomial-likelihood backward, a short shallow-MLP training step, and a GATv2 training step on CUDA. `test_graph_model.py` checks two actions yield different graphs, mask and selected-edge invariants, a finite CPU backward, checkpoint reload, and inference without GenMol or CUDA. `torch-geometric` is an optional dependency; the GATv2 tests skip until `.[models]` is installed.

## Order of work

1. Keep V2 paths in `configs/experiment_v2.yaml`. Do not write V2 reports into `artifacts/`.
2. Run the action-count audit on the GPU machine and freeze `min_eligible_actions`, `max_eligible_actions`, `target_eligible_pool`, and seed `20261001`.
3. Build and hash `manifests_v2`. Core sets are pairwise disjoint and each split has its exact size. Every action round-trips and stores valid attachment indices.
4. Generate schema-v2 labels for train and validation only. Aggregation must reproduce `completion_success` on every stored row. Reward fields remain.
5. Fit the four candidates on train, select on validation only, and archive the chosen checkpoint with schema versions, feature names, manifest hashes, label hashes, config hash, seeds, package versions, hyperparameters, parameter count, and validation metrics.
6. Freeze that checkpoint and run the new test once. Random and largest-fragment use the same actions and the same 16 completions.

Minimum commands before labels, on the GPU machine:

```bash
python -m pytest tests/test_labels.py tests/test_eligibility_counts.py tests/test_graph_model.py tests/test_gpu_models.py -q
python scripts/audit_action_counts.py --config configs/experiment_v2.yaml
python scripts/prepare_manifest.py --config configs/experiment_v2.yaml
```

Stop if the audit cannot pack 40,000 core-disjoint parents, if any training script reads the V2 test manifest, or if spend reaches $185 before the pre-registered 16-sample test can finish. A failed packing or a null learned policy is a reportable result. It is not a reason to edit the locked protocol or to choose a model from the new test set.
