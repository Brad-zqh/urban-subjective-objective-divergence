# V211 execution-closure audit — 2026-09-10

## Outcome

The exact PyTorch architecture contract used by the current V211 R3 proxy run
is portable and has been vendored. The complete input → training →
post-processing → validation chain is **not yet a clean-clone execution
closure**. Publishing the remaining files without adaptation would overstate
reproducibility.

This audit is code-path evidence only. It does not rerun training, change a
checkpoint or promote `proxy_analysis=true`, `formal=false` results.

## Verified portable component

`src/mm_gtgnnwr/model_contract.py` is byte-identical to the workspace source
`workstreams/RQ2_model/engineering_v207/v207_model_contract.py`. Both have
SHA-256
`30772daf98f31a65544858673900ca472c6ffefb95b96158c5e315387771a67c`.
The retained class name `V207TransparentMMGTGNNWR` records implementation
provenance; V211 bound this class to V211 inputs and outputs.

The release smoke tests verify exact additive prediction, NaN-safe missing
channel gating and equality between sparse zero-weight-padding aggregation and
the dense contract. They pass 3/3 in the project PyTorch environment. The
original source contract suite passes 8/8. That environment is Python 3.10.18
with PyTorch 2.12.1+cu130; the smoke tensors run on CPU. The release package
targets Python 3.11+, so a PyTorch-enabled 3.11/3.12 clean CI run is still a
separate gate.

## Why the remaining closure is blocked

| Component | Current workspace implementation | Release issue |
|---|---|---|
| Input assembly | Four V211 input/validation scripts set `ROOT` to a developer-specific absolute Windows workspace path | Must become config- or repository-relative and retain source hashes |
| R3 training | `engineering_v207/run_v207_matched_retraining_engineering_r1.py`, selected through `V207_*` environment variables | Needs a V211 release entry point, explicit schema/config validation and documented resume semantics |
| Training helpers | `run_v207_first_fold_e2e_engineering_r1.py` supplies relations, resource gates, quality construction and standardisation | Pulling only the runner would leave an incomplete import and data-contract closure |
| Post-processing | The V211 R3 wrapper dynamically loads three V208-named engines and rewrites `V208` strings in JSON after computation | Needs shared version-neutral engines or audited V211 adapters; string replacement is too brittle for a public API |
| Validation | The 11-check chain calls V211 wrappers plus a V208-named inequality validator | Must be staged as one dependency-complete validation closure |
| Data/runtime | Fold roles, six frozen 768-D channels, graph edges, outcome table and derived source tables remain outside the candidate | Each item needs schema, hash, licence/access status and fail-closed configuration |
| Checkpoints | Per-fold `model_state.pt` files are excluded by the release boundary | Retraining and checkpoint-based reproduction are separate tiers |

The four local-path files are:

- `build_multimodal_semantic_proxy_v1.py`;
- `validate_multimodal_semantic_proxy_v1.py`;
- `assemble_compact_input_candidate_v1.py`;
- `build_v211_compact_core_inputs_r1.py`.

## Relation representation requiring explicit treatment

The frozen training helper declares five relation slots. Queen, roads,
mobility and temporal-forward have candidate edges; transit has source `None`,
contributes no synthetic edge and is padded with zero weights for tensor-shape
compatibility. The sparse model removes zero-weight padding before gathering
sender context.

Public figures correctly show only the four implemented relations and label
transit unavailable. A release refactor must make this absence explicit rather
than presenting transit as a measured zero graph. Changing the trained model's
relation count or tensor layout would be a behavioural change and would require
a new matched run; renaming or path adaptation alone does not justify rerunning
the completed R3 results.

## Release-safe completion plan

1. Freeze a source-to-release file map and SHA-256 manifest for the dependency-
   complete runner, helpers, post-processing engines and validators.
2. Replace local absolute paths with a validated configuration whose missing
   restricted inputs fail closed.
3. Replace post-hoc `V208` string rewriting with version-neutral engine
   parameters while retaining regression tests against the sealed R3 metadata.
4. Add schema fixtures and synthetic smoke inputs; do not add raw platform
   media, resident-level records or checkpoints.
5. Compare adapted outputs with sealed artefacts. If code semantics are
   unchanged, no scientific rerun is needed. If relation layout, numeric
   standardisation, fold seeds or model computation changes, create a new run
   version rather than overwriting R3.

Until these steps pass, the repository supports architecture and figure-code
audit tiers, not full clean-clone retraining.
