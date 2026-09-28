# Reproducibility entry points

The public package distinguishes a figure-only audit from complete retraining.
The exact current-only V211 renderer snapshot is vendored under `figures/` and
`reproducibility/v211_nature_figures_20260906/`. Its 18-file SHA-256 manifest is
`data_manifest/v211_renderer_manifest.json`.

This boundary is intentional: the repository must not silently claim that a
clean clone can retrain restricted Google/Flickr/Street View inputs. Until the
derived source tables and Chicago geometry pass disclosure/access review,
public clean-clone support is limited to the fixed-order dry-run, manifest
verification, package checks and synthetic smoke contracts. The required
Nimbus Sans files are already hash-verified and separately licensed. Full
rendering and retraining remain parent-workspace operations.

The package also includes the exact PyTorch architecture contract used by V211
R3. Its frozen digest is recorded in the release manifest. This is a runnable
architecture-level contract, not the complete restricted-data training or
post-processing workflow.
