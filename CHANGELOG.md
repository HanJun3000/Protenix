# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Structure scoring through a job-level `scoreStructure` field. Protenix computes
  its confidence metrics (pLDDT, PAE, pTM/ipTM, clash flag, ranking score) for
  coordinates from a local protein PDB or mmCIF file instead of a diffusion
  sample. Diffusion is skipped when the file covers every model atom; otherwise
  missing atoms are taken from a superposed diffusion sample. Protein chains
  only. See `docs/infer_json_format.md` and
  `examples/example_score_structure/`.
  - New modules `protenix/data/template/custom_structure.py` (reads protein
    chains from a local PDB/mmCIF) and
    `protenix/data/inference/score_structure.py` (maps them onto the model
    atoms), with `tests/test_custom_structure.py` and
    `tests/test_score_structure.py`.
  - MC dropout is disabled for scored jobs so that scores are reproducible for
    a fixed seed; normal predictions are unchanged.
- Test cases for installation and compatibility issues by Shad Nygren (@ShadNygren)
  - `test_installation.py` - Tests for DeepSpeed/Pydantic compatibility (issue #182)
  - `test_triton_compatibility.py` - Tests for Triton GPU compatibility (issue #185)
  - `test_esm_loading.py` - Tests for ESM weights loading with PyTorch 2.6+ (issue #176)
- CHANGELOG.md to track project changes (Apache 2.0 compliance)

### Fixed
- Issue #182: DeepSpeed and Pydantic v2.x compatibility by upgrading to DeepSpeed 0.17.5
- Issue #185: Triton kernel compatibility with consumer GPUs (RTX 3090/4090)
  - Added PyTorch fallback when Triton is unavailable or unsupported
  - Enables Protenix to run on consumer hardware without code changes
  - Modified `protenix/model/tri_attention/__init__.py` to provide transparent fallback
- Inference robustness:
  - Fixed invalid MSA success assertion in FASTA flow (`runner/batch_inference.py`).
  - Added explicit input JSON validation for non-empty top-level list (`runner/inference.py`).
  - Hardened model-name parsing to avoid crashes on unexpected naming (`runner/inference.py`, `runner/batch_inference.py`).

### Contributors
- Shad Nygren, Virtual Hipster Corporation (@ShadNygren)

## [v0.6.1] - 2025-08-20

### Fixed
- ESM models loading in PyTorch 2.6 and newer versions (commit dbbee14)

## [v0.6.0] - Previous releases

See commit history for earlier changes.
