# Local artifact notes

Large binaries and JSONL outputs are gitignored. Commit only:

- checksum files (`*.sha256`)
- small provenance / selection reports
- this README

Record GenMol checkpoint SHA-256, source commit, CUDA/PyTorch versions, and
`pip freeze` in `provenance.json` during Phase 0.
