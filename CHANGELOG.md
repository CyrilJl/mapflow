# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Improved project metadata, distribution contents, CI coverage, and release security.
- Made tests deterministic and independent of live tutorial-data downloads.
- Deferred the FFmpeg availability check until animation creation.

## [0.3.0] - 2026-07-05

### Added

- Streamed lazy and temporally interpolated frames instead of materializing complete animations in memory.
- Added configurable target video width.

[Unreleased]: https://github.com/CyrilJl/mapflow/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/CyrilJl/mapflow/releases/tag/v0.3.0
