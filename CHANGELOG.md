# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive code documentation with numpydoc-style docstrings
- Architecture documentation explaining design decisions
- Contributing guidelines for contributors
- ReadTheDocs setup for hosted documentation
- Getting Started tutorial
- API reference documentation
- CHANGELOG.md for version history
- CONTRIBUTING.md for contribution guidelines

### Changed
- Improved inline code comments and examples
- Enhanced README with links to full documentation
- Restructured documentation for better organization
- Updated pyproject.toml with docs dependencies

## [0.1.0] - 2025-01-18

### Added
- Initial implementation of MPIPool class
- Support for master-worker parallel processing pattern
- Context manager support (`with MPIPool() as pool:`)
- Optional dill integration for complex object pickling
- Basic test infrastructure with pytest
- pyproject.toml with setuptools backend
- Comprehensive test suite (31 tests total)
- MIT License
- Enhanced README with usage examples
- GitHub repository setup
- Unit tests with proper MPI mocking (16 tests)
- Integration tests with real MPI (11 tests)
- Performance smoke tests (4 tests)
- Mock MPI infrastructure for fast unit testing

### Fixed
- Single process validation (raises ValueError)
- Import error handling with clear messages
- Worker process cleanup and termination
- Unit test mocking issues (13/16 tests were failing)
- MPI environment detection in pytest
- Integration test compatibility with pytest-mpi

## [Unreleased] - Planned Features
- [ ] GitHub Actions CI/CD integration
- [ ] Performance benchmarks and profiling
- [ ] Additional examples and tutorials
- [ ] Support for task cancellation
- [ ] Progress reporting callbacks
- [ ] Asynchronous task submission
- [ ] Batch task processing
- [ ] Dynamic worker pool resizing
