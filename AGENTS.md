# Repository rules

- Automated tests are unit tests or integration tests only. Do not add regression,
  smoke, or API test suites.
- Tests must execute real production code. Do not replace package modules,
  classes, or functions with mocks, stubs, spies, or duplicate implementations.
- Simulation is allowed for input data, noise, images, coordinates, and expected
  physical results. Reuse production models and algorithms where available.
- Keep the native ABI validation as a unit test against the compiled library.
- GPU integration tests require a real CUDA device; do not substitute a fake GPU
  backend or fitter.
- Builds always use the CUDA SDK declared in the package build dependencies.
  Never fall back to an existing system CUDA installation or CUDA path.
