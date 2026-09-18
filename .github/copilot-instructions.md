# Cheetah Accelerator Physics Simulation Library

Cheetah is a fast and differentiable particle accelerator optics simulation library built with PyTorch. Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Bootstrap, Build, and Test the Repository
- **Install in editable mode**: `pip install -e .` -- takes 2-3 minutes. NEVER CANCEL. Set timeout to 5+ minutes.
- **Install test dependencies**: `pip install pytest pytest-benchmark pytest-cov` -- takes 30-60 seconds.
- **Install formatting tools**: `pip install black isort flake8 flake8-bugbear` -- takes 30 seconds.
- **Install pre-commit hooks**: `pip install pre-commit && pre-commit install` -- takes 30 seconds.

### Running Tests
- **Core tests (recommended)**: `pytest tests/test_drift.py tests/test_quadrupole.py tests/test_cavity.py tests/test_elements.py --benchmark-skip` -- takes 5-10 seconds. NEVER CANCEL. Set timeout to 2+ minutes.
- **All compatible tests**: `pytest --benchmark-skip` excludes a few tests that require external dependencies like `ocelot-collab`.
- **Benchmark tests**: `pytest --benchmark-only` -- runs performance benchmarks. Takes 30-60 seconds.
- **Tests with dependencies**: Files `test_compare_ocelot.py`, `test_ocelot_import.py`, and `test_sextupole.py` require `ocelot-collab` package.

### Code Formatting and Linting
- **Black formatting**: `black --check cheetah/ tests/ docs/ setup.py --exclude="/.ipynb/"` -- takes 2-3 seconds. To apply changes: `black cheetah/ tests/ docs/ setup.py --exclude="/.ipynb/"`
- **Import sorting**: `isort --check --diff --profile black .` -- takes <1 second. To apply changes: `isort . --profile black`
- **Linting**: `flake8 .` -- takes 1-2 seconds.
- **Pre-commit hooks**: `pre-commit run --all-files` -- runs all formatting checks together.

### Documentation
- **Install docs dependencies**: `cd docs && pip install -r requirements.txt` -- takes 2-3 minutes.
- **Build documentation**: `cd docs && make html` -- takes 8-10 seconds. Note: May fail due to missing pandoc but core sphinx functionality works.
- **Clean docs**: `cd docs && make clean` -- removes built documentation.

## Validation

### Manual Validation Scenarios
ALWAYS manually validate any new code with these scenarios after making changes:

#### Basic Functionality Test
```python
import cheetah
import torch

# Create a realistic beamline
elements = [
    cheetah.Drift(length=torch.tensor(0.175)),
    cheetah.Quadrupole(length=torch.tensor(0.122), k1=torch.tensor(2.5), name='QUAD1'),
    cheetah.Drift(length=torch.tensor(0.428)),
    cheetah.Dipole(length=torch.tensor(0.2), angle=torch.tensor(0.1), name='BEND1'),
    cheetah.Drift(length=torch.tensor(0.3)),
    cheetah.BPM(name='BPM1'),
    cheetah.Screen(name='SCREEN1'),
]
segment = cheetah.Segment(elements=elements)

# Create realistic particle beam
particles = cheetah.ParticleBeam.from_parameters(
    num_particles=int(5e3),
    energy=torch.tensor(1e9),  # 1 GeV
    sigma_x=torch.tensor(100e-6),
    sigma_y=torch.tensor(100e-6),
    sigma_px=torch.tensor(10e-6),
    sigma_py=torch.tensor(10e-6),
)

# Track beam through beamline
tracked = segment.track(particles)
print(f'Final beam sigma_x: {tracked.sigma_x.item()*1e6:.1f} μm')

# Test element access by name
quad = segment.QUAD1
print(f'Accessed quadrupole: {quad.name}')

# Test transfer matrix
transfer_matrix = segment.first_order_transfer_map(energy=particles.energy, species=particles.species)
print(f'Transfer matrix shape: {transfer_matrix.shape}')
```

#### CI/Pre-commit Validation
Always run these commands before committing to ensure CI will pass:
- `black cheetah/ tests/ docs/ setup.py --exclude="/.ipynb/"`
- `isort . --profile black`
- `flake8 .`
- `pytest tests/test_drift.py tests/test_quadrupole.py tests/test_cavity.py --benchmark-skip`

### You CAN build and run the code
- The build succeeds with `pip install -e .`
- Core functionality works without external accelerator physics libraries
- Tests run successfully with pytest
- Formatting tools work correctly
- Documentation builds with sphinx (may have warnings about missing pandoc)

## Common Tasks

### Repository Structure
```
cheetah/                  # Main package directory
├── accelerator/          # Core accelerator physics elements
├── particles/            # Particle beam classes
├── converters/           # Converters for other codes (OCELOT, etc.)
├── utils/               # Utility functions
└── track_methods.py     # Tracking method implementations

tests/                   # Test suite
├── test_drift.py        # Drift element tests (no dependencies)
├── test_quadrupole.py   # Quadrupole tests (no dependencies)  
├── test_cavity.py       # Cavity tests (no dependencies)
├── test_elements.py     # General element tests (no dependencies)
└── test_*_ocelot.py     # Tests requiring ocelot-collab

docs/                    # Sphinx documentation
├── examples/            # Jupyter notebook examples
├── requirements.txt     # Documentation dependencies
└── Makefile            # Documentation build commands
```

### Key Package Dependencies
- **Core dependencies**: torch, numpy, scipy, matplotlib (automatically installed)
- **Optional dependencies**: ocelot-collab, openpmd-beamphysics (for specialized tests/conversions)
- **Development dependencies**: pytest, black, isort, flake8 (manual installation)
- **Documentation dependencies**: sphinx, furo, nbsphinx (manual installation)

### Working with Elements
- **Basic elements**: Drift, Quadrupole, Dipole, BPM, Screen, Cavity
- **Element access**: Use `segment.ELEMENT_NAME` to access elements by name
- **Tracking methods**: 'linear', 'second_order', 'drift_kick_drift'
- **Vector dimensions**: Elements support batch processing with multiple beam parameters

### Working with Beams
- **ParticleBeam**: Main beam class with torch tensors for particle coordinates
- **ParameterBeam**: Gaussian beam parameterization
- **Beam creation**: Use `cheetah.ParticleBeam.from_parameters()` for typical beam creation
- **Beam tracking**: Use `segment.track(beam)` to track through beamline

### Performance Considerations
- Cheetah is designed for GPU acceleration with PyTorch
- Use float32 for memory efficiency, float64 for precision
- Vectorized operations are preferred over loops
- Benchmark tests measure tracking performance

### Known Issues and Workarounds
- Some documentation builds require pandoc (external dependency)
- Tests requiring `ocelot-collab` may fail in environments without it -- document as "tests skipped due to missing ocelot dependency"
- GPU acceleration requires CUDA-enabled PyTorch installation
- Screen readings require beam tracking to be active
- **Network issues**: `pip install` may occasionally timeout due to PyPI connectivity. If installation fails with timeout errors, retry the command -- this is usually temporary
- **Pre-commit hooks**: May fail with network timeouts when installing dependencies. Use `git commit --no-verify` to bypass hooks temporarily during network issues
- **Virtual environments**: When testing in fresh virtual environments, ensure adequate timeout values for pip operations

## Timing Expectations
- **Package installation**: 2-3 minutes (NEVER CANCEL - Set 5+ minute timeout)
- **Core test suite**: 5-10 seconds (NEVER CANCEL - Set 2+ minute timeout)  
- **Full test suite**: 30-60 seconds (NEVER CANCEL - Set 5+ minute timeout)
- **Code formatting**: 1-3 seconds total (black + isort + flake8)
- **Documentation build**: 8-10 seconds (may have warnings)
- **Pre-commit hooks**: 5-15 seconds depending on scope

## Development Workflow
1. **Setup**: `pip install -e . && pip install pytest black isort flake8`
2. **Development**: Make changes to code
3. **Format**: `black cheetah/ tests/ docs/ setup.py --exclude="/.ipynb/" && isort . --profile black`
4. **Test**: `pytest tests/test_drift.py tests/test_quadrupole.py tests/test_cavity.py --benchmark-skip`
5. **Lint**: `flake8 .`
6. **Manual validation**: Run the validation scenario above
7. **Commit**: Changes should now pass CI

Always test your changes with the manual validation scenario and ensure core tests pass before committing.