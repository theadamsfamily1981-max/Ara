# Contributing to SNN Kernel

Thank you for your interest in contributing to the SNN-Optimized Kernel project!

## Getting Started

### Development Environment

Required tools:
- Linux kernel 6.0+ source
- GCC 11+ or Clang 14+
- Kernel development packages
- Git
- CUDA Toolkit (for GPU support)
- FPGA vendor tools (Xilinx Vivado/Intel Quartus)

### Building from Source

```bash
git clone <repository-url>
cd mcp
make
```

## Coding Standards

### Kernel Code

Follow Linux kernel coding style:
- Indentation: Tabs (8 spaces)
- Line length: 80 characters (prefer), 100 max
- Function declarations: Return type on same line
- Comments: C-style /* */ for multi-line

Run before committing:
```bash
make format  # Format code
make check   # Static analysis
```

### User-Space Code

- Indentation: 4 spaces
- Line length: 100 characters
- Follow C11 standard

## Contribution Process

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/snn-kernel
cd snn-kernel
git remote add upstream <original-repo-url>
```

### 2. Create Feature Branch

```bash
git checkout -b feature/my-feature
```

### 3. Make Changes

- Write clean, documented code
- Add tests for new features
- Update documentation
- Follow coding standards

### 4. Test

```bash
# Build
make clean && make

# Load module
sudo insmod snn_kernel.ko

# Run tests
make test

# Test example
cd examples && make && sudo ./simple_snn
```

### 5. Commit

Follow conventional commit format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Build/tooling changes

Example:
```bash
git commit -m "feat(pcie): add PCIe 6.0 support

Implements PCIe 6.0 protocol extensions for higher bandwidth.
Maintains backward compatibility with PCIe 5.0.

Closes #123"
```

### 6. Push and Create PR

```bash
git push origin feature/my-feature
```

Then create a Pull Request on GitHub.

## Areas for Contribution

### High Priority

1. **FPGA Integration**
   - Xilinx Alveo drivers
   - Intel Stratix drivers
   - Custom SNN IP cores

2. **Performance Optimization**
   - P2P transfer optimization
   - Memory bandwidth improvements
   - RT scheduler enhancements

3. **Hardware Support**
   - Additional GPU vendors (AMD)
   - More FPGA platforms
   - PCIe 6.0 support

### Medium Priority

4. **Features**
   - Multi-GPU support
   - Network training support (RDMA)
   - Power management
   - Auto-tuning

5. **Documentation**
   - More examples
   - Tutorial videos
   - Hardware compatibility matrix

6. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks

### Help Wanted

- Hardware testing on different platforms
- Documentation improvements
- Bug reports and fixes
- Performance profiling

## Testing Guidelines

### Unit Tests

Add tests in `tests/` directory:

```c
// tests/test_memory.c
#include "test_framework.h"

TEST(memory_alloc_free) {
    void *ptr = snn_alloc_pinned(4096, SNN_MEM_GPU);
    ASSERT_NOT_NULL(ptr);
    snn_free_pinned(ptr);
    PASS();
}
```

### Integration Tests

Test complete workflows:
- Memory allocation → P2P transfer → Computation
- NVMe read → SNN inference → Results write

### Performance Tests

Benchmark critical paths:
- P2P bandwidth
- Memory allocation latency
- RT scheduling latency
- End-to-end inference time

## Code Review Process

All contributions go through code review:

1. **Automated Checks**
   - Build verification
   - Static analysis
   - Style checking

2. **Manual Review**
   - Code quality
   - Performance impact
   - Documentation
   - Testing coverage

3. **Approval**
   - Requires 2 approvals from maintainers
   - All CI checks must pass

## Security

### Reporting Vulnerabilities

**DO NOT** open public issues for security vulnerabilities.

Email: security@snn-kernel.org

Include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Security Guidelines

- Validate all user inputs
- Check buffer boundaries
- Use safe string functions
- Avoid integer overflows
- Handle errors properly
- Document security implications

## Documentation

### Code Documentation

Use Doxygen-style comments:

```c
/**
 * snn_p2p_transfer - Perform peer-to-peer transfer
 * @mgr: PCIe manager
 * @transfer: Transfer descriptor
 *
 * Initiates a peer-to-peer DMA transfer between devices.
 * The transfer can be synchronous or asynchronous based on flags.
 *
 * Return: 0 on success, negative error code on failure
 */
int snn_p2p_transfer(struct snn_pcie_manager *mgr,
                     snn_p2p_transfer_t *transfer);
```

### User Documentation

Update relevant docs in `docs/`:
- README.md for overview changes
- API_GUIDE.md for API changes
- PERFORMANCE_TUNING.md for optimization tips
- ARCHITECTURE.md for design changes

## Performance Considerations

### Critical Paths

Optimize these code paths:
- P2P transfers (< 2μs latency target)
- Memory allocation (< 100μs target)
- RT task switching (< 10μs target)

### Profiling

Use these tools:
```bash
# Kernel profiling
perf record -g ./your_test
perf report

# GPU profiling
nsys profile ./your_app
ncu --set full ./your_app
```

### Benchmarking

Always benchmark before/after:
```bash
# Run standard benchmark suite
make benchmark

# Compare results
./scripts/compare_benchmarks.sh before.json after.json
```

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Questions?

- GitHub Issues: For bugs and feature requests
- GitHub Discussions: For questions and discussions
- Email: dev@snn-kernel.org

## Acknowledgments

We appreciate all contributions, whether code, documentation, bug reports, or suggestions!

Contributors are recognized in:
- README.md
- CONTRIBUTORS file
- Release notes

Thank you for helping make SNN Kernel better!
