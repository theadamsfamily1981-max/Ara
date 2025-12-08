# ============================================================================
# CorrSpike-HDC HLS Build Makefile
# ============================================================================
#
# Targets:
#   make sim      - Run C simulation (requires Vitis HLS)
#   make synth    - Run HLS synthesis
#   make tb       - Build standalone testbench (g++)
#   make test     - Run standalone testbench
#   make clean    - Clean build artifacts
#
# ============================================================================

# Tool paths (adjust for your environment)
VITIS_HLS ?= vitis_hls
CXX ?= g++

# Source files
SRCS = corr_spike_hdc.cpp
HDRS = corr_spike_hdc.h
TB_SRC = corr_spike_hdc_tb.cpp

# Output
TB_EXE = corr_spike_tb

# HLS project
HLS_PROJ = corr_spike_hdc_proj

# Compiler flags for standalone testbench
# Note: We need to mock ap_int.h for standalone compilation
CXXFLAGS = -std=c++17 -O2 -Wall -DSTANDALONE_SIM

# ============================================================================
# Targets
# ============================================================================

.PHONY: all sim synth tb test clean help

all: help

help:
	@echo "CorrSpike-HDC HLS Build System"
	@echo "=============================="
	@echo ""
	@echo "Targets:"
	@echo "  make sim      - Run C simulation (requires Vitis HLS)"
	@echo "  make synth    - Run HLS synthesis (requires Vitis HLS)"
	@echo "  make tb       - Build standalone testbench"
	@echo "  make test     - Run standalone testbench"
	@echo "  make clean    - Clean build artifacts"
	@echo ""
	@echo "Environment:"
	@echo "  VITIS_HLS = $(VITIS_HLS)"
	@echo "  CXX       = $(CXX)"

# ============================================================================
# HLS Targets (require Xilinx tools)
# ============================================================================

sim: $(SRCS) $(HDRS) $(TB_SRC)
	@echo "Running C Simulation..."
	$(VITIS_HLS) -f run_hls.tcl 2>&1 | tee sim.log
	@echo "Done. See sim.log for details."

synth: $(SRCS) $(HDRS)
	@echo "Running HLS Synthesis..."
	$(VITIS_HLS) -f run_hls.tcl 2>&1 | tee synth.log
	@echo "Done. See synth.log for details."

# ============================================================================
# Standalone Testbench (for development without Xilinx tools)
# ============================================================================

# Generate mock ap_int.h for standalone compilation
ap_mock.h:
	@echo "Generating ap_int mock header..."
	@echo '// Mock ap_int.h for standalone compilation' > ap_mock.h
	@echo '#ifndef AP_INT_H' >> ap_mock.h
	@echo '#define AP_INT_H' >> ap_mock.h
	@echo '#include <cstdint>' >> ap_mock.h
	@echo 'template<int W> class ap_int { public: int64_t val; ap_int() : val(0) {} ap_int(int64_t v) : val(v) {} operator int64_t() const { return val; } ap_int operator+(const ap_int& o) const { return ap_int(val + o.val); } ap_int operator-(const ap_int& o) const { return ap_int(val - o.val); } ap_int operator*(const ap_int& o) const { return ap_int(val * o.val); } ap_int& operator+=(const ap_int& o) { val += o.val; return *this; } ap_int& operator-=(const ap_int& o) { val -= o.val; return *this; } ap_int operator-() const { return ap_int(-val); } bool operator>(const ap_int& o) const { return val > o.val; } bool operator<(const ap_int& o) const { return val < o.val; } bool operator>=(const ap_int& o) const { return val >= o.val; } bool operator==(const ap_int& o) const { return val == o.val; } bool operator!=(const ap_int& o) const { return val != o.val; } };' >> ap_mock.h
	@echo 'template<int W> class ap_uint { public: uint64_t val; ap_uint() : val(0) {} ap_uint(uint64_t v) : val(v) {} operator uint64_t() const { return val; } ap_uint operator+(const ap_uint& o) const { return ap_uint(val + o.val); } ap_uint& operator|=(uint64_t v) { val |= v; return *this; } };' >> ap_mock.h
	@echo 'template<int W, int I> class ap_fixed { public: double val; ap_fixed() : val(0) {} ap_fixed(double v) : val(v) {} operator double() const { return val; } ap_fixed operator*(const ap_fixed& o) const { return ap_fixed(val * o.val); } ap_fixed operator+(const ap_fixed& o) const { return ap_fixed(val + o.val); } };' >> ap_mock.h
	@echo '#endif' >> ap_mock.h

# Build standalone testbench
$(TB_EXE): $(SRCS) $(TB_SRC) $(HDRS) ap_mock.h
	@echo "Building standalone testbench..."
	$(CXX) $(CXXFLAGS) -include ap_mock.h -o $(TB_EXE) $(SRCS) $(TB_SRC)

tb: $(TB_EXE)

# Run standalone testbench
test: $(TB_EXE)
	@echo "Running standalone testbench..."
	./$(TB_EXE)

# ============================================================================
# Clean
# ============================================================================

clean:
	rm -rf $(HLS_PROJ)
	rm -rf *.log
	rm -rf $(TB_EXE)
	rm -rf ap_mock.h
	rm -rf corr_spike_hdc_ip.zip
	rm -rf .Xil
	rm -rf vitis_hls.log

