# Makefile for A10PED Neuromorphic - AI Tile v0
# Phase 1: Complete AI tile with PCIe + DDR4 + CSR + Memcopy

PROJECT = ai_tile_v0
QSYS_SYSTEM = ai_tile_v0_sys

.PHONY: all qsys build program clean help report

# Default target
all: build

# Generate Platform Designer system
qsys:
	@echo "Generating Platform Designer system..."
	qsys-script --script=create_qsys.tcl
	qsys-generate $(QSYS_SYSTEM).qsys --synthesis=VERILOG

# Build complete project
build: qsys
	@echo "Building Quartus project (this will take 1-2 hours)..."
	quartus_sh -t build.tcl

# Program FPGA via JTAG (FPGA 0)
program:
	@echo "Programming FPGA 0 via JTAG..."
	quartus_pgm -m jtag -o "p;output_files/$(PROJECT).sof@1"

# Program both FPGAs
program-dual:
	@echo "Programming both FPGAs via JTAG..."
	quartus_pgm -m jtag -o "p;output_files/$(PROJECT).sof@1"
	quartus_pgm -m jtag -o "p;output_files/$(PROJECT).sof@2"

# Show timing and resource reports
report:
	@echo "========================================="
	@echo " Timing Summary"
	@echo "========================================="
	@grep -A 20 "Timing Analyzer Summary" output_files/$(PROJECT).sta.rpt || echo "No timing report found"
	@echo ""
	@echo "========================================="
	@echo " Resource Utilization"
	@echo "========================================="
	@grep -A 30 "Fitter Resource Usage Summary" output_files/$(PROJECT).fit.summary || echo "No resource report found"

# Clean build artifacts
clean:
	rm -rf output_files/
	rm -rf db/
	rm -rf incremental_db/
	rm -rf $(QSYS_SYSTEM)/
	rm -rf $(QSYS_SYSTEM).sopcinfo
	rm -rf .qsys_edit/
	rm -rf simulation/
	rm -f *.rpt *.summary *.qws *.jdi *.sld *.stp

# Check JTAG cable
check-jtag:
	@echo "Checking JTAG cables..."
	jtagconfig

# Show help
help:
	@echo "A10PED Neuromorphic - AI Tile v0 Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all          - Build complete project (default)"
	@echo "  make qsys         - Generate Platform Designer system only"
	@echo "  make build        - Build Quartus project (qsys + synthesis + fit + asm)"
	@echo "  make program      - Program FPGA 0 via JTAG"
	@echo "  make program-dual - Program both FPGAs via JTAG"
	@echo "  make report       - Show timing and resource reports"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make check-jtag   - Check JTAG cable connection"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Example workflow:"
	@echo "  make clean        # Start fresh"
	@echo "  make build        # Build bitstream (1-2 hours)"
	@echo "  make report       # Check timing/resources"
	@echo "  make check-jtag   # Verify cable connection"
	@echo "  make program      # Program FPGA"
	@echo ""
	@echo "⚠️  Build time: 1-2 hours for Arria 10 GX1150"
