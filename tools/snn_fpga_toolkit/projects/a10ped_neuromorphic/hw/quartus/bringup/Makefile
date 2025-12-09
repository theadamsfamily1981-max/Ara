# Makefile for A10PED Neuromorphic - Bring-Up
# Milestone 1.2: JTAG-to-Avalon Bridge

PROJECT = bringup
QSYS_SYSTEM = bringup_system

.PHONY: all qsys build program test clean help

# Default target
all: build

# Generate Platform Designer system
qsys:
	@echo "Generating Platform Designer system..."
	qsys-script --script=create_qsys.tcl
	qsys-generate $(QSYS_SYSTEM).qsys --synthesis=VERILOG

# Build complete project
build: qsys
	@echo "Building Quartus project..."
	quartus_sh -t build.tcl

# Program FPGA via JTAG (FPGA 0 only)
program:
	@echo "Programming FPGA 0 via JTAG..."
	quartus_pgm -m jtag -o "p;output_files/$(PROJECT).sof@1"

# Program both FPGAs
program-dual:
	@echo "Programming both FPGAs via JTAG..."
	quartus_pgm -m jtag -o "p;output_files/$(PROJECT).sof@1"
	quartus_pgm -m jtag -o "p;output_files/$(PROJECT).sof@2"

# Run JTAG test script
test:
	@echo "Running JTAG RAM test..."
	cd ../../../sw/tools && python test_jtag_ram.py

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
	@echo "A10PED Neuromorphic - Bring-Up Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all          - Build complete project (default)"
	@echo "  make qsys         - Generate Platform Designer system only"
	@echo "  make build        - Build Quartus project (qsys + synthesis + fit + asm)"
	@echo "  make program      - Program FPGA 0 via JTAG"
	@echo "  make program-dual - Program both FPGAs via JTAG"
	@echo "  make test         - Run JTAG RAM test script"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make check-jtag   - Check JTAG cable connection"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Example workflow:"
	@echo "  make clean        # Start fresh"
	@echo "  make build        # Build bitstream (~20 min)"
	@echo "  make check-jtag   # Verify cable connection"
	@echo "  make program      # Program FPGA"
	@echo "  make test         # Run automated tests"
