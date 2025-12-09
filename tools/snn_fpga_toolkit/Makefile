# SNN Kernel Makefile
# Build system for SNN-optimized kernel modules

# Module name
MODULE_NAME := snn_kernel

# Kernel module object
obj-m := $(MODULE_NAME).o

# Source files - all subsystems
$(MODULE_NAME)-objs := \
	kernel/core/snn_core.o \
	kernel/pcie/snn_pcie.o \
	kernel/pcie/snn_fpga_hpc.o \
	kernel/memory/snn_memory.o \
	kernel/rt_sched/snn_rt_sched.o \
	kernel/cuda_bridge/snn_cuda_bridge.o \
	kernel/cuda_bridge/snn_cuda_hpc.o \
	kernel/nvme_dio/snn_nvme.o \
	kernel/snn_pipeline/snn_pipeline.o \
	kernel/semantic_ai/snn_ai_engine_v2.o \
	kernel/semantic_ai/snn_knowledge_graph.o \
	kernel/semantic_ai/snn_csr_graph.o \
	kernel/semantic_ai/snn_gnn.o \
	kernel/semantic_ai/snn_cold_start.o \
	kernel/observability/snn_hpc.o

# Keep v1 for comparison (commented out)
# kernel/semantic_ai/snn_ai_engine.o

# Kernel build directory
KDIR ?= /lib/modules/$(shell uname -r)/build

# Current directory
PWD := $(shell pwd)

# Compiler flags
ccflags-y := -I$(PWD)/include -DDEBUG -Wall -Werror
ccflags-y += -O2

# Build targets
all: modules api

# Build kernel modules
modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# Clean kernel modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	$(MAKE) -C api clean
	rm -f Module.symvers modules.order

# Install kernel modules
install: modules
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

# Uninstall kernel modules
uninstall:
	rm -f /lib/modules/$(shell uname -r)/extra/$(MODULE_NAME).ko
	depmod -a

# Load module
load:
	insmod $(MODULE_NAME).ko debug_level=2

# Unload module
unload:
	rmmod $(MODULE_NAME)

# Reload module (for development)
reload: unload load

# Build user-space API library
api:
	$(MAKE) -C api

# Build monitoring tools
tools:
	$(MAKE) -C tools

# Run tests (requires kernel module loaded)
test: modules
	$(MAKE) -C tests test

# Generate documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs/html
	doxygen Doxyfile 2>/dev/null || echo "Doxygen not installed"

# Format code (requires clang-format)
format:
	find kernel include -name '*.c' -o -name '*.h' | xargs clang-format -i

# Static analysis (requires sparse)
check:
	$(MAKE) -C $(KDIR) M=$(PWD) C=2 CF="-D__CHECK_ENDIAN__" modules

# Show module info
info:
	@echo "SNN Kernel Module Information:"
	@echo "  Module: $(MODULE_NAME).ko"
	@echo "  Kernel: $(shell uname -r)"
	@echo "  Architecture: $(shell uname -m)"
	@echo "  Sources:"
	@echo "$($(MODULE_NAME)-objs)" | tr ' ' '\n' | sed 's/^/    /'

# Help
help:
	@echo "SNN Kernel Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build kernel modules and API library"
	@echo "  modules   - Build kernel modules only"
	@echo "  api       - Build user-space API library"
	@echo "  tools     - Build monitoring/debugging tools"
	@echo "  clean     - Clean build artifacts"
	@echo "  install   - Install kernel modules (requires root)"
	@echo "  uninstall - Remove installed modules (requires root)"
	@echo "  load      - Load kernel module (requires root)"
	@echo "  unload    - Unload kernel module (requires root)"
	@echo "  reload    - Unload and reload module (dev only)"
	@echo "  test      - Run test suite"
	@echo "  docs      - Generate documentation"
	@echo "  format    - Format source code"
	@echo "  check     - Run static analysis"
	@echo "  info      - Show module information"
	@echo ""
	@echo "Variables:"
	@echo "  KDIR      - Kernel build directory (default: /lib/modules/\$$(uname -r)/build)"
	@echo ""
	@echo "Examples:"
	@echo "  make                  # Build everything"
	@echo "  make modules          # Build kernel modules"
	@echo "  sudo make install     # Install modules"
	@echo "  sudo make load        # Load module"

.PHONY: all modules clean install uninstall load unload reload api tools test docs format check info help
