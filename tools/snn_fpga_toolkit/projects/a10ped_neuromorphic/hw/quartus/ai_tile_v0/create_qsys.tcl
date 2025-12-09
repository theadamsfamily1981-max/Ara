#!/usr/bin/env qsys-script
#
# Platform Designer System Creation Script
# A10PED Neuromorphic - AI Tile v0
#
# Creates a complete AI tile with:
#   - PCIe Gen3 x8 Hard IP (Avalon-MM DMA)
#   - DDR4 EMIF Controller (8GB)
#   - AI CSR block
#   - Memcopy DMA kernel
#   - Clock and reset infrastructure
#
# Usage:
#   qsys-script --script=create_qsys.tcl
#
# Author: A10PED Neuromorphic Project
# License: BSD-3-Clause
#

package require qsys

# Create new system
create_system ai_tile_v0_sys

# Set device family
set_project_property DEVICE_FAMILY "Arria 10"
set_project_property DEVICE 10AX115N2F40E2LG

#
# Clock and Reset Infrastructure
#

# PCIe reference clock (100 MHz external)
add_instance pcie_refclk clock_source
set_instance_parameter_value pcie_refclk clockFrequency 100000000
set_instance_parameter_value pcie_refclk clockFrequencyKnown true
set_instance_parameter_value pcie_refclk resetSynchronousEdges DEASSERT

# Global reset source
add_instance global_reset reset_source
set_instance_parameter_value global_reset associatedClock pcie_refclk
set_instance_parameter_value global_reset synchronousEdges DEASSERT

#
# PCIe Gen3 x8 Hard IP with Avalon-MM DMA
#

add_instance pcie_hip altera_pcie_a10_hip
set_instance_parameter_value pcie_hip pcie_mode "Shared Mode"
set_instance_parameter_value pcie_hip pcie_spec_version "3.0"
set_instance_parameter_value pcie_hip lane_rate "Gen3 (8.0 Gbps)"
set_instance_parameter_value pcie_hip lane_mask "x8"
set_instance_parameter_value pcie_hip port_type "Native Endpoint"
set_instance_parameter_value pcie_hip bar_size_bar0 20
set_instance_parameter_value pcie_hip bar_size_bar2 28
set_instance_parameter_value pcie_hip bar_type_bar0 "32-bit non-prefetchable memory"
set_instance_parameter_value pcie_hip bar_type_bar2 "64-bit prefetchable memory"
set_instance_parameter_value pcie_hip enable_dma true
set_instance_parameter_value pcie_hip dma_width 256
set_instance_parameter_value pcie_hip max_payload_size 256
set_instance_parameter_value pcie_hip max_read_request_size 512

# PCIe Avalon-MM DMA bridge
add_instance pcie_dma altera_pcie_av_mm_bridge
set_instance_parameter_value pcie_dma DATA_WIDTH 256
set_instance_parameter_value pcie_dma ADDRESS_WIDTH 64
set_instance_parameter_value pcie_dma USE_DESCRIPTOR true
set_instance_parameter_value pcie_dma MAX_BURST_SIZE 16

#
# DDR4 EMIF Controller (8GB SO-DIMM)
#

add_instance ddr4_emif altera_emif_a10_hps
set_instance_parameter_value ddr4_emif MEM_DDR4_FORMAT_ENUM "MEM_FORMAT_UDIMM"
set_instance_parameter_value ddr4_emif MEM_DDR4_TCL 18
set_instance_parameter_value ddr4_emif MEM_DDR4_WTCL 14
set_instance_parameter_value ddr4_emif MEM_DDR4_ROW_ADDR_WIDTH 16
set_instance_parameter_value ddr4_emif MEM_DDR4_COL_ADDR_WIDTH 10
set_instance_parameter_value ddr4_emif MEM_DDR4_BANK_ADDR_WIDTH 2
set_instance_parameter_value ddr4_emif MEM_DDR4_BANK_GROUP_WIDTH 2
set_instance_parameter_value ddr4_emif MEM_DDR4_DM_EN true
set_instance_parameter_value ddr4_emif MEM_DDR4_READ_DBI true
set_instance_parameter_value ddr4_emif MEM_DDR4_WRITE_DBI false
set_instance_parameter_value ddr4_emif MEM_DDR4_DATA_WIDTH 72
set_instance_parameter_value ddr4_emif MEM_DDR4_SPEEDBIN_ENUM "DDR4-2400"
set_instance_parameter_value ddr4_emif MEM_DDR4_TIMING_ENABLE_USER_MODE false
set_instance_parameter_value ddr4_emif PHY_DDR4_MEM_CLK_FREQ_MHZ 1200.0
set_instance_parameter_value ddr4_emif PHY_DDR4_DEFAULT_REF_CLK_FREQ false
set_instance_parameter_value ddr4_emif PHY_DDR4_USER_REF_CLK_FREQ_MHZ 100.0

# DDR4 Avalon-MM bridge (for DMA access)
add_instance ddr4_bridge altera_avalon_mm_bridge
set_instance_parameter_value ddr4_bridge DATA_WIDTH 512
set_instance_parameter_value ddr4_bridge ADDRESS_WIDTH 30
set_instance_parameter_value ddr4_bridge USE_AUTO_ADDRESS_WIDTH true
set_instance_parameter_value ddr4_bridge MAX_BURST_SIZE 16
set_instance_parameter_value ddr4_bridge MAX_PENDING_RESPONSES 16

#
# AI CSR Block (generated from YAML)
#

# Wrapper to expose ai_csr as Avalon-MM slave
add_instance ai_csr_wrapper altera_avalon_mm_slave_bfm
set_instance_parameter_value ai_csr_wrapper AV_ADDRESS_W 12
set_instance_parameter_value ai_csr_wrapper AV_SYMBOL_W 8
set_instance_parameter_value ai_csr_wrapper AV_NUMSYMBOLS 4
set_instance_parameter_value ai_csr_wrapper USE_READ true
set_instance_parameter_value ai_csr_wrapper USE_WRITE true
set_instance_parameter_value ai_csr_wrapper USE_WAITREQUEST true

# Note: The actual ai_csr.v will be instantiated in the top-level RTL
# This is a placeholder for the CSR address space in Platform Designer

#
# Memcopy DMA Kernel (SNN core stub)
#

add_instance memcopy_kernel altera_avalon_mm_master_bfm
set_instance_parameter_value memcopy_kernel AV_ADDRESS_W 64
set_instance_parameter_value memcopy_kernel AV_SYMBOL_W 8
set_instance_parameter_value memcopy_kernel AV_NUMSYMBOLS 32
set_instance_parameter_value memcopy_kernel USE_READ true
set_instance_parameter_value memcopy_kernel USE_WRITE true
set_instance_parameter_value memcopy_kernel USE_WAITREQUEST true
set_instance_parameter_value memcopy_kernel MAX_BURST_SIZE 16

# Note: The actual memcopy_kernel.v will be instantiated in the top-level RTL

#
# On-Chip RAM for high-bandwidth scratchpad
#

add_instance onchip_scratch altera_avalon_onchip_memory2
set_instance_parameter_value onchip_scratch allowInSystemMemoryContentEditor false
set_instance_parameter_value onchip_scratch blockType "AUTO"
set_instance_parameter_value onchip_scratch dataWidth 512
set_instance_parameter_value onchip_scratch dualPort false
set_instance_parameter_value onchip_scratch initMemContent false
set_instance_parameter_value onchip_scratch memorySize 65536.0
set_instance_parameter_value onchip_scratch readDuringWriteMode "DONT_CARE"
set_instance_parameter_value onchip_scratch writable true
set_instance_parameter_value onchip_scratch ecc_enabled false

#
# Clock and Reset Connections
#

# Connect all clocks to PCIe reference clock (will be derived by PCIe HIP)
add_connection pcie_refclk.clk pcie_hip.refclk
add_connection pcie_hip.coreclkout_hip pcie_dma.clk
add_connection pcie_hip.coreclkout_hip ddr4_bridge.clk
add_connection pcie_hip.coreclkout_hip ai_csr_wrapper.clk
add_connection pcie_hip.coreclkout_hip memcopy_kernel.clk
add_connection pcie_hip.coreclkout_hip onchip_scratch.clk1

# DDR4 uses separate PLL-generated clock
add_connection ddr4_emif.emif_usr_clk ddr4_emif.ctrl_amm_clk

# Connect resets
add_connection global_reset.reset pcie_hip.npor
add_connection pcie_hip.reset_status pcie_dma.reset
add_connection pcie_hip.reset_status ddr4_bridge.reset
add_connection pcie_hip.reset_status ai_csr_wrapper.reset
add_connection pcie_hip.reset_status memcopy_kernel.reset
add_connection pcie_hip.reset_status onchip_scratch.reset1
add_connection global_reset.reset ddr4_emif.global_reset_n

#
# Avalon-MM Interconnect
#

# PCIe DMA master → Memory subsystem
add_connection pcie_dma.master ddr4_bridge.s0
set_connection_parameter_value pcie_dma.master/ddr4_bridge.s0 arbitrationPriority 1
set_connection_parameter_value pcie_dma.master/ddr4_bridge.s0 baseAddress 0x0000000000000000
set_connection_parameter_value pcie_dma.master/ddr4_bridge.s0 defaultConnection 1

# PCIe DMA master → On-chip scratchpad
add_connection pcie_dma.master onchip_scratch.s1
set_connection_parameter_value pcie_dma.master/onchip_scratch.s1 arbitrationPriority 1
set_connection_parameter_value pcie_dma.master/onchip_scratch.s1 baseAddress 0x00000000F0000000
set_connection_parameter_value pcie_dma.master/onchip_scratch.s1 defaultConnection 0

# PCIe → AI CSR (via PCIe BAR0)
add_connection pcie_hip.rxm_bar0 ai_csr_wrapper.avs
set_connection_parameter_value pcie_hip.rxm_bar0/ai_csr_wrapper.avs arbitrationPriority 1
set_connection_parameter_value pcie_hip.rxm_bar0/ai_csr_wrapper.avs baseAddress 0x0000
set_connection_parameter_value pcie_hip.rxm_bar0/ai_csr_wrapper.avs defaultConnection 0

# Memcopy kernel master → DDR4 (for reads)
add_connection memcopy_kernel.master ddr4_bridge.s0
set_connection_parameter_value memcopy_kernel.master/ddr4_bridge.s0 arbitrationPriority 2
set_connection_parameter_value memcopy_kernel.master/ddr4_bridge.s0 baseAddress 0x0000000000000000

# DDR4 bridge → EMIF controller
add_connection ddr4_bridge.m0 ddr4_emif.ctrl_amm_0
set_connection_parameter_value ddr4_bridge.m0/ddr4_emif.ctrl_amm_0 arbitrationPriority 1
set_connection_parameter_value ddr4_bridge.m0/ddr4_emif.ctrl_amm_0 baseAddress 0x0000
set_connection_parameter_value ddr4_bridge.m0/ddr4_emif.ctrl_amm_0 defaultConnection 0

#
# Export Interfaces
#

# PCIe SerDes pins (connected at top level)
add_interface pcie_hip_serial conduit end
set_interface_property pcie_hip_serial EXPORT_OF pcie_hip.hip_serial

# PCIe control signals
add_interface pcie_refclk clock sink
set_interface_property pcie_refclk EXPORT_OF pcie_refclk.clk_in
add_interface pcie_reset reset sink
set_interface_property pcie_reset EXPORT_OF global_reset.reset_in

# DDR4 memory pins (connected at top level)
add_interface ddr4_mem conduit end
set_interface_property ddr4_mem EXPORT_OF ddr4_emif.mem

# Status interface (for monitoring)
add_interface status_signals conduit end
set_interface_property status_signals EXPORT_OF pcie_hip.hip_status

#
# System Info
#

set_module_property NAME ai_tile_v0_sys
set_module_property DISPLAY_NAME "A10PED AI Tile v0 System"
set_module_property DESCRIPTION "Complete AI tile with PCIe, DDR4, CSR, and memcopy kernel"
set_module_property VERSION 1.0

# Save system
save_system ai_tile_v0_sys.qsys

puts ""
puts "✅ Platform Designer system created: ai_tile_v0_sys.qsys"
puts ""
puts "⚠️  IMPORTANT NOTES:"
puts "  1. This system uses PCIe Hard IP - you MUST have Quartus Prime Pro"
puts "  2. DDR4 EMIF requires pin planning - use Pin Planner to assign DDR4 pins"
puts "  3. The ai_csr and memcopy_kernel are instantiated in top-level RTL"
puts "  4. PCIe requires 100 MHz reference clock on dedicated pin"
puts ""
puts "Next steps:"
puts "  1. Review ai_tile_v0_sys.qsys in Platform Designer GUI"
puts "  2. Use Pin Planner to assign DDR4 pins (Tools → Pin Planner)"
puts "  3. Generate system: qsys-generate ai_tile_v0_sys.qsys --synthesis=VERILOG"
puts "  4. Build Quartus project: quartus_sh -t build.tcl"
puts ""
