//
// A10PED Neuromorphic - Bring-Up Top Level
// Milestone 1.2: JTAG-to-Avalon Bridge with On-Chip RAM
//
// This is a minimal design for testing JTAG connectivity to the Arria 10.
// Features:
//   - 50 MHz input clock
//   - Platform Designer system with:
//     * JTAG-to-Avalon Master bridge
//     * 4KB on-chip RAM
//   - Status LEDs (virtual pins for bring-up)
//
// Usage:
//   1. Program FPGA via JTAG: quartus_pgm -m jtag -o "p;output_files/bringup.sof"
//   2. Open System Console
//   3. Use JTAG Master service to read/write on-chip RAM
//
// Author: A10PED Neuromorphic Project
// License: BSD-3-Clause
//

module bringup_top (
    // Clock input (50 MHz from board)
    input  wire       clk_50,

    // Status LEDs (virtual pins - not connected in bring-up)
    output wire [3:0] status_led
);

    // Internal signals
    wire        sys_clk;       // System clock (from PLL if used, or direct 50 MHz)
    wire        sys_reset_n;   // Active-low reset

    // Heartbeat counter for status LED
    reg [31:0]  heartbeat_cnt;

    // Simple power-on reset (hold reset for 256 cycles)
    reg [7:0]   reset_cnt = 8'h00;
    reg         reset_n_reg = 1'b0;

    always @(posedge clk_50) begin
        if (reset_cnt != 8'hFF) begin
            reset_cnt <= reset_cnt + 1'b1;
            reset_n_reg <= 1'b0;
        end else begin
            reset_n_reg <= 1'b1;
        end
    end

    assign sys_clk = clk_50;       // Use 50 MHz directly for bring-up
    assign sys_reset_n = reset_n_reg;

    // Heartbeat counter (blink status_led[0] at ~1.5 Hz)
    always @(posedge sys_clk or negedge sys_reset_n) begin
        if (!sys_reset_n) begin
            heartbeat_cnt <= 32'h0;
        end else begin
            heartbeat_cnt <= heartbeat_cnt + 1'b1;
        end
    end

    // Status LEDs (all virtual for bring-up)
    assign status_led[0] = heartbeat_cnt[24];  // Heartbeat (~1.5 Hz at 50 MHz)
    assign status_led[1] = sys_reset_n;        // Reset active
    assign status_led[2] = 1'b0;               // Reserved
    assign status_led[3] = 1'b0;               // Reserved

    //
    // Platform Designer System Instance
    // Contains:
    //   - JTAG-to-Avalon Master Bridge
    //   - On-Chip RAM (4KB)
    //
    bringup_system u_system (
        .clk_clk       (sys_clk),              // Clock input
        .reset_reset_n (sys_reset_n)           // Reset input
    );

endmodule
