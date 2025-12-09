// ===========================================================================
// Leaky Integrate-and-Fire (LIF) Neuron Core
// ===========================================================================
//
// Implements biological-inspired spiking neuron with:
// - Membrane potential integration
// - Leak current (exponential decay)
// - Spike threshold detection
// - Refractory period
//
// Optimized for FPGA implementation:
// - Fixed-point arithmetic (16.8 format)
// - Parallel neuron array (configurable width)
// - Low latency (<10 cycles per update)
//
// Author: FPGA Salvage Project
// License: Apache 2.0
// ===========================================================================

module neuron_core #(
    parameter NUM_NEURONS = 256,        // Neurons per core
    parameter VMEM_WIDTH = 24,          // Membrane potential bits (16.8 fixed-point)
    parameter THRESHOLD = 24'h010000,   // Spike threshold (1.0 in 16.8)
    parameter LEAK_RATE = 24'h000100,   // Leak coefficient (1/256 decay per cycle)
    parameter REFRACT_CYCLES = 8        // Refractory period length
)(
    input wire clk,
    input wire rst_n,

    // Input spikes (from synapses)
    input wire [NUM_NEURONS-1:0] spike_in_valid,
    input wire [VMEM_WIDTH-1:0] spike_in_weight [NUM_NEURONS-1:0],

    // Output spikes (to routing network)
    output reg [NUM_NEURONS-1:0] spike_out_valid,
    output reg [NUM_NEURONS-1:0] spike_out_id,

    // Configuration
    input wire [VMEM_WIDTH-1:0] cfg_threshold,
    input wire [VMEM_WIDTH-1:0] cfg_leak_rate,
    input wire cfg_enable
);

// ===========================================================================
// Internal State
// ===========================================================================

// Membrane potential for each neuron
reg [VMEM_WIDTH-1:0] vmem [NUM_NEURONS-1:0];

// Refractory counter (0 = ready, >0 = refractory)
reg [3:0] refract_cnt [NUM_NEURONS-1:0];

// Leak accumulator (for precise exponential decay)
reg [VMEM_WIDTH-1:0] leak_accum [NUM_NEURONS-1:0];

// ===========================================================================
// Neuron Update Logic (Parallel per Neuron)
// ===========================================================================

integer i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        // Reset all neurons
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            vmem[i] <= 0;
            refract_cnt[i] <= 0;
            leak_accum[i] <= 0;
            spike_out_valid[i] <= 0;
        end
    end else if (cfg_enable) begin
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            // Default: no spike output
            spike_out_valid[i] <= 0;

            // ===================================================
            // 1. Refractory Period Logic
            // ===================================================
            if (refract_cnt[i] > 0) begin
                // Neuron in refractory period
                refract_cnt[i] <= refract_cnt[i] - 1;
                vmem[i] <= 0;  // Hold at reset potential

            end else begin
                // ===================================================
                // 2. Leak Current (Exponential Decay)
                // ===================================================
                // V_mem = V_mem - (V_mem * leak_rate)
                // Implemented as: leak_accum += V_mem * leak_rate
                //                 V_mem -= leak_accum >> 16

                leak_accum[i] <= leak_accum[i] +
                                 ((vmem[i] * cfg_leak_rate) >> 8);

                if (leak_accum[i] >= 24'h010000) begin
                    // Accumulated enough leak to decrement
                    vmem[i] <= vmem[i] - (leak_accum[i] >> 16);
                    leak_accum[i] <= leak_accum[i] & 24'h00FFFF;
                end

                // ===================================================
                // 3. Input Integration
                // ===================================================
                if (spike_in_valid[i]) begin
                    // Add weighted input to membrane potential
                    vmem[i] <= vmem[i] + spike_in_weight[i];
                end

                // ===================================================
                // 4. Spike Generation
                // ===================================================
                if (vmem[i] >= cfg_threshold) begin
                    // Threshold crossed - generate output spike
                    spike_out_valid[i] <= 1;
                    spike_out_id[i] <= i;

                    // Reset membrane potential
                    vmem[i] <= 0;

                    // Enter refractory period
                    refract_cnt[i] <= REFRACT_CYCLES;
                end
            end
        end
    end
end

// ===========================================================================
// Performance Counters (for debugging/profiling)
// ===========================================================================

reg [31:0] total_spikes;
reg [31:0] total_cycles;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        total_spikes <= 0;
        total_cycles <= 0;
    end else if (cfg_enable) begin
        total_cycles <= total_cycles + 1;

        // Count output spikes
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            if (spike_out_valid[i])
                total_spikes <= total_spikes + 1;
        end
    end
end

// ===========================================================================
// Assertions for Simulation
// ===========================================================================

`ifdef SIMULATION
    // Check for overflow
    always @(posedge clk) begin
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin
            if (vmem[i] > 24'h100000) begin  // 2.0 in 16.8 format
                $display("WARNING: Neuron %d membrane overflow: %h", i, vmem[i]);
            end
        end
    end
`endif

endmodule
