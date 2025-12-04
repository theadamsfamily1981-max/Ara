//============================================================================
// BANOS - Bio-Affective Neuromorphic Operating System
// LIF Neuron - Leaky Integrate-and-Fire Neuron Implementation
//============================================================================
// This is the fundamental building block of Ara's hindbrain.
// Each neuron integrates weighted synaptic inputs, leaks toward rest,
// and fires a spike when membrane potential exceeds threshold.
//
// Key Features:
// - Fixed-point arithmetic for FPGA efficiency
// - Configurable leak rate (tau_m equivalent)
// - Refractory period support
// - Spike output with timing information
//============================================================================

`timescale 1ns / 1ps

module lif_neuron #(
    parameter DATA_WIDTH     = 16,          // Fixed-point width
    parameter FRAC_BITS      = 8,           // Fractional bits for Q8.8 format
    parameter THRESHOLD      = 16'h6000,    // Firing threshold (0.375 in Q8.8)
    parameter V_RESET        = 16'h0000,    // Reset potential after spike
    parameter V_REST         = 16'h2000,    // Resting potential (0.125 in Q8.8)
    parameter LEAK_RATE      = 16'h0100,    // Leak factor per cycle (tau_m ~ 256 cycles)
    parameter REFRACTORY     = 8'd10        // Refractory period in clock cycles
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    enable,

    // Synaptic input (weighted sum from presynaptic neurons)
    input  wire signed [DATA_WIDTH-1:0] synaptic_current,
    input  wire                    input_valid,

    // Membrane potential (for monitoring/debugging)
    output reg  signed [DATA_WIDTH-1:0] membrane_potential,

    // Spike output
    output reg                     spike,
    output reg  [31:0]             spike_time,    // Timestamp of last spike

    // Status
    output wire                    in_refractory,
    output wire                    ready
);

    //------------------------------------------------------------------------
    // Internal Signals
    //------------------------------------------------------------------------

    reg [7:0]  refractory_counter;
    reg [31:0] time_counter;

    wire signed [DATA_WIDTH-1:0] leak_term;
    wire signed [DATA_WIDTH+1:0] potential_update;  // Extra bits for overflow
    wire signed [DATA_WIDTH-1:0] new_potential;
    wire                         threshold_crossed;

    //------------------------------------------------------------------------
    // Leak Calculation
    // leak_term = LEAK_RATE * (membrane_potential - V_REST)
    // This implements: dV/dt = -g_L * (V - E_L)
    //------------------------------------------------------------------------

    wire signed [DATA_WIDTH-1:0] v_diff;
    wire signed [2*DATA_WIDTH-1:0] leak_product;

    assign v_diff = membrane_potential - $signed(V_REST);
    assign leak_product = v_diff * $signed(LEAK_RATE);
    assign leak_term = leak_product[DATA_WIDTH+FRAC_BITS-1:FRAC_BITS]; // Scale back

    //------------------------------------------------------------------------
    // Potential Update
    // V(t+1) = V(t) - leak_term + synaptic_current
    //------------------------------------------------------------------------

    assign potential_update = $signed({membrane_potential[DATA_WIDTH-1], membrane_potential})
                            - $signed({leak_term[DATA_WIDTH-1], leak_term})
                            + $signed({synaptic_current[DATA_WIDTH-1], synaptic_current});

    // Saturate to prevent overflow
    assign new_potential = (potential_update > $signed({{2{1'b0}}, {DATA_WIDTH{1'b1}}})) ?
                           {1'b0, {(DATA_WIDTH-1){1'b1}}} :  // Max positive
                           (potential_update < $signed({{2{1'b1}}, {DATA_WIDTH{1'b0}}})) ?
                           {1'b1, {(DATA_WIDTH-1){1'b0}}} :  // Max negative
                           potential_update[DATA_WIDTH-1:0];

    //------------------------------------------------------------------------
    // Threshold Detection
    //------------------------------------------------------------------------

    assign threshold_crossed = (membrane_potential >= $signed(THRESHOLD)) && !in_refractory;

    //------------------------------------------------------------------------
    // Status Outputs
    //------------------------------------------------------------------------

    assign in_refractory = (refractory_counter > 0);
    assign ready = enable && !in_refractory;

    //------------------------------------------------------------------------
    // Time Counter (for spike timestamps)
    //------------------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            time_counter <= 32'b0;
        end else if (enable) begin
            time_counter <= time_counter + 1;
        end
    end

    //------------------------------------------------------------------------
    // Main Neuron Dynamics
    //------------------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane_potential <= V_REST;
            spike <= 1'b0;
            spike_time <= 32'b0;
            refractory_counter <= 8'b0;
        end else if (enable) begin
            // Default: no spike this cycle
            spike <= 1'b0;

            // Refractory period countdown
            if (refractory_counter > 0) begin
                refractory_counter <= refractory_counter - 1;
                // During refractory, potential decays toward rest
                if (membrane_potential > $signed(V_REST)) begin
                    membrane_potential <= membrane_potential - $signed(LEAK_RATE);
                end else begin
                    membrane_potential <= V_REST;
                end
            end
            // Normal operation
            else if (input_valid) begin
                if (threshold_crossed) begin
                    // FIRE!
                    spike <= 1'b1;
                    spike_time <= time_counter;
                    membrane_potential <= V_RESET;
                    refractory_counter <= REFRACTORY;
                end else begin
                    // Integrate
                    membrane_potential <= new_potential;
                end
            end
            // No input - just leak
            else begin
                if (membrane_potential > $signed(V_REST)) begin
                    membrane_potential <= membrane_potential - $signed(leak_term);
                end else if (membrane_potential < $signed(V_REST)) begin
                    membrane_potential <= membrane_potential + $signed(LEAK_RATE);
                end
            end
        end
    end

endmodule


//============================================================================
// LIF Neuron Array - Multiple neurons with shared control
//============================================================================

module lif_neuron_array #(
    parameter NUM_NEURONS    = 16,
    parameter DATA_WIDTH     = 16,
    parameter FRAC_BITS      = 8,
    parameter THRESHOLD      = 16'h6000,
    parameter V_RESET        = 16'h0000,
    parameter V_REST         = 16'h2000,
    parameter LEAK_RATE      = 16'h0100,
    parameter REFRACTORY     = 8'd10
)(
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              enable,

    // Synaptic inputs for all neurons
    input  wire signed [NUM_NEURONS*DATA_WIDTH-1:0] synaptic_currents,
    input  wire [NUM_NEURONS-1:0]            input_valid,

    // Membrane potentials (for monitoring)
    output wire signed [NUM_NEURONS*DATA_WIDTH-1:0] membrane_potentials,

    // Spike outputs
    output wire [NUM_NEURONS-1:0]            spikes,

    // Aggregate outputs
    output wire [$clog2(NUM_NEURONS):0]      spike_count,  // Number of spikes this cycle
    output wire [NUM_NEURONS-1:0]            refractory_states
);

    genvar i;

    // Instantiate neuron array
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : neuron_gen
            lif_neuron #(
                .DATA_WIDTH(DATA_WIDTH),
                .FRAC_BITS(FRAC_BITS),
                .THRESHOLD(THRESHOLD),
                .V_RESET(V_RESET),
                .V_REST(V_REST),
                .LEAK_RATE(LEAK_RATE),
                .REFRACTORY(REFRACTORY)
            ) neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .enable(enable),
                .synaptic_current(synaptic_currents[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH]),
                .input_valid(input_valid[i]),
                .membrane_potential(membrane_potentials[(i+1)*DATA_WIDTH-1:i*DATA_WIDTH]),
                .spike(spikes[i]),
                .spike_time(),  // Could connect to FIFO for spike timing
                .in_refractory(refractory_states[i]),
                .ready()
            );
        end
    endgenerate

    // Population spike counter
    integer j;
    reg [$clog2(NUM_NEURONS):0] spike_sum;

    always @(*) begin
        spike_sum = 0;
        for (j = 0; j < NUM_NEURONS; j = j + 1) begin
            spike_sum = spike_sum + spikes[j];
        end
    end

    assign spike_count = spike_sum;

endmodule
