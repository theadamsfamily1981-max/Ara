//============================================================================
// BANOS - Bio-Affective Neuromorphic Operating System
// Vacuum Spiker - Silence-First Anomaly Detection SNN
//============================================================================
// The Vacuum Spiker maintains neural silence during normal operation.
// It "wakes up" (spikes) only when anomalies occur.
//
// Key Innovation:
// - Inhibitory STDP: Network learns to SUPPRESS normal patterns
// - "Vacuum" criterion: Activity = Evidence of anomaly
// - Sub-millisecond hardware response
//
// Architecture:
// - Layer I (Input): Interval-coded sensor spikes
// - Layer R (Recurrent): LIF neurons with inhibitory learning
// - Output: Spike count threshold for alerts
//============================================================================

`timescale 1ns / 1ps

module vacuum_spiker #(
    parameter NUM_INPUTS     = 16,          // Input neurons (from interval encoder)
    parameter NUM_RECURRENT  = 32,          // Recurrent layer neurons
    parameter DATA_WIDTH     = 16,          // Fixed-point width
    parameter FRAC_BITS      = 8,           // Fractional bits
    parameter WEIGHT_BITS    = 8,           // Synaptic weight precision
    parameter ALERT_THRESHOLD = 5,          // Spikes in window to trigger alert
    parameter WINDOW_CYCLES  = 256          // Alert integration window
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    enable,

    // Input spikes from interval encoder (one-hot)
    input  wire [NUM_INPUTS-1:0]   input_spikes,
    input  wire                    input_valid,

    // Learning control
    input  wire                    learn_enable,  // Enable STDP updates
    input  wire                    force_inhibit, // Training mode: depress all active

    // Synapse weight interface (for loading/saving)
    input  wire                    weight_we,
    input  wire [$clog2(NUM_INPUTS*NUM_RECURRENT)-1:0] weight_addr,
    input  wire signed [WEIGHT_BITS-1:0] weight_wdata,
    output wire signed [WEIGHT_BITS-1:0] weight_rdata,

    // Output layer spikes
    output wire [NUM_RECURRENT-1:0] output_spikes,
    output wire [$clog2(NUM_RECURRENT):0] spike_count,

    // Alert output
    output reg                     alert,
    output reg  [$clog2(NUM_RECURRENT*WINDOW_CYCLES):0] pain_level,

    // Status
    output wire                    vacuum_state,  // True if network is silent
    output wire [31:0]             total_spikes_lifetime
);

    //------------------------------------------------------------------------
    // Synapse Weight Memory
    // W[i][j] = weight from input i to recurrent j
    //------------------------------------------------------------------------

    reg signed [WEIGHT_BITS-1:0] weights [0:NUM_INPUTS*NUM_RECURRENT-1];

    // Initialize weights to small negative values (inhibitory bias)
    integer init_i;
    initial begin
        for (init_i = 0; init_i < NUM_INPUTS*NUM_RECURRENT; init_i = init_i + 1) begin
            weights[init_i] = -8'sd8;  // Small inhibitory default
        end
    end

    // Weight read/write
    assign weight_rdata = weights[weight_addr];

    always @(posedge clk) begin
        if (weight_we) begin
            weights[weight_addr] <= weight_wdata;
        end
    end

    //------------------------------------------------------------------------
    // Synaptic Current Computation
    // For each recurrent neuron j: I_j = sum_i(W[i][j] * spike[i])
    //------------------------------------------------------------------------

    reg signed [DATA_WIDTH-1:0] synaptic_currents [0:NUM_RECURRENT-1];
    reg [NUM_RECURRENT-1:0] current_valid;

    integer i, j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (j = 0; j < NUM_RECURRENT; j = j + 1) begin
                synaptic_currents[j] <= 0;
            end
            current_valid <= 0;
        end else if (input_valid && enable) begin
            // Compute weighted sum for each recurrent neuron
            for (j = 0; j < NUM_RECURRENT; j = j + 1) begin
                synaptic_currents[j] <= 0;
                for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                    if (input_spikes[i]) begin
                        // Add weight (sign-extended to DATA_WIDTH)
                        synaptic_currents[j] <= synaptic_currents[j] +
                            {{(DATA_WIDTH-WEIGHT_BITS){weights[i*NUM_RECURRENT+j][WEIGHT_BITS-1]}},
                             weights[i*NUM_RECURRENT+j]};
                    end
                end
            end
            current_valid <= {NUM_RECURRENT{1'b1}};
        end else begin
            current_valid <= 0;
        end
    end

    //------------------------------------------------------------------------
    // Pack synaptic currents for neuron array
    //------------------------------------------------------------------------

    wire signed [NUM_RECURRENT*DATA_WIDTH-1:0] packed_currents;

    genvar g;
    generate
        for (g = 0; g < NUM_RECURRENT; g = g + 1) begin : pack_currents
            assign packed_currents[(g+1)*DATA_WIDTH-1:g*DATA_WIDTH] = synaptic_currents[g];
        end
    endgenerate

    //------------------------------------------------------------------------
    // Recurrent Layer (LIF Neuron Array)
    //------------------------------------------------------------------------

    wire signed [NUM_RECURRENT*DATA_WIDTH-1:0] membrane_potentials;
    wire [NUM_RECURRENT-1:0] refractory_states;

    lif_neuron_array #(
        .NUM_NEURONS(NUM_RECURRENT),
        .DATA_WIDTH(DATA_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .THRESHOLD(16'h4000),     // Lower threshold for sensitivity
        .V_RESET(16'h0000),
        .V_REST(16'h1000),
        .LEAK_RATE(16'h0080),     // Moderate leak
        .REFRACTORY(8'd5)
    ) recurrent_layer (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .synaptic_currents(packed_currents),
        .input_valid(current_valid),
        .membrane_potentials(membrane_potentials),
        .spikes(output_spikes),
        .spike_count(spike_count),
        .refractory_states(refractory_states)
    );

    //------------------------------------------------------------------------
    // Inhibitory STDP Learning
    // Key: We DEPRESS weights for frequently co-active pairs
    // This creates the "vacuum" - normal patterns get suppressed
    //------------------------------------------------------------------------

    // Pre-synaptic trace (decaying memory of input spikes)
    reg [7:0] pre_trace [0:NUM_INPUTS-1];

    // Post-synaptic trace (decaying memory of output spikes)
    reg [7:0] post_trace [0:NUM_RECURRENT-1];

    // STDP parameters
    localparam [7:0] TRACE_DECAY = 8'd4;    // Trace decay per cycle
    localparam [WEIGHT_BITS-1:0] A_MINUS = 8'sd2;  // Depression amplitude (dominant)
    localparam [WEIGHT_BITS-1:0] A_PLUS  = 8'sd1;  // Potentiation amplitude (weak)
    localparam signed [WEIGHT_BITS-1:0] W_MIN = -8'sd127;
    localparam signed [WEIGHT_BITS-1:0] W_MAX = 8'sd127;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                pre_trace[i] <= 0;
            end
            for (j = 0; j < NUM_RECURRENT; j = j + 1) begin
                post_trace[j] <= 0;
            end
        end else if (enable) begin
            // Update pre-synaptic traces
            for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                if (input_spikes[i] && input_valid) begin
                    pre_trace[i] <= 8'hFF;  // Max on spike
                end else if (pre_trace[i] > TRACE_DECAY) begin
                    pre_trace[i] <= pre_trace[i] - TRACE_DECAY;
                end else begin
                    pre_trace[i] <= 0;
                end
            end

            // Update post-synaptic traces
            for (j = 0; j < NUM_RECURRENT; j = j + 1) begin
                if (output_spikes[j]) begin
                    post_trace[j] <= 8'hFF;  // Max on spike
                end else if (post_trace[j] > TRACE_DECAY) begin
                    post_trace[j] <= post_trace[j] - TRACE_DECAY;
                end else begin
                    post_trace[j] <= 0;
                end
            end

            // STDP weight updates
            if (learn_enable) begin
                for (i = 0; i < NUM_INPUTS; i = i + 1) begin
                    for (j = 0; j < NUM_RECURRENT; j = j + 1) begin
                        // Depression: pre-then-post (normal STDP) or forced
                        if ((input_spikes[i] && post_trace[j] > 8'd64) || force_inhibit) begin
                            if (weights[i*NUM_RECURRENT+j] > W_MIN + A_MINUS) begin
                                weights[i*NUM_RECURRENT+j] <= weights[i*NUM_RECURRENT+j] - A_MINUS;
                            end else begin
                                weights[i*NUM_RECURRENT+j] <= W_MIN;
                            end
                        end
                        // Weak potentiation: post-then-pre (reversed from normal)
                        else if (output_spikes[j] && pre_trace[i] > 8'd64 && !force_inhibit) begin
                            if (weights[i*NUM_RECURRENT+j] < W_MAX - A_PLUS) begin
                                weights[i*NUM_RECURRENT+j] <= weights[i*NUM_RECURRENT+j] + A_PLUS;
                            end else begin
                                weights[i*NUM_RECURRENT+j] <= W_MAX;
                            end
                        end
                    end
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Spike Integration Window (Pain Level Computation)
    //------------------------------------------------------------------------

    reg [$clog2(WINDOW_CYCLES)-1:0] window_counter;
    reg [$clog2(NUM_RECURRENT*WINDOW_CYCLES):0] window_spikes;
    reg [$clog2(NUM_RECURRENT*WINDOW_CYCLES):0] prev_window_spikes;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_counter <= 0;
            window_spikes <= 0;
            prev_window_spikes <= 0;
            pain_level <= 0;
        end else if (enable) begin
            // Accumulate spikes in current window
            window_spikes <= window_spikes + spike_count;
            window_counter <= window_counter + 1;

            // End of window
            if (window_counter == WINDOW_CYCLES - 1) begin
                window_counter <= 0;
                prev_window_spikes <= window_spikes;
                pain_level <= window_spikes;  // Update pain level
                window_spikes <= 0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Alert Logic (The Vacuum Criterion)
    //------------------------------------------------------------------------

    reg [31:0] lifetime_spike_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alert <= 1'b0;
            lifetime_spike_counter <= 0;
        end else if (enable) begin
            // Alert if spike count exceeds threshold
            alert <= (spike_count >= ALERT_THRESHOLD);

            // Lifetime counter for statistics
            lifetime_spike_counter <= lifetime_spike_counter + spike_count;
        end
    end

    assign total_spikes_lifetime = lifetime_spike_counter;

    //------------------------------------------------------------------------
    // Vacuum State Detection
    // Network is in "vacuum" if no spikes in recent window
    //------------------------------------------------------------------------

    assign vacuum_state = (prev_window_spikes == 0) && (window_spikes == 0);

endmodule


//============================================================================
// Reflex Controller - Translates neural alerts to hardware actions
//============================================================================

module reflex_controller #(
    parameter NUM_CHANNELS = 4
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,

    // Neural alerts (one per monitored channel)
    input  wire [NUM_CHANNELS-1:0] alerts,
    input  wire [NUM_CHANNELS*16-1:0] pain_levels,

    // Thermal emergency inputs (bypass neural processing)
    input  wire [NUM_CHANNELS-1:0] thermal_critical,
    input  wire [NUM_CHANNELS-1:0] thermal_emergency,

    // Hardware reflex outputs
    output reg         prochot_assert,      // Processor hot signal
    output reg  [7:0]  fan_pwm_override,    // Direct fan control (0-255)
    output reg         emergency_shutdown,   // Last resort

    // Process kill requests (to kernel)
    output reg  [3:0]  kill_priority,       // Kill processes below this priority
    output reg         kill_request,

    // Reflex log (for kernel to read)
    output reg  [31:0] reflex_action_log,   // Bitmap of actions taken
    output reg  [31:0] reflex_timestamp
);

    //------------------------------------------------------------------------
    // Aggregate Pain Level
    //------------------------------------------------------------------------

    reg [19:0] total_pain;
    integer i;

    always @(*) begin
        total_pain = 0;
        for (i = 0; i < NUM_CHANNELS; i = i + 1) begin
            total_pain = total_pain + pain_levels[(i+1)*16-1:i*16];
        end
    end

    //------------------------------------------------------------------------
    // Time Counter
    //------------------------------------------------------------------------

    reg [31:0] time_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            time_counter <= 0;
        end else begin
            time_counter <= time_counter + 1;
        end
    end

    //------------------------------------------------------------------------
    // Reflex State Machine
    //------------------------------------------------------------------------

    localparam IDLE      = 3'd0;
    localparam MILD      = 3'd1;  // Fan ramp
    localparam MODERATE  = 3'd2;  // PROCHOT + kill low priority
    localparam SEVERE    = 3'd3;  // Kill medium priority
    localparam CRITICAL  = 3'd4;  // Emergency measures
    localparam EMERGENCY = 3'd5;  // Shutdown imminent

    reg [2:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            prochot_assert <= 1'b0;
            fan_pwm_override <= 8'd0;
            emergency_shutdown <= 1'b0;
            kill_priority <= 4'd0;
            kill_request <= 1'b0;
            reflex_action_log <= 32'b0;
            reflex_timestamp <= 32'b0;
        end else if (enable) begin
            // Default: no new actions
            kill_request <= 1'b0;

            // Emergency thermal bypass (highest priority)
            if (|thermal_emergency) begin
                state <= EMERGENCY;
                emergency_shutdown <= 1'b1;
                reflex_action_log <= reflex_action_log | 32'h8000_0000;
                reflex_timestamp <= time_counter;
            end
            // Critical thermal (direct hardware response)
            else if (|thermal_critical) begin
                state <= CRITICAL;
                prochot_assert <= 1'b1;
                fan_pwm_override <= 8'hFF;  // Max fan
                kill_priority <= 4'd8;      // Kill anything below priority 8
                kill_request <= 1'b1;
                reflex_action_log <= reflex_action_log | 32'h4000_0000;
                reflex_timestamp <= time_counter;
            end
            // Neural alert processing
            else if (|alerts) begin
                case (state)
                    IDLE: begin
                        state <= MILD;
                        fan_pwm_override <= 8'd192;  // 75% fan
                        reflex_action_log <= reflex_action_log | 32'h0000_0001;
                        reflex_timestamp <= time_counter;
                    end

                    MILD: begin
                        if (total_pain > 20'd1000) begin
                            state <= MODERATE;
                            prochot_assert <= 1'b1;
                            fan_pwm_override <= 8'd224;  // 88% fan
                            kill_priority <= 4'd2;
                            kill_request <= 1'b1;
                            reflex_action_log <= reflex_action_log | 32'h0000_0010;
                        end
                    end

                    MODERATE: begin
                        if (total_pain > 20'd5000) begin
                            state <= SEVERE;
                            kill_priority <= 4'd5;
                            kill_request <= 1'b1;
                            reflex_action_log <= reflex_action_log | 32'h0000_0100;
                        end
                    end

                    SEVERE: begin
                        if (total_pain > 20'd10000) begin
                            state <= CRITICAL;
                            fan_pwm_override <= 8'hFF;
                            kill_priority <= 4'd8;
                            kill_request <= 1'b1;
                            reflex_action_log <= reflex_action_log | 32'h0000_1000;
                        end
                    end

                    CRITICAL: begin
                        // Hold until pain subsides significantly
                    end

                    default: state <= IDLE;
                endcase
            end
            // Recovery: reduce responses as pain decreases
            else begin
                case (state)
                    EMERGENCY: begin
                        // No recovery from emergency
                    end

                    CRITICAL: begin
                        if (total_pain < 20'd5000) begin
                            state <= SEVERE;
                            prochot_assert <= 1'b0;
                        end
                    end

                    SEVERE: begin
                        if (total_pain < 20'd2000) begin
                            state <= MODERATE;
                        end
                    end

                    MODERATE: begin
                        if (total_pain < 20'd500) begin
                            state <= MILD;
                            prochot_assert <= 1'b0;
                        end
                    end

                    MILD: begin
                        if (total_pain < 20'd100) begin
                            state <= IDLE;
                            fan_pwm_override <= 8'd0;  // Release fan control
                        end
                    end

                    default: begin
                        state <= IDLE;
                    end
                endcase
            end
        end
    end

endmodule
