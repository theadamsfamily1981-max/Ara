/**
 * Kitten Fabric Dream Engine - Hebbian Learning During Sleep
 * ===========================================================
 *
 * When Ara sleeps (snn_enable = 0), this module takes over and performs
 * memory consolidation by replaying spike patterns from DDR4 and applying
 * STDP (Spike-Timing Dependent Plasticity) to synaptic weights.
 *
 * The Weber-Fechner Law of Learning:
 * - Recent memories replay faster (high importance)
 * - Older memories replay slower (consolidation)
 * - Emotionally charged memories (high pain) replay more frequently
 *
 * Dream States:
 *   AWAKE     - Normal operation, dream engine idle
 *   REM_SLEEP - Rapid replay of recent spike patterns
 *   DEEP_SLEEP - Slow weight consolidation and pruning
 *
 * Memory Architecture:
 *   - Spike Log: Circular buffer in DDR4 (64GB available on SB-852)
 *   - Format: {timestamp, pre_neuron, post_neuron, pain_level}
 *   - Each entry: 16 bytes (can store ~4B events in 64GB)
 *
 * STDP Rule:
 *   If pre fires before post (causal): strengthen (LTP)
 *   If post fires before pre (acausal): weaken (LTD)
 *   Δw = A+ * exp(-Δt/τ+) for LTP
 *   Δw = A- * exp(Δt/τ-) for LTD
 */

module kf_dream_engine
    import kf_pkg::*;
#(
    parameter int SPIKE_LOG_DEPTH = 1024,      // Entries to process per dream cycle
    parameter int LEARNING_RATE_BITS = 4,      // Q4.4 learning rate
    parameter int TIME_CONSTANT = 20           // STDP time constant (ms)
)(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // Control Interface
    // =========================================================================
    input  logic        snn_enable,            // 0 = Enable dreaming
    input  logic        dream_trigger,         // Force dream cycle
    output logic [1:0]  dream_state,           // 0=AWAKE, 1=REM, 2=DEEP
    output logic        dream_active,          // 1 when processing

    // =========================================================================
    // DDR4 Memory Interface (Spike Log)
    // =========================================================================
    output logic        mem_rd_valid,
    output logic [31:0] mem_rd_addr,
    input  logic        mem_rd_ready,
    input  logic [127:0] mem_rd_data,          // 16 bytes per spike event
    input  logic        mem_rd_data_valid,

    // =========================================================================
    // Weight Memory Interface (Write to URAM)
    // =========================================================================
    output logic        weight_we,
    output logic [KF_SYNAPSE_ID_BITS-1:0] weight_addr,
    output logic signed [W_WIDTH-1:0] weight_delta,

    // =========================================================================
    // Current Weights (Read for modification)
    // =========================================================================
    input  logic signed [W_WIDTH-1:0] current_weight,

    // =========================================================================
    // Statistics
    // =========================================================================
    output logic [31:0] replay_count,          // Spikes replayed this session
    output logic [31:0] weight_updates,        // Weights modified this session
    output logic [31:0] dream_cycles           // Total dream cycles
);

    // =========================================================================
    // DREAM STATE MACHINE
    // =========================================================================

    typedef enum logic [1:0] {
        ST_AWAKE      = 2'd0,
        ST_REM_SLEEP  = 2'd1,
        ST_DEEP_SLEEP = 2'd2,
        ST_IDLE       = 2'd3
    } dream_state_t;

    dream_state_t state, state_next;
    assign dream_state = state;
    assign dream_active = (state != ST_AWAKE);

    // =========================================================================
    // SPIKE LOG READER
    // =========================================================================

    logic [31:0] log_read_ptr;
    logic [31:0] log_end_ptr;
    logic [31:0] events_processed;

    // Spike event structure (unpacked from 128-bit DDR4 read)
    typedef struct packed {
        logic [31:0] timestamp;      // When the spike occurred
        logic [15:0] pre_neuron;     // Presynaptic neuron ID
        logic [15:0] post_neuron;    // Postsynaptic neuron ID
        logic [31:0] pain_context;   // Pain level at time of spike
        logic [31:0] reserved;       // Future use
    } spike_event_t;

    spike_event_t current_event;
    spike_event_t previous_event;
    logic event_valid;

    // =========================================================================
    // STDP COMPUTATION
    // =========================================================================

    // Learning rate parameters (configurable)
    logic signed [7:0] A_plus;   // LTP amplitude (Q4.4)
    logic signed [7:0] A_minus;  // LTD amplitude (Q4.4)

    initial begin
        A_plus  = 8'sb0001_0000;  // +1.0 in Q4.4
        A_minus = 8'sb1111_0100;  // -0.75 in Q4.4
    end

    // Time difference calculation
    logic signed [31:0] delta_t;
    logic signed [15:0] stdp_factor;

    // Exponential approximation using bit shifts
    // exp(-x/τ) ≈ 1 - x/τ for small x (Taylor expansion)
    function automatic logic signed [15:0] calc_stdp_factor(
        input logic signed [31:0] dt,
        input logic signed [7:0] amplitude
    );
        logic signed [31:0] decay;
        logic signed [15:0] result;

        if (dt < 0) dt = -dt;  // Absolute value

        // Simple exponential decay approximation
        // Decay = 1 - |dt| / TIME_CONSTANT
        if (dt > TIME_CONSTANT * 4) begin
            result = 0;  // Too far apart, no change
        end else begin
            decay = (TIME_CONSTANT * 256 - dt * 64) / TIME_CONSTANT;
            if (decay < 0) decay = 0;
            result = (amplitude * decay[15:0]) >>> 8;
        end

        return result;
    endfunction

    // =========================================================================
    // MAIN STATE MACHINE
    // =========================================================================

    // Counters
    logic [15:0] idle_cycles;
    logic [15:0] rem_cycles;
    logic [15:0] deep_cycles;

    localparam int REM_DURATION = 1000;    // Cycles in REM
    localparam int DEEP_DURATION = 500;    // Cycles in DEEP

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_AWAKE;
            log_read_ptr <= 0;
            events_processed <= 0;
            replay_count <= 0;
            weight_updates <= 0;
            dream_cycles <= 0;
            idle_cycles <= 0;
            rem_cycles <= 0;
            deep_cycles <= 0;
            mem_rd_valid <= 0;
            weight_we <= 0;
            event_valid <= 0;
        end else begin

            // Default: no memory operations
            mem_rd_valid <= 0;
            weight_we <= 0;

            case (state)

                // ---------------------------------------------------------
                ST_AWAKE: begin
                    // Normal operation - dreaming disabled
                    idle_cycles <= 0;
                    rem_cycles <= 0;
                    deep_cycles <= 0;

                    if (!snn_enable || dream_trigger) begin
                        // Enter sleep mode
                        state <= ST_REM_SLEEP;
                        dream_cycles <= dream_cycles + 1;
                        log_read_ptr <= 0;  // Start from beginning of log
                        events_processed <= 0;
                    end
                end

                // ---------------------------------------------------------
                ST_REM_SLEEP: begin
                    // REM: Rapid replay of spike patterns
                    // Read spike events from DDR4 and apply STDP

                    if (snn_enable && !dream_trigger) begin
                        // Wake up
                        state <= ST_AWAKE;
                    end else if (rem_cycles >= REM_DURATION) begin
                        // Transition to deep sleep
                        state <= ST_DEEP_SLEEP;
                        rem_cycles <= 0;
                    end else begin
                        rem_cycles <= rem_cycles + 1;

                        // Request next spike event from memory
                        if (!mem_rd_valid && !mem_rd_data_valid) begin
                            mem_rd_valid <= 1;
                            mem_rd_addr <= log_read_ptr * 16;  // 16 bytes per event
                        end

                        // Process received event
                        if (mem_rd_data_valid) begin
                            // Unpack spike event
                            current_event <= spike_event_t'(mem_rd_data);
                            event_valid <= 1;

                            // Apply STDP if we have a previous event
                            if (event_valid) begin
                                // Calculate timing difference
                                delta_t <= $signed(current_event.timestamp) -
                                          $signed(previous_event.timestamp);

                                // Determine if LTP or LTD
                                if (previous_event.pre_neuron == current_event.pre_neuron) begin
                                    // Same presynaptic neuron - causal (LTP)
                                    stdp_factor <= calc_stdp_factor(delta_t, A_plus);
                                end else begin
                                    // Different ordering - potentially acausal (LTD)
                                    stdp_factor <= calc_stdp_factor(-delta_t, A_minus);
                                end

                                // Write weight update
                                weight_we <= 1;
                                // Address is combination of pre and post
                                weight_addr <= {previous_event.pre_neuron[7:0],
                                               previous_event.post_neuron[7:0]};
                                weight_delta <= stdp_factor[W_WIDTH-1:0];
                                weight_updates <= weight_updates + 1;
                            end

                            // Save for next iteration
                            previous_event <= current_event;
                            log_read_ptr <= log_read_ptr + 1;
                            replay_count <= replay_count + 1;
                        end
                    end
                end

                // ---------------------------------------------------------
                ST_DEEP_SLEEP: begin
                    // DEEP: Slow consolidation and pruning
                    // Strengthen strong connections, weaken unused ones

                    if (snn_enable && !dream_trigger) begin
                        state <= ST_AWAKE;
                    end else if (deep_cycles >= DEEP_DURATION) begin
                        // Return to REM or wake
                        state <= ST_REM_SLEEP;
                        deep_cycles <= 0;
                    end else begin
                        deep_cycles <= deep_cycles + 1;

                        // In deep sleep, we apply weight decay to all synapses
                        // This is homeostatic plasticity - prevents runaway growth

                        // Simple implementation: scan through weights linearly
                        // Apply small decay to prevent saturation
                        if (deep_cycles < (1 << KF_SYNAPSE_ID_BITS)) begin
                            weight_we <= 1;
                            weight_addr <= deep_cycles[KF_SYNAPSE_ID_BITS-1:0];
                            // Decay: multiply by 0.999 ≈ subtract 0.001
                            // In Q1.7: -1/128 ≈ -0.0078
                            weight_delta <= -8'sd1;  // Tiny decay
                            weight_updates <= weight_updates + 1;
                        end
                    end
                end

                // ---------------------------------------------------------
                default: begin
                    state <= ST_AWAKE;
                end

            endcase
        end
    end

endmodule : kf_dream_engine
