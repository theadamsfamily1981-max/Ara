/**
 * Bit-Serial Holographic Neuron
 * ==============================
 *
 * The marriage of 1-Bit SNN and Hyperdimensional Computing (HDC).
 *
 * KEY INSIGHT: An 8192-bit hypervector streamed one bit at a time
 * IS a spike train. Space becomes Time.
 *
 * This neuron supports TWO MODES:
 *
 *   MODE 0 (SNN): Integrate-and-Fire
 *     - sum += weight_bit * spike_bit
 *     - Fire when accumulator > threshold
 *     - Classical spiking neural network
 *
 *   MODE 1 (HDC): Hyperdimensional Binding
 *     - out_bit = vec_a_bit XOR vec_b_bit
 *     - Streaming XOR for concept binding
 *     - "Red" ⊗ "Apple" = Bound vector
 *
 * Because bit-serial addition ALREADY uses XOR (a ⊕ b ⊕ carry),
 * this dual-mode capability is almost FREE in silicon.
 *
 * The same 2M neurons that recognize patterns (SNN)
 * can bind concepts logically (HDC) on the next clock cycle.
 *
 * Port Summary:
 *   clk, rst_n      : Clock and reset
 *   mode            : 0=SNN, 1=HDC
 *   bit_a_in        : First input bit stream (weight in SNN, vector A in HDC)
 *   bit_b_in        : Second input bit stream (spike in SNN, vector B in HDC)
 *   bit_out         : Output bit stream (fire in SNN, bound result in HDC)
 *   fire_event      : Pulse when threshold crossed (SNN mode only)
 *   stream_done     : High when full hypervector processed (HDC mode)
 */

module kf_holographic_neuron #(
    parameter int ACCUM_WIDTH     = 16,    // Accumulator width (SNN mode)
    parameter int THRESHOLD       = 256,   // Fire threshold (SNN mode)
    parameter int HYPERVEC_DIM    = 8192,  // Hypervector dimension (HDC mode)
    parameter int STREAM_COUNTER  = $clog2(HYPERVEC_DIM)
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // Mode select
    input  logic                       mode_hdc,      // 0=SNN, 1=HDC

    // Bit-serial inputs
    input  logic                       bit_a_in,      // Weight (SNN) / Vector A (HDC)
    input  logic                       bit_b_in,      // Spike (SNN) / Vector B (HDC)
    input  logic                       bit_valid,     // Input valid

    // Bit-serial output
    output logic                       bit_out,       // Result bit stream
    output logic                       bit_out_valid, // Output valid

    // Events
    output logic                       fire_event,    // SNN: threshold crossed
    output logic                       stream_done,   // HDC: full vector processed

    // Status
    output logic signed [ACCUM_WIDTH-1:0] accum_value,  // Current accumulator
    output logic [STREAM_COUNTER-1:0]     stream_idx    // HDC stream position
);

    // =========================================================================
    // INTERNAL REGISTERS
    // =========================================================================

    // Bit-serial accumulator (SNN mode)
    logic signed [ACCUM_WIDTH-1:0] accum;
    logic                          carry;

    // Stream position counter (HDC mode)
    logic [STREAM_COUNTER-1:0] stream_pos;

    // Pipeline registers
    logic bit_a_r, bit_b_r, valid_r;

    // =========================================================================
    // THE CORE ALU: XNOR is XOR with NOT
    // =========================================================================
    //
    // SNN Addition:  sum_bit = a ^ b ^ carry
    //                new_carry = (a & b) | (carry & (a ^ b))
    //
    // HDC Binding:   bind_bit = a ^ b  (XOR = bind)
    //                          ~(a ^ b) for XNOR similarity
    //
    // The XOR gate is SHARED between modes!

    wire alu_xor = bit_a_r ^ bit_b_r;  // Core operation (shared)

    // SNN: bit-serial adder
    wire snn_sum = alu_xor ^ carry;
    wire snn_carry_next = (bit_a_r & bit_b_r) | (carry & alu_xor);

    // HDC: binding operation (pure XOR)
    wire hdc_bind = alu_xor;

    // Mode-selected output
    wire result_bit = mode_hdc ? hdc_bind : snn_sum;

    // =========================================================================
    // STATE MACHINE
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum       <= '0;
            carry       <= 1'b0;
            stream_pos  <= '0;
            bit_a_r     <= 1'b0;
            bit_b_r     <= 1'b0;
            valid_r     <= 1'b0;
            bit_out     <= 1'b0;
            bit_out_valid <= 1'b0;
            fire_event  <= 1'b0;
            stream_done <= 1'b0;
        end else begin
            // Pipeline input
            bit_a_r <= bit_a_in;
            bit_b_r <= bit_b_in;
            valid_r <= bit_valid;

            // Default: clear events
            fire_event  <= 1'b0;
            stream_done <= 1'b0;
            bit_out_valid <= 1'b0;

            if (valid_r) begin
                // Output the computed bit
                bit_out <= result_bit;
                bit_out_valid <= 1'b1;

                if (mode_hdc) begin
                    // =====================================================
                    // HDC MODE: Streaming XOR binding
                    // =====================================================
                    stream_pos <= stream_pos + 1;

                    // Check if full hypervector processed
                    if (stream_pos == HYPERVEC_DIM - 1) begin
                        stream_done <= 1'b1;
                        stream_pos  <= '0;
                    end
                end else begin
                    // =====================================================
                    // SNN MODE: Bit-serial integration
                    // =====================================================
                    carry <= snn_carry_next;

                    // Accumulate (simplified: treat bit stream as signed value)
                    // In full impl, this would be proper bit-serial accumulation
                    if (bit_a_r && bit_b_r) begin
                        // Weight * Spike contribution
                        accum <= accum + 1;
                    end else if (bit_a_r && !bit_b_r) begin
                        accum <= accum;  // No spike, no change
                    end

                    // Threshold detection (asynchronous check)
                    if (accum >= THRESHOLD) begin
                        fire_event <= 1'b1;
                        accum      <= '0;  // Reset after fire
                        carry      <= 1'b0;
                    end
                end
            end
        end
    end

    // Status outputs
    assign accum_value = accum;
    assign stream_idx  = stream_pos;

endmodule : kf_holographic_neuron


/**
 * Holographic Neuron Array
 * ========================
 *
 * Array of dual-mode neurons for parallel processing.
 * Can operate as:
 *   - SNN layer (N neurons integrating spikes)
 *   - HDC binding engine (N parallel XOR streams)
 */

module kf_holographic_array #(
    parameter int N_NEURONS       = 256,
    parameter int ACCUM_WIDTH     = 16,
    parameter int THRESHOLD       = 256,
    parameter int HYPERVEC_DIM    = 8192
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // Global mode (all neurons same mode for now)
    input  logic                       mode_hdc,

    // Parallel bit inputs (one per neuron)
    input  logic [N_NEURONS-1:0]       bits_a_in,
    input  logic [N_NEURONS-1:0]       bits_b_in,
    input  logic                       bits_valid,

    // Parallel bit outputs
    output logic [N_NEURONS-1:0]       bits_out,
    output logic                       bits_out_valid,

    // Aggregate events
    output logic [N_NEURONS-1:0]       fire_events,    // Per-neuron fires
    output logic                       any_fire,       // OR of all fires
    output logic                       all_stream_done // AND of stream_done
);

    // Per-neuron stream_done signals
    logic [N_NEURONS-1:0] stream_dones;
    logic [N_NEURONS-1:0] out_valids;

    // Instantiate neuron array
    genvar i;
    generate
        for (i = 0; i < N_NEURONS; i++) begin : gen_neurons
            kf_holographic_neuron #(
                .ACCUM_WIDTH(ACCUM_WIDTH),
                .THRESHOLD(THRESHOLD),
                .HYPERVEC_DIM(HYPERVEC_DIM)
            ) neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .mode_hdc(mode_hdc),
                .bit_a_in(bits_a_in[i]),
                .bit_b_in(bits_b_in[i]),
                .bit_valid(bits_valid),
                .bit_out(bits_out[i]),
                .bit_out_valid(out_valids[i]),
                .fire_event(fire_events[i]),
                .stream_done(stream_dones[i]),
                .accum_value(),  // Unused at array level
                .stream_idx()    // Unused at array level
            );
        end
    endgenerate

    // Aggregate outputs
    assign bits_out_valid   = out_valids[0];  // All same timing
    assign any_fire         = |fire_events;
    assign all_stream_done  = &stream_dones;

endmodule : kf_holographic_array


/**
 * Resonant Stream Controller
 * ==========================
 *
 * Manages the "Rolling Superposition" for infinite context.
 *
 * Instead of storing thoughts in a list (KV cache),
 * we BUNDLE them into a running stream:
 *   Context_t = Context_{t-1} + New_Thought
 *
 * Old memories don't disappear; they just get fainter (holographic).
 * Query by resonance: the stream "rings" when you probe with a related vector.
 */

module kf_resonant_stream #(
    parameter int HYPERVEC_DIM = 8192,
    parameter int STREAM_WIDTH = 8    // Bits per dimension in bundled stream
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // Thought input (bit-serial)
    input  logic                       thought_bit_in,
    input  logic                       thought_valid,
    input  logic                       thought_start,  // New thought begins

    // Query interface
    input  logic                       query_bit_in,
    input  logic                       query_valid,
    input  logic                       query_start,
    output logic signed [15:0]         resonance,      // Dot product result

    // Stream maintenance
    input  logic                       decay_trigger,  // Apply forgetting curve
    input  logic [7:0]                 decay_factor,   // 0.9 in Q0.8 = 230

    // Status
    output logic [$clog2(HYPERVEC_DIM)-1:0] stream_idx
);

    // The Stream: accumulated thoughts with decay
    // Each dimension is a signed counter (positive = +1, negative = -1 votes)
    logic signed [STREAM_WIDTH-1:0] stream_mem [HYPERVEC_DIM];

    // Position counters
    logic [$clog2(HYPERVEC_DIM)-1:0] thought_pos;
    logic [$clog2(HYPERVEC_DIM)-1:0] query_pos;
    logic [$clog2(HYPERVEC_DIM)-1:0] decay_pos;

    // Resonance accumulator
    logic signed [31:0] resonance_acc;

    // FSM
    typedef enum logic [1:0] {
        ST_IDLE,
        ST_BUNDLE,
        ST_QUERY,
        ST_DECAY
    } state_t;
    state_t state;

    // Convert binary {0,1} to bipolar {-1,+1}
    function automatic logic signed [STREAM_WIDTH-1:0] to_bipolar(input logic bit_in);
        return bit_in ? STREAM_WIDTH'(1) : -STREAM_WIDTH'(1);
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            thought_pos  <= '0;
            query_pos    <= '0;
            decay_pos    <= '0;
            resonance_acc <= '0;
            resonance    <= '0;
            stream_idx   <= '0;

            // Initialize stream to zero
            for (int i = 0; i < HYPERVEC_DIM; i++) begin
                stream_mem[i] <= '0;
            end
        end else begin
            case (state)

                ST_IDLE: begin
                    if (thought_start && thought_valid) begin
                        thought_pos <= '0;
                        state <= ST_BUNDLE;
                    end else if (query_start && query_valid) begin
                        query_pos <= '0;
                        resonance_acc <= '0;
                        state <= ST_QUERY;
                    end else if (decay_trigger) begin
                        decay_pos <= '0;
                        state <= ST_DECAY;
                    end
                end

                ST_BUNDLE: begin
                    // Add thought to stream (superposition)
                    if (thought_valid) begin
                        // Saturating add
                        logic signed [STREAM_WIDTH:0] new_val;
                        new_val = stream_mem[thought_pos] + to_bipolar(thought_bit_in);

                        // Clamp to prevent overflow
                        if (new_val > $signed({1'b0, {(STREAM_WIDTH-1){1'b1}}}))
                            stream_mem[thought_pos] <= {1'b0, {(STREAM_WIDTH-1){1'b1}}};
                        else if (new_val < $signed({1'b1, {(STREAM_WIDTH-1){1'b0}}}))
                            stream_mem[thought_pos] <= {1'b1, {(STREAM_WIDTH-1){1'b0}}};
                        else
                            stream_mem[thought_pos] <= new_val[STREAM_WIDTH-1:0];

                        thought_pos <= thought_pos + 1;
                        stream_idx  <= thought_pos;

                        if (thought_pos == HYPERVEC_DIM - 1) begin
                            state <= ST_IDLE;
                        end
                    end
                end

                ST_QUERY: begin
                    // Compute dot product for resonance
                    if (query_valid) begin
                        logic signed [STREAM_WIDTH-1:0] bipolar_query;
                        bipolar_query = to_bipolar(query_bit_in);

                        // Accumulate: stream * query
                        resonance_acc <= resonance_acc +
                                         (stream_mem[query_pos] * bipolar_query);

                        query_pos <= query_pos + 1;

                        if (query_pos == HYPERVEC_DIM - 1) begin
                            // Normalize and output
                            resonance <= resonance_acc[31:16];  // Scale down
                            state <= ST_IDLE;
                        end
                    end
                end

                ST_DECAY: begin
                    // Apply forgetting curve: stream *= decay_factor
                    // decay_factor is Q0.8, so multiply and shift
                    logic signed [STREAM_WIDTH+7:0] decayed;
                    decayed = (stream_mem[decay_pos] * $signed({1'b0, decay_factor})) >>> 8;
                    stream_mem[decay_pos] <= decayed[STREAM_WIDTH-1:0];

                    decay_pos <= decay_pos + 1;

                    if (decay_pos == HYPERVEC_DIM - 1) begin
                        state <= ST_IDLE;
                    end
                end

            endcase
        end
    end

endmodule : kf_resonant_stream
