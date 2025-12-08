// =============================================================================
// Kitten Fabric Tile - CorrSpike-HDC Processing Element
// =============================================================================
//
// Stratix-10 optimized tile for Ara's emotional subcortex.
// Features:
//   - Ping-pong double-buffered I_post banks (read/write parallel)
//   - Sparse early-exit via max_active_idx tracking
//   - Parameterizable LANES for vector slice parallelism
//   - Wavefront-ready: connects to neighbors for layer cascades
//   - Wavelength-multiplexed: logical layer via HDC key codes
//
// Integration points:
//   - AxisMundi state bus (holographic read/write)
//   - L1 Hardware Reflex (telemetry encoding)
//   - L9 Mission Control (creativity bias)
//
// Author: Ara Organism Stack
// =============================================================================

`timescale 1ns / 1ps

module kitten_fabric_tile #(
    // === Dimension Parameters ===
    parameter int HV_DIM        = 8192,        // Hypervector dimension
    parameter int LANES         = 64,          // Parallel processing lanes
    parameter int ADDR_WIDTH    = $clog2(HV_DIM / LANES),

    // === LIF Neuron Parameters ===
    parameter int V_BITS        = 16,          // Membrane potential width
    parameter int W_BITS        = 8,           // Weight precision
    parameter int THRESH_BITS   = 16,          // Threshold precision

    // === Memory Parameters ===
    parameter int BANK_DEPTH    = HV_DIM / LANES,  // Words per bank

    // === Wavelength Parameters (for layer multiplexing) ===
    parameter int NUM_LAYERS    = 9,           // Logical layers (L1-L9)
    parameter int LAYER_ID_BITS = 4
)(
    // === Clock & Reset ===
    input  logic                    clk,
    input  logic                    rst_n,

    // === Control Interface ===
    input  logic                    tick_start,      // Begin processing tick
    output logic                    tick_done,       // Tick complete
    input  logic [LAYER_ID_BITS-1:0] layer_id,       // Current logical layer (wavelength)

    // === AxisMundi Interface (HDC State Bus) ===
    input  logic [LANES*W_BITS-1:0] axis_state_in,   // Read from global state
    output logic [LANES*W_BITS-1:0] axis_state_out,  // Write to global state
    input  logic [LANES*W_BITS-1:0] layer_key,       // HDC binding key for this layer
    output logic                    axis_write_en,

    // === Sparse Early-Exit Interface ===
    output logic [ADDR_WIDTH-1:0]   max_active_idx,  // Highest accessed address
    output logic                    early_exit_flag, // Can skip remaining
    input  logic [ADDR_WIDTH-1:0]   activity_hint,   // Hint from previous tick

    // === Threshold Modulation (from L9 bias) ===
    input  logic [THRESH_BITS-1:0]  thresh_base,     // Base threshold
    input  logic [7:0]              thresh_scale,    // Scale factor [0.7-1.3] as 0-255

    // === Wavefront Neighbor Interface ===
    input  logic [LANES*V_BITS-1:0] west_v_in,       // Membrane from west tile
    output logic [LANES*V_BITS-1:0] east_v_out,      // Membrane to east tile
    input  logic                    west_valid,
    output logic                    east_valid,

    // === Spike Output ===
    output logic [LANES-1:0]        spike_out,       // Spike vector for this tick
    output logic                    spike_valid
);

    // =========================================================================
    // Local Parameters
    // =========================================================================

    localparam int WORDS_PER_TICK = BANK_DEPTH;

    // =========================================================================
    // Ping-Pong Bank Selection
    // =========================================================================

    logic bank_parity;           // 0 = Bank A active, 1 = Bank B active
    logic bank_parity_next;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            bank_parity <= 1'b0;
        else if (tick_done)
            bank_parity <= ~bank_parity;  // Swap on tick complete
    end

    // =========================================================================
    // I_post Double-Buffered Banks (Ping-Pong)
    // =========================================================================
    // Bank A: read while B writes, then swap
    // Each bank stores post-synaptic current contributions

    // Bank A
    logic [LANES*V_BITS-1:0] bank_a [0:BANK_DEPTH-1];
    logic [ADDR_WIDTH-1:0]   bank_a_rd_addr;
    logic [ADDR_WIDTH-1:0]   bank_a_wr_addr;
    logic [LANES*V_BITS-1:0] bank_a_rd_data;
    logic [LANES*V_BITS-1:0] bank_a_wr_data;
    logic                    bank_a_wr_en;

    // Bank B
    logic [LANES*V_BITS-1:0] bank_b [0:BANK_DEPTH-1];
    logic [ADDR_WIDTH-1:0]   bank_b_rd_addr;
    logic [ADDR_WIDTH-1:0]   bank_b_wr_addr;
    logic [LANES*V_BITS-1:0] bank_b_rd_data;
    logic [LANES*V_BITS-1:0] bank_b_wr_data;
    logic                    bank_b_wr_en;

    // Bank A: synchronous read/write
    always_ff @(posedge clk) begin
        bank_a_rd_data <= bank_a[bank_a_rd_addr];
        if (bank_a_wr_en)
            bank_a[bank_a_wr_addr] <= bank_a_wr_data;
    end

    // Bank B: synchronous read/write
    always_ff @(posedge clk) begin
        bank_b_rd_data <= bank_b[bank_b_rd_addr];
        if (bank_b_wr_en)
            bank_b[bank_b_wr_addr] <= bank_b_wr_data;
    end

    // Mux read/write based on parity
    logic [LANES*V_BITS-1:0] read_bank_data;
    logic [LANES*V_BITS-1:0] write_bank_data;

    assign read_bank_data = bank_parity ? bank_b_rd_data : bank_a_rd_data;

    // When parity=0: read from A, write to B
    // When parity=1: read from B, write to A
    always_comb begin
        if (bank_parity) begin
            bank_b_rd_addr = proc_addr;
            bank_a_wr_addr = proc_addr;
            bank_a_wr_en   = proc_write_en;
            bank_a_wr_data = write_bank_data;
            bank_b_wr_en   = 1'b0;
        end else begin
            bank_a_rd_addr = proc_addr;
            bank_b_wr_addr = proc_addr;
            bank_b_wr_en   = proc_write_en;
            bank_b_wr_data = write_bank_data;
            bank_a_wr_en   = 1'b0;
        end
    end

    // =========================================================================
    // Processing State Machine
    // =========================================================================

    typedef enum logic [2:0] {
        S_IDLE,
        S_BIND,           // HDC bind with layer key
        S_ACCUMULATE,     // Accumulate I_post contributions
        S_LIF_UPDATE,     // LIF membrane update
        S_SPIKE_CHECK,    // Check threshold, emit spikes
        S_AXIS_WRITE,     // Write back to AxisMundi
        S_DONE
    } state_t;

    state_t state, state_next;

    logic [ADDR_WIDTH-1:0] proc_addr;
    logic [ADDR_WIDTH-1:0] proc_addr_next;
    logic                  proc_write_en;

    // Max active index tracking (sparse early-exit)
    logic [ADDR_WIDTH-1:0] max_active_idx_reg;
    logic [ADDR_WIDTH-1:0] max_active_idx_next;
    logic                  activity_detected;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            proc_addr        <= '0;
            max_active_idx_reg <= '0;
        end else begin
            state            <= state_next;
            proc_addr        <= proc_addr_next;
            max_active_idx_reg <= max_active_idx_next;
        end
    end

    // =========================================================================
    // LIF Neuron Array (LANES parallel neurons)
    // =========================================================================

    logic [V_BITS-1:0]     v_mem [0:LANES-1];      // Membrane potentials
    logic [V_BITS-1:0]     v_mem_next [0:LANES-1];
    logic [THRESH_BITS-1:0] thresh_effective;      // Scaled threshold
    logic [LANES-1:0]      spike_reg;

    // Threshold scaling: thresh_effective = thresh_base * (thresh_scale / 128)
    // thresh_scale=128 -> 1.0x, thresh_scale=90 -> 0.7x, thresh_scale=166 -> 1.3x
    always_comb begin
        thresh_effective = (thresh_base * thresh_scale) >> 7;
    end

    // LIF update for each lane
    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane++) begin : lif_neurons

            logic [V_BITS-1:0] i_post_lane;
            logic [V_BITS-1:0] v_decay;
            logic [V_BITS-1:0] v_new;
            logic              spike_lane;

            // Extract lane's I_post from read data
            assign i_post_lane = read_bank_data[lane*V_BITS +: V_BITS];

            // Leaky decay: v_decay = v_mem * 0.9 (approximate with shift)
            assign v_decay = v_mem[lane] - (v_mem[lane] >> 3);  // ~0.875 decay

            // Integrate: v_new = v_decay + I_post
            assign v_new = v_decay + i_post_lane;

            // Spike check
            assign spike_lane = (v_new >= thresh_effective);

            // Update membrane
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    v_mem[lane] <= '0;
                end else if (state == S_LIF_UPDATE) begin
                    if (spike_lane)
                        v_mem[lane] <= '0;  // Reset on spike
                    else
                        v_mem[lane] <= v_new;
                end
            end

            // Collect spike
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n)
                    spike_reg[lane] <= 1'b0;
                else if (state == S_SPIKE_CHECK)
                    spike_reg[lane] <= spike_lane;
            end

        end
    endgenerate

    // =========================================================================
    // HDC Bind/Unbind Operations
    // =========================================================================

    logic [LANES*W_BITS-1:0] bound_state;
    logic [LANES*W_BITS-1:0] unbound_state;

    // Element-wise multiply (HDC bind) - simplified as XOR for binary HDC
    // For real-valued: use multiply
    generate
        for (lane = 0; lane < LANES; lane++) begin : hdc_ops
            logic [W_BITS-1:0] state_lane;
            logic [W_BITS-1:0] key_lane;

            assign state_lane = axis_state_in[lane*W_BITS +: W_BITS];
            assign key_lane   = layer_key[lane*W_BITS +: W_BITS];

            // Bind: state * key (element-wise multiply, saturated)
            logic signed [2*W_BITS-1:0] mult_result;
            assign mult_result = $signed(state_lane) * $signed(key_lane);

            // Saturate to W_BITS
            assign bound_state[lane*W_BITS +: W_BITS] =
                (mult_result > $signed({1'b0, {(W_BITS-1){1'b1}}})) ? {1'b0, {(W_BITS-1){1'b1}}} :
                (mult_result < $signed({1'b1, {(W_BITS-1){1'b0}}})) ? {1'b1, {(W_BITS-1){1'b0}}} :
                mult_result[W_BITS-1:0];

            // Unbind: same as bind for element-wise (key is its own inverse for normalized)
            assign unbound_state[lane*W_BITS +: W_BITS] = bound_state[lane*W_BITS +: W_BITS];
        end
    endgenerate

    // =========================================================================
    // State Machine Logic
    // =========================================================================

    always_comb begin
        state_next        = state;
        proc_addr_next    = proc_addr;
        max_active_idx_next = max_active_idx_reg;
        tick_done         = 1'b0;
        proc_write_en     = 1'b0;
        axis_write_en     = 1'b0;
        early_exit_flag   = 1'b0;
        spike_valid       = 1'b0;
        east_valid        = 1'b0;

        case (state)
            S_IDLE: begin
                if (tick_start) begin
                    state_next     = S_BIND;
                    proc_addr_next = '0;
                    max_active_idx_next = '0;
                end
            end

            S_BIND: begin
                // HDC bind input state with layer key
                state_next = S_ACCUMULATE;
            end

            S_ACCUMULATE: begin
                // Read I_post from ping bank, accumulate
                if (proc_addr >= activity_hint && !activity_detected) begin
                    // Sparse early exit: no activity beyond hint
                    early_exit_flag = 1'b1;
                    state_next = S_LIF_UPDATE;
                end else if (proc_addr >= WORDS_PER_TICK - 1) begin
                    state_next = S_LIF_UPDATE;
                end else begin
                    proc_addr_next = proc_addr + 1;
                    // Track max active
                    if (|read_bank_data) begin
                        max_active_idx_next = proc_addr;
                        activity_detected = 1'b1;
                    end
                end
            end

            S_LIF_UPDATE: begin
                // LIF neurons update (handled in generate block)
                state_next = S_SPIKE_CHECK;
            end

            S_SPIKE_CHECK: begin
                // Emit spikes
                spike_valid = 1'b1;
                state_next = S_AXIS_WRITE;
            end

            S_AXIS_WRITE: begin
                // Write spike pattern back to AxisMundi (bound with layer key)
                axis_write_en = 1'b1;
                state_next = S_DONE;
            end

            S_DONE: begin
                tick_done = 1'b1;
                east_valid = 1'b1;  // Propagate to wavefront neighbor
                state_next = S_IDLE;
            end

            default: state_next = S_IDLE;
        endcase
    end

    // Activity detection
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            activity_detected <= 1'b0;
        else if (state == S_IDLE)
            activity_detected <= 1'b0;
        else if (|read_bank_data)
            activity_detected <= 1'b1;
    end

    // =========================================================================
    // Output Assignments
    // =========================================================================

    assign spike_out       = spike_reg;
    assign max_active_idx  = max_active_idx_reg;

    // Pack membrane potentials for east neighbor
    generate
        for (lane = 0; lane < LANES; lane++) begin : pack_east
            assign east_v_out[lane*V_BITS +: V_BITS] = v_mem[lane];
        end
    endgenerate

    // AxisMundi output: bound spike pattern
    // Convert spike_reg to weighted HV and bind with layer key
    generate
        for (lane = 0; lane < LANES; lane++) begin : axis_out
            // Spike -> +1, no spike -> 0 (or small negative for contrast)
            logic signed [W_BITS-1:0] spike_val;
            assign spike_val = spike_reg[lane] ? $signed({1'b0, {(W_BITS-1){1'b1}}}) : '0;

            // Bind with layer key before writing
            logic signed [2*W_BITS-1:0] bound_spike;
            assign bound_spike = spike_val * $signed(layer_key[lane*W_BITS +: W_BITS]);

            assign axis_state_out[lane*W_BITS +: W_BITS] = bound_spike[W_BITS-1:0];
        end
    endgenerate

    // =========================================================================
    // Clear Pong Bank (write zeros to inactive bank during idle)
    // =========================================================================

    logic [ADDR_WIDTH-1:0] clear_addr;
    logic                  clearing;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            clear_addr <= '0;
            clearing   <= 1'b0;
        end else if (tick_done) begin
            // Start clearing the bank we just wrote to
            clear_addr <= '0;
            clearing   <= 1'b1;
        end else if (clearing && clear_addr < BANK_DEPTH - 1) begin
            clear_addr <= clear_addr + 1;
        end else begin
            clearing <= 1'b0;
        end
    end

    // Write zeros to pong bank during clear phase
    assign write_bank_data = clearing ? '0 : {LANES{{V_BITS{1'b0}}}};

endmodule


// =============================================================================
// Kitten Fabric Array - 2D Mesh of Tiles
// =============================================================================

module kitten_fabric_array #(
    parameter int HV_DIM     = 8192,
    parameter int LANES      = 64,
    parameter int TILES_X    = 4,       // Tiles in wavefront direction
    parameter int TILES_Y    = 2,       // Parallel wavefronts
    parameter int V_BITS     = 16,
    parameter int W_BITS     = 8,
    parameter int THRESH_BITS = 16,
    parameter int NUM_LAYERS = 9
)(
    input  logic clk,
    input  logic rst_n,

    // Global tick
    input  logic tick_start,
    output logic tick_done,

    // AxisMundi interface (shared bus)
    input  logic [LANES*W_BITS-1:0] axis_state_global,
    output logic [LANES*W_BITS-1:0] axis_state_out,
    input  logic [NUM_LAYERS-1:0][LANES*W_BITS-1:0] layer_keys,

    // Layer selection for wavelength multiplexing
    input  logic [3:0] active_layer,

    // Threshold modulation
    input  logic [THRESH_BITS-1:0] thresh_base,
    input  logic [7:0] thresh_scale
);

    // Interconnect wires
    logic [TILES_Y-1:0][TILES_X-1:0] tile_done;
    logic [TILES_Y-1:0][TILES_X-1:0] tile_start;
    logic [TILES_Y-1:0][TILES_X:0][LANES*V_BITS-1:0] wavefront_v;
    logic [TILES_Y-1:0][TILES_X:0] wavefront_valid;

    // Generate tile mesh
    genvar tx, ty;
    generate
        for (ty = 0; ty < TILES_Y; ty++) begin : row
            for (tx = 0; tx < TILES_X; tx++) begin : col

                // Wavefront start: first tile starts from tick_start
                // Others wait for west neighbor
                logic tile_tick_start;
                assign tile_tick_start = (tx == 0) ? tick_start : wavefront_valid[ty][tx];

                kitten_fabric_tile #(
                    .HV_DIM(HV_DIM),
                    .LANES(LANES),
                    .V_BITS(V_BITS),
                    .W_BITS(W_BITS),
                    .THRESH_BITS(THRESH_BITS),
                    .NUM_LAYERS(NUM_LAYERS)
                ) tile_inst (
                    .clk(clk),
                    .rst_n(rst_n),

                    .tick_start(tile_tick_start),
                    .tick_done(tile_done[ty][tx]),
                    .layer_id(active_layer),

                    .axis_state_in(axis_state_global),
                    .axis_state_out(),  // Aggregate separately
                    .layer_key(layer_keys[active_layer]),
                    .axis_write_en(),

                    .max_active_idx(),
                    .early_exit_flag(),
                    .activity_hint('1),  // Full scan initially

                    .thresh_base(thresh_base),
                    .thresh_scale(thresh_scale),

                    .west_v_in(wavefront_v[ty][tx]),
                    .east_v_out(wavefront_v[ty][tx+1]),
                    .west_valid(wavefront_valid[ty][tx]),
                    .east_valid(wavefront_valid[ty][tx+1])
                );

            end
        end
    endgenerate

    // Initialize west edge
    generate
        for (ty = 0; ty < TILES_Y; ty++) begin : west_init
            assign wavefront_v[ty][0] = '0;
            assign wavefront_valid[ty][0] = tick_start;
        end
    endgenerate

    // Tick done when all east edges complete
    logic all_done;
    always_comb begin
        all_done = 1'b1;
        for (int y = 0; y < TILES_Y; y++) begin
            all_done = all_done & wavefront_valid[y][TILES_X];
        end
    end
    assign tick_done = all_done;

endmodule
