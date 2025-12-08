// -----------------------------------------------------------------------------
// kf_ensemble_choir.sv
// Iteration 31: The Choir - Ensemble Lanes & Consensus Field
//
// KEY INSIGHT: Same job to multiple lanes with perturbations, then vote.
// You get TWO outputs:
//   1. Robust result (ensemble consensus)
//   2. Confidence signal (how much lanes agreed)
//
// High agreement → Steward auto-acts
// Low agreement  → Escalate to human / bigger model
//
// Architecture:
//   - Broadcast: Fan-out job to N lanes
//   - Perturb: Each lane gets different noise seed / threshold
//   - Compute: Lanes run in parallel (SNN or HDC)
//   - Reduce: Vote / average / consensus
//   - Output: Result + confidence
// -----------------------------------------------------------------------------
module kf_ensemble_choir #(
    parameter integer N_LANES       = 8,     // Lanes in the choir
    parameter integer HV_DIM        = 8192,  // Hypervector dimension
    parameter integer STREAM_BITS   = $clog2(HV_DIM),
    parameter integer THRESH_WIDTH  = 16,
    parameter integer CONF_WIDTH    = 8      // Confidence precision (0-255)
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // =========================================================================
    // Job Input (from host / controller)
    // =========================================================================
    input  wire                     job_valid,
    output wire                     job_ready,
    input  wire [1:0]               job_mode,       // 00=SNN, 01=HDC_BIND, 10=HDC_BUNDLE
    input  wire [2:0]               reduction_op,   // 000=majority, 001=average, 010=min, 011=max
    input  wire [3:0]               ensemble_size,  // 1-N_LANES

    // Hypervector inputs (bit-serial stream)
    input  wire                     hv_a_bit,
    input  wire                     hv_b_bit,
    input  wire                     hv_valid,
    input  wire                     hv_last,        // Last bit of hypervector

    // =========================================================================
    // Result Output
    // =========================================================================
    output reg                      result_valid,
    output reg                      result_bit,     // Consensus result (bit-serial)
    output reg  [CONF_WIDTH-1:0]    confidence,     // 0=total disagreement, 255=unanimous
    output reg                      result_last,

    // =========================================================================
    // Per-Lane Configuration (set by host at init)
    // =========================================================================
    input  wire [N_LANES-1:0]       lane_enable,
    input  wire [7:0]               noise_seeds [N_LANES],
    input  wire [THRESH_WIDTH-1:0]  thresholds  [N_LANES],

    // =========================================================================
    // Debug / Stats
    // =========================================================================
    output wire [N_LANES-1:0]       lane_results,   // Raw results for debug
    output wire [N_LANES-1:0]       lane_done
);

    // =========================================================================
    // LANE INSTANTIATION
    // =========================================================================

    // Per-lane signals
    wire [N_LANES-1:0] lane_out_bits;
    wire [N_LANES-1:0] lane_out_valid;
    wire [N_LANES-1:0] lane_fire;

    // Perturbed inputs per lane
    wire [N_LANES-1:0] hv_a_perturbed;
    wire [N_LANES-1:0] hv_b_perturbed;

    // Stream position counter
    reg [STREAM_BITS-1:0] stream_pos;
    reg job_active;

    // Generate perturbation (XOR with LFSR-derived noise)
    genvar i;
    generate
        for (i = 0; i < N_LANES; i = i + 1) begin : gen_lanes

            // Simple LFSR for per-lane noise
            reg [7:0] lfsr;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    lfsr <= noise_seeds[i];
                end else if (hv_valid && lane_enable[i]) begin
                    // LFSR tap at bits 7,5,4,3 (maximal length for 8-bit)
                    lfsr <= {lfsr[6:0], lfsr[7] ^ lfsr[5] ^ lfsr[4] ^ lfsr[3]};
                end
            end

            // Perturb input with low probability (noise_bit AND lfsr LSB)
            // This gives each lane a slightly different "view"
            assign hv_a_perturbed[i] = hv_a_bit ^ (lfsr[0] & lfsr[7]);
            assign hv_b_perturbed[i] = hv_b_bit ^ (lfsr[1] & lfsr[6]);

            // Instantiate bit-serial neuron
            bit_serial_neuron #(
                .WEIGHT_WIDTH(8),
                .ACC_WIDTH(16),
                .THRESH_WIDTH(THRESH_WIDTH)
            ) lane_neuron (
                .clk(clk),
                .rst_n(rst_n),
                .mode_hdc(job_mode[0]),  // bit 0 selects HDC vs SNN
                .in_valid(hv_valid && lane_enable[i] && job_active),
                .weight_bit(hv_a_perturbed[i]),
                .state_bit_in(hv_b_perturbed[i]),
                .start(job_valid && job_ready),
                .threshold(thresholds[i]),
                .state_bit_out(lane_out_bits[i]),
                .fire_event(lane_fire[i])
            );

            assign lane_out_valid[i] = hv_valid && lane_enable[i];
        end
    endgenerate

    // =========================================================================
    // REDUCTION LOGIC
    // =========================================================================

    // Count how many lanes output '1' vs '0'
    reg [3:0] ones_count;
    reg [3:0] active_lanes;

    always @(*) begin
        ones_count = 0;
        active_lanes = 0;
        for (int j = 0; j < N_LANES; j = j + 1) begin
            if (lane_enable[j] && (j < ensemble_size)) begin
                active_lanes = active_lanes + 1;
                if (lane_out_bits[j]) begin
                    ones_count = ones_count + 1;
                end
            end
        end
    end

    // Majority vote
    wire majority_result = (ones_count > (active_lanes >> 1));

    // Confidence = |agreement - 0.5| * 2, scaled to 0-255
    // If all agree: confidence = 255
    // If 50/50 split: confidence = 0
    wire [7:0] agreement_ratio;
    wire [7:0] disagreement;

    assign agreement_ratio = (ones_count * 255) / (active_lanes > 0 ? active_lanes : 1);
    assign disagreement = (agreement_ratio > 127) ?
                          (agreement_ratio - 127) :
                          (127 - agreement_ratio);

    wire [CONF_WIDTH-1:0] computed_confidence = disagreement << 1;  // Scale to 0-255

    // =========================================================================
    // OUTPUT SELECTION BASED ON REDUCTION OP
    // =========================================================================

    reg reduced_bit;

    always @(*) begin
        case (reduction_op)
            3'b000: reduced_bit = majority_result;              // Majority vote
            3'b001: reduced_bit = (ones_count >= (active_lanes >> 1)); // Average (same as majority for bits)
            3'b010: reduced_bit = &lane_out_bits[N_LANES-1:0] & lane_enable; // AND (all must agree on 1)
            3'b011: reduced_bit = |lane_out_bits[N_LANES-1:0] & lane_enable; // OR (any 1)
            default: reduced_bit = majority_result;
        endcase
    end

    // =========================================================================
    // JOB STATE MACHINE
    // =========================================================================

    localparam ST_IDLE    = 2'b00;
    localparam ST_STREAM  = 2'b01;
    localparam ST_OUTPUT  = 2'b10;

    reg [1:0] state;

    assign job_ready = (state == ST_IDLE);
    assign lane_results = lane_out_bits;
    assign lane_done = {N_LANES{state == ST_IDLE}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            stream_pos   <= '0;
            job_active   <= 1'b0;
            result_valid <= 1'b0;
            result_bit   <= 1'b0;
            result_last  <= 1'b0;
            confidence   <= '0;
        end else begin
            // Defaults
            result_valid <= 1'b0;
            result_last  <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (job_valid) begin
                        state      <= ST_STREAM;
                        stream_pos <= '0;
                        job_active <= 1'b1;
                    end
                end

                ST_STREAM: begin
                    if (hv_valid) begin
                        // Output reduced result bit
                        result_valid <= 1'b1;
                        result_bit   <= reduced_bit;
                        confidence   <= computed_confidence;
                        stream_pos   <= stream_pos + 1;

                        if (hv_last) begin
                            result_last <= 1'b1;
                            state       <= ST_IDLE;
                            job_active  <= 1'b0;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule


// -----------------------------------------------------------------------------
// kf_choir_tile.sv
// A tile containing an ensemble choir with local BRAM for HV storage
// -----------------------------------------------------------------------------
module kf_choir_tile #(
    parameter integer N_LANES    = 8,
    parameter integer HV_DIM     = 8192,
    parameter integer N_HV_SLOTS = 64,    // How many HVs can we store on-chip
    parameter integer THRESH_WIDTH = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // AXI-Lite config interface (simplified)
    input  wire                     cfg_we,
    input  wire [15:0]              cfg_addr,
    input  wire [31:0]              cfg_wdata,
    output reg  [31:0]              cfg_rdata,

    // Job interface
    input  wire                     job_valid,
    output wire                     job_ready,
    input  wire [7:0]               job_opcode,    // {ensemble_size[3:0], reduction[2:0], mode[0]}
    input  wire [5:0]               hv_a_slot,     // Which stored HV to use as A
    input  wire [5:0]               hv_b_slot,     // Which stored HV to use as B

    // Result
    output wire                     result_valid,
    output wire                     result_bit,
    output wire [7:0]               confidence,
    output wire                     result_last,

    // Status
    output wire                     tile_busy
);

    // =========================================================================
    // HV STORAGE (BRAM)
    // =========================================================================

    // Store HVs as packed bits
    localparam HV_WORDS = (HV_DIM + 63) / 64;

    (* ram_style = "block" *)
    reg [63:0] hv_bram [N_HV_SLOTS * HV_WORDS];

    // Read pointers
    reg [5:0] rd_slot_a, rd_slot_b;
    reg [$clog2(HV_WORDS)-1:0] rd_word;
    reg [5:0] rd_bit;

    wire [$clog2(N_HV_SLOTS * HV_WORDS)-1:0] bram_addr_a = rd_slot_a * HV_WORDS + rd_word;
    wire [$clog2(N_HV_SLOTS * HV_WORDS)-1:0] bram_addr_b = rd_slot_b * HV_WORDS + rd_word;

    reg [63:0] word_a, word_b;

    // Stream bits from stored HVs
    wire hv_a_bit_stream = word_a[rd_bit];
    wire hv_b_bit_stream = word_b[rd_bit];

    // =========================================================================
    // CHOIR INSTANCE
    // =========================================================================

    // Parse opcode
    wire [3:0] ensemble_size = job_opcode[7:4];
    wire [2:0] reduction_op  = job_opcode[3:1];
    wire [1:0] job_mode      = {1'b0, job_opcode[0]};

    // Lane config (simplified: same for all lanes in this version)
    wire [N_LANES-1:0] lane_enable = (1 << ensemble_size) - 1;
    wire [7:0] noise_seeds [N_LANES];
    wire [THRESH_WIDTH-1:0] thresholds [N_LANES];

    genvar k;
    generate
        for (k = 0; k < N_LANES; k = k + 1) begin : gen_config
            assign noise_seeds[k] = 8'h37 + k * 17;  // Different seed per lane
            assign thresholds[k]  = 16'd100 + k * 10; // Slightly different thresholds
        end
    endgenerate

    // Streaming control
    reg stream_active;
    reg stream_valid;
    reg stream_last;
    reg [$clog2(HV_DIM)-1:0] stream_pos;

    kf_ensemble_choir #(
        .N_LANES(N_LANES),
        .HV_DIM(HV_DIM),
        .THRESH_WIDTH(THRESH_WIDTH)
    ) choir_inst (
        .clk(clk),
        .rst_n(rst_n),
        .job_valid(job_valid && !stream_active),
        .job_ready(job_ready),
        .job_mode(job_mode),
        .reduction_op(reduction_op),
        .ensemble_size(ensemble_size),
        .hv_a_bit(hv_a_bit_stream),
        .hv_b_bit(hv_b_bit_stream),
        .hv_valid(stream_valid),
        .hv_last(stream_last),
        .result_valid(result_valid),
        .result_bit(result_bit),
        .confidence(confidence),
        .result_last(result_last),
        .lane_enable(lane_enable),
        .noise_seeds(noise_seeds),
        .thresholds(thresholds),
        .lane_results(),
        .lane_done()
    );

    assign tile_busy = stream_active;

    // =========================================================================
    // STREAMING STATE MACHINE
    // =========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            stream_active <= 1'b0;
            stream_valid  <= 1'b0;
            stream_last   <= 1'b0;
            stream_pos    <= '0;
            rd_slot_a     <= '0;
            rd_slot_b     <= '0;
            rd_word       <= '0;
            rd_bit        <= '0;
            word_a        <= '0;
            word_b        <= '0;
        end else begin
            stream_valid <= 1'b0;
            stream_last  <= 1'b0;

            if (job_valid && !stream_active) begin
                // Start new job
                stream_active <= 1'b1;
                stream_pos    <= '0;
                rd_slot_a     <= hv_a_slot;
                rd_slot_b     <= hv_b_slot;
                rd_word       <= '0;
                rd_bit        <= '0;
                // Pre-load first words
                word_a <= hv_bram[hv_a_slot * HV_WORDS];
                word_b <= hv_bram[hv_b_slot * HV_WORDS];
            end else if (stream_active) begin
                stream_valid <= 1'b1;
                stream_pos   <= stream_pos + 1;

                // Advance bit pointer
                if (rd_bit == 63) begin
                    rd_bit  <= 0;
                    rd_word <= rd_word + 1;
                    // Load next words
                    word_a <= hv_bram[rd_slot_a * HV_WORDS + rd_word + 1];
                    word_b <= hv_bram[rd_slot_b * HV_WORDS + rd_word + 1];
                end else begin
                    rd_bit <= rd_bit + 1;
                end

                // Check for last bit
                if (stream_pos == HV_DIM - 1) begin
                    stream_last   <= 1'b1;
                    stream_active <= 1'b0;
                end
            end
        end
    end

    // =========================================================================
    // CONFIG INTERFACE (write HVs to BRAM)
    // =========================================================================

    always @(posedge clk) begin
        if (cfg_we) begin
            // addr[15:6] = slot, addr[5:0] = word within slot (upper), addr[1:0] = 32-bit half
            if (cfg_addr[1]) begin
                hv_bram[cfg_addr[15:2]][63:32] <= cfg_wdata;
            end else begin
                hv_bram[cfg_addr[15:2]][31:0] <= cfg_wdata;
            end
        end
    end

endmodule
