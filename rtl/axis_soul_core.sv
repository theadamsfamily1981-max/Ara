//-----------------------------------------------------------------------------
// Ara Soul Core (Revised Architecture)
// Iteration 27: The Engineer's Soul
//
// A streaming, chunk-based HDC associative memory with on-chip plasticity.
// Combines inference (query) and learning (plasticity) in a unified core.
//
// Target: Intel Stratix-10 / Agilex
// Fmax:   350 MHz
//
// Architecture:
//   - Banked BRAM for parallel row access
//   - Streaming XOR + Popcount for inference
//   - Background plasticity engine on Port B
//   - AXI-Lite control + AXI-Stream data
//
// Key Parameters:
//   DIM:        16384 (Total Hypervector Dimension)
//   CHUNK:      512   (Bits processed per cycle)
//   ROWS:       2048  (Number of attractors)
//   PAR_LAYERS: 8     (Rows processed in parallel)
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps
`default_nettype none

module axis_soul_core #(
    parameter int DIM        = 16384,   // Hypervector dimension
    parameter int CHUNK      = 512,     // Bits per memory access
    parameter int ROWS       = 2048,    // Number of attractors
    parameter int PAR_LAYERS = 8,       // Parallel inference lanes
    parameter int ACC_WIDTH  = 7        // Accumulator precision (7-bit signed)
)(
    input  wire             clk,
    input  wire             rst_n,

    // =========================================================================
    // AXI-Stream Input (Query / State HV)
    // =========================================================================
    input  wire [CHUNK-1:0] s_axis_tdata,
    input  wire             s_axis_tvalid,
    output wire             s_axis_tready,
    input  wire             s_axis_tlast,

    // =========================================================================
    // AXI-Stream Output (Top-K Results)
    // =========================================================================
    output reg  [31:0]      m_axis_tdata,   // Score data
    output reg              m_axis_tvalid,
    input  wire             m_axis_tready,
    output reg              m_axis_tlast,

    // =========================================================================
    // AXI-Lite Control (Commands & Registers)
    // =========================================================================
    input  wire [7:0]       ctrl_addr,
    input  wire             ctrl_write,
    input  wire [31:0]      ctrl_wdata,
    output reg  [31:0]      ctrl_rdata,

    // =========================================================================
    // Status
    // =========================================================================
    output wire             o_busy,
    output wire             o_inference_done,
    output wire             o_plasticity_done
);

    // =========================================================================
    // 1. LOCAL PARAMETERS
    // =========================================================================
    localparam int ROWS_PER_BANK  = ROWS / PAR_LAYERS;
    localparam int CHUNKS_PER_ROW = DIM / CHUNK;
    localparam int BANK_DEPTH     = ROWS_PER_BANK * CHUNKS_PER_ROW;
    localparam int BANK_ADDR_W    = $clog2(BANK_DEPTH);
    localparam int ROW_ADDR_W     = $clog2(ROWS);
    localparam int CHUNK_ADDR_W   = $clog2(CHUNKS_PER_ROW);

    // Timing constants (cycles @ 350 MHz)
    localparam int CYCLES_PER_QUERY = CHUNKS_PER_ROW;  // 32 cycles
    localparam int CYCLES_PER_PLASTY = ROWS * CHUNKS_PER_ROW * 3;  // ~200k cycles

    // =========================================================================
    // 2. MEMORY ARCHITECTURE (Banked BRAMs)
    // =========================================================================
    // Each bank holds ROWS/PAR_LAYERS rows of CHUNKS_PER_ROW chunks each.
    // This allows PAR_LAYERS rows to be read simultaneously during inference.

    // Weight Memory (1-bit bipolar stored as 0/1)
    (* ram_style = "block" *)
    reg [CHUNK-1:0] mem_weights [PAR_LAYERS-1:0][BANK_DEPTH-1:0];

    // Accumulator Memory (Signed 7-bit per position)
    // Stored as packed CHUNK*ACC_WIDTH bits per entry
    (* ram_style = "block" *)
    reg [CHUNK*ACC_WIDTH-1:0] mem_accums [PAR_LAYERS-1:0][BANK_DEPTH-1:0];

    // =========================================================================
    // 3. FSM STATES
    // =========================================================================
    typedef enum logic [3:0] {
        S_IDLE        = 4'd0,
        S_LOAD_QUERY  = 4'd1,
        S_INFER_READ  = 4'd2,
        S_INFER_XOR   = 4'd3,
        S_INFER_POP   = 4'd4,
        S_INFER_ACCUM = 4'd5,
        S_INFER_NEXT  = 4'd6,
        S_OUTPUT      = 4'd7,
        S_LEARN_LATCH = 4'd8,
        S_LEARN_READ  = 4'd9,
        S_LEARN_UPDATE= 4'd10,
        S_LEARN_WRITE = 4'd11,
        S_LEARN_NEXT  = 4'd12,
        S_DONE        = 4'd13
    } state_t;

    state_t state, next_state;

    // =========================================================================
    // 4. CONTROL REGISTERS (AXI-Lite accessible)
    // =========================================================================
    // Addr 0x00: Command (W): 0=NOP, 1=Query, 2=Learn
    // Addr 0x04: Reward (W): Signed 8-bit reward for learning
    // Addr 0x08: Status (R): {busy, infer_done, plast_done, ...}
    // Addr 0x0C: Query Count (R)
    // Addr 0x10: Learn Count (R)
    // Addr 0x14: Top Score (R)
    // Addr 0x18: Top Row Index (R)

    localparam CMD_NOP   = 32'd0;
    localparam CMD_QUERY = 32'd1;
    localparam CMD_LEARN = 32'd2;

    reg [31:0] cmd_reg;
    reg signed [7:0] reward_reg;
    reg [31:0] query_count;
    reg [31:0] learn_count;
    reg [31:0] top_score;
    reg [ROW_ADDR_W-1:0] top_row_idx;

    // Control register write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_reg    <= CMD_NOP;
            reward_reg <= 8'sd0;
        end else if (ctrl_write) begin
            case (ctrl_addr)
                8'h00: cmd_reg    <= ctrl_wdata;
                8'h04: reward_reg <= ctrl_wdata[7:0];
            endcase
        end else if (state == S_IDLE) begin
            cmd_reg <= CMD_NOP;  // Auto-clear command
        end
    end

    // Control register read
    always_comb begin
        case (ctrl_addr)
            8'h00: ctrl_rdata = cmd_reg;
            8'h04: ctrl_rdata = {{24{reward_reg[7]}}, reward_reg};
            8'h08: ctrl_rdata = {29'b0, o_plasticity_done, o_inference_done, o_busy};
            8'h0C: ctrl_rdata = query_count;
            8'h10: ctrl_rdata = learn_count;
            8'h14: ctrl_rdata = top_score;
            8'h18: ctrl_rdata = {{(32-ROW_ADDR_W){1'b0}}, top_row_idx};
            default: ctrl_rdata = 32'hDEADBEEF;
        endcase
    end

    // =========================================================================
    // 5. QUERY BUFFER
    // =========================================================================
    // Store incoming query HV chunk by chunk
    reg [CHUNK-1:0] query_buffer [CHUNKS_PER_ROW-1:0];
    reg [CHUNK_ADDR_W-1:0] query_chunk_idx;
    reg query_complete;

    // =========================================================================
    // 6. INFERENCE PIPELINE
    // =========================================================================
    reg [CHUNK_ADDR_W-1:0] infer_chunk_idx;
    reg [ROW_ADDR_W-1:0] infer_row_batch;  // Which batch of PAR_LAYERS rows
    reg [BANK_ADDR_W-1:0] infer_bank_addr;

    // Per-lane working registers
    reg [CHUNK-1:0] weight_read [PAR_LAYERS-1:0];
    reg [CHUNK-1:0] xor_result [PAR_LAYERS-1:0];
    reg [15:0] popcount_result [PAR_LAYERS-1:0];
    reg [31:0] row_scores [PAR_LAYERS-1:0];

    // Score memory for all rows
    reg [31:0] all_scores [ROWS-1:0];

    // Popcount function (synthesis will infer LUT tree or DSP)
    function automatic [15:0] popcount512(input [CHUNK-1:0] vec);
        integer i;
        reg [15:0] cnt;
        begin
            cnt = 16'd0;
            for (i = 0; i < CHUNK; i = i + 1)
                cnt = cnt + vec[i];
            popcount512 = cnt;
        end
    endfunction

    // =========================================================================
    // 7. PLASTICITY ENGINE
    // =========================================================================
    reg [DIM-1:0] context_hv_latched;
    reg [ROW_ADDR_W-1:0] learn_row_idx;
    reg [CHUNK_ADDR_W-1:0] learn_chunk_idx;
    reg [BANK_ADDR_W-1:0] learn_bank_addr;
    reg [$clog2(PAR_LAYERS)-1:0] learn_bank_idx;
    reg signed [1:0] reward_sign;

    // Weight and accumulator working registers for learning
    reg [CHUNK-1:0] learn_weight_rd;
    reg [CHUNK*ACC_WIDTH-1:0] learn_accum_rd;
    reg [CHUNK-1:0] learn_weight_wr;
    reg [CHUNK*ACC_WIDTH-1:0] learn_accum_wr;

    // =========================================================================
    // 8. MAIN FSM
    // =========================================================================
    reg busy_reg;
    reg infer_done_reg;
    reg plast_done_reg;

    assign o_busy = busy_reg;
    assign o_inference_done = infer_done_reg;
    assign o_plasticity_done = plast_done_reg;
    assign s_axis_tready = (state == S_LOAD_QUERY);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            busy_reg        <= 1'b0;
            infer_done_reg  <= 1'b0;
            plast_done_reg  <= 1'b0;
            query_chunk_idx <= '0;
            query_complete  <= 1'b0;
            query_count     <= 32'd0;
            learn_count     <= 32'd0;
            top_score       <= 32'd0;
            top_row_idx     <= '0;
            m_axis_tvalid   <= 1'b0;
            m_axis_tlast    <= 1'b0;

            for (int i = 0; i < CHUNKS_PER_ROW; i++)
                query_buffer[i] <= '0;
            for (int i = 0; i < PAR_LAYERS; i++)
                row_scores[i] <= 32'd0;
            for (int i = 0; i < ROWS; i++)
                all_scores[i] <= 32'd0;

        end else begin
            // Default: clear one-shot signals
            infer_done_reg <= 1'b0;
            plast_done_reg <= 1'b0;
            m_axis_tvalid  <= 1'b0;
            m_axis_tlast   <= 1'b0;

            case (state)
                // ---------------------------------------------------------
                S_IDLE: begin
                    busy_reg <= 1'b0;

                    if (cmd_reg == CMD_QUERY && s_axis_tvalid) begin
                        state           <= S_LOAD_QUERY;
                        busy_reg        <= 1'b1;
                        query_chunk_idx <= '0;
                        query_complete  <= 1'b0;
                    end
                    else if (cmd_reg == CMD_LEARN) begin
                        state    <= S_LEARN_LATCH;
                        busy_reg <= 1'b1;
                    end
                end

                // ---------------------------------------------------------
                // LOAD QUERY: Stream in HV chunks
                // ---------------------------------------------------------
                S_LOAD_QUERY: begin
                    if (s_axis_tvalid) begin
                        query_buffer[query_chunk_idx] <= s_axis_tdata;

                        if (s_axis_tlast || query_chunk_idx == CHUNKS_PER_ROW - 1) begin
                            query_complete <= 1'b1;
                            state <= S_INFER_READ;
                            infer_chunk_idx <= '0;
                            infer_row_batch <= '0;
                            for (int i = 0; i < PAR_LAYERS; i++)
                                row_scores[i] <= 32'd0;
                        end else begin
                            query_chunk_idx <= query_chunk_idx + 1'b1;
                        end
                    end
                end

                // ---------------------------------------------------------
                // INFERENCE: Read weights from all parallel banks
                // ---------------------------------------------------------
                S_INFER_READ: begin
                    // Calculate bank address
                    infer_bank_addr <= (infer_row_batch * CHUNKS_PER_ROW) + infer_chunk_idx;

                    // Read from all banks simultaneously
                    for (int i = 0; i < PAR_LAYERS; i++) begin
                        weight_read[i] <= mem_weights[i][infer_bank_addr];
                    end
                    state <= S_INFER_XOR;
                end

                // ---------------------------------------------------------
                // INFERENCE: XOR with query chunk
                // ---------------------------------------------------------
                S_INFER_XOR: begin
                    for (int i = 0; i < PAR_LAYERS; i++) begin
                        // Match bits: where query == weight (both 0 or both 1)
                        xor_result[i] <= ~(weight_read[i] ^ query_buffer[infer_chunk_idx]);
                    end
                    state <= S_INFER_POP;
                end

                // ---------------------------------------------------------
                // INFERENCE: Popcount
                // ---------------------------------------------------------
                S_INFER_POP: begin
                    for (int i = 0; i < PAR_LAYERS; i++) begin
                        popcount_result[i] <= popcount512(xor_result[i]);
                    end
                    state <= S_INFER_ACCUM;
                end

                // ---------------------------------------------------------
                // INFERENCE: Accumulate scores
                // ---------------------------------------------------------
                S_INFER_ACCUM: begin
                    for (int i = 0; i < PAR_LAYERS; i++) begin
                        row_scores[i] <= row_scores[i] + popcount_result[i];
                    end
                    state <= S_INFER_NEXT;
                end

                // ---------------------------------------------------------
                // INFERENCE: Next chunk or next batch
                // ---------------------------------------------------------
                S_INFER_NEXT: begin
                    if (infer_chunk_idx == CHUNKS_PER_ROW - 1) begin
                        // Save scores for this batch
                        for (int i = 0; i < PAR_LAYERS; i++) begin
                            all_scores[infer_row_batch * PAR_LAYERS + i] <= row_scores[i];
                        end

                        if (infer_row_batch == ROWS_PER_BANK - 1) begin
                            // All rows done
                            state <= S_OUTPUT;
                            query_count <= query_count + 1'b1;
                        end else begin
                            // Next batch of rows
                            infer_row_batch <= infer_row_batch + 1'b1;
                            infer_chunk_idx <= '0;
                            for (int i = 0; i < PAR_LAYERS; i++)
                                row_scores[i] <= 32'd0;
                            state <= S_INFER_READ;
                        end
                    end else begin
                        // Next chunk in current batch
                        infer_chunk_idx <= infer_chunk_idx + 1'b1;
                        state <= S_INFER_READ;
                    end
                end

                // ---------------------------------------------------------
                // OUTPUT: Find best score and output
                // ---------------------------------------------------------
                S_OUTPUT: begin
                    // Find top score (simplified linear scan)
                    // In production, use parallel reduction tree
                    reg [31:0] best_score;
                    reg [ROW_ADDR_W-1:0] best_idx;
                    best_score = 32'd0;
                    best_idx = '0;
                    for (int i = 0; i < ROWS; i++) begin
                        if (all_scores[i] > best_score) begin
                            best_score = all_scores[i];
                            best_idx = i[ROW_ADDR_W-1:0];
                        end
                    end
                    top_score <= best_score;
                    top_row_idx <= best_idx;

                    // Output best score
                    m_axis_tdata <= best_score;
                    m_axis_tvalid <= 1'b1;
                    m_axis_tlast <= 1'b1;

                    infer_done_reg <= 1'b1;
                    state <= S_DONE;
                end

                // ---------------------------------------------------------
                // LEARN: Latch context and reward
                // ---------------------------------------------------------
                S_LEARN_LATCH: begin
                    // Reconstruct full context HV from query buffer
                    for (int c = 0; c < CHUNKS_PER_ROW; c++) begin
                        for (int b = 0; b < CHUNK; b++) begin
                            context_hv_latched[c*CHUNK + b] <= query_buffer[c][b];
                        end
                    end
                    reward_sign <= (reward_reg > 8'sd0) ? 2'sb01 : -2'sb01;
                    learn_row_idx   <= '0;
                    learn_chunk_idx <= '0;
                    state <= S_LEARN_READ;
                end

                // ---------------------------------------------------------
                // LEARN: Read weights and accumulators
                // ---------------------------------------------------------
                S_LEARN_READ: begin
                    learn_bank_idx <= learn_row_idx % PAR_LAYERS;
                    learn_bank_addr <= (learn_row_idx / PAR_LAYERS) * CHUNKS_PER_ROW + learn_chunk_idx;

                    learn_weight_rd <= mem_weights[learn_bank_idx][learn_bank_addr];
                    learn_accum_rd  <= mem_accums[learn_bank_idx][learn_bank_addr];
                    state <= S_LEARN_UPDATE;
                end

                // ---------------------------------------------------------
                // LEARN: Update accumulators and weights
                // ---------------------------------------------------------
                S_LEARN_UPDATE: begin
                    // Parallel update all bits in chunk
                    for (int i = 0; i < CHUNK; i++) begin
                        logic in_bit;
                        logic signed [ACC_WIDTH-1:0] acc_old, acc_new;
                        logic signed [ACC_WIDTH-1:0] step;

                        in_bit = context_hv_latched[learn_chunk_idx * CHUNK + i];
                        acc_old = learn_accum_rd[i*ACC_WIDTH +: ACC_WIDTH];

                        // Target-directed: step = input * sign(reward)
                        step = in_bit ? {{(ACC_WIDTH-2){reward_sign[1]}}, reward_sign} :
                                       -{{(ACC_WIDTH-2){reward_sign[1]}}, reward_sign};

                        // Saturating add
                        acc_new = sat_add_fn(acc_old, step);

                        learn_accum_wr[i*ACC_WIDTH +: ACC_WIDTH] <= acc_new;

                        // Update weight based on accumulator sign
                        if (acc_new > 0)
                            learn_weight_wr[i] <= 1'b1;
                        else if (acc_new < 0)
                            learn_weight_wr[i] <= 1'b0;
                        else
                            learn_weight_wr[i] <= learn_weight_rd[i];  // Preserve
                    end
                    state <= S_LEARN_WRITE;
                end

                // ---------------------------------------------------------
                // LEARN: Write back to memory
                // ---------------------------------------------------------
                S_LEARN_WRITE: begin
                    mem_weights[learn_bank_idx][learn_bank_addr] <= learn_weight_wr;
                    mem_accums[learn_bank_idx][learn_bank_addr]  <= learn_accum_wr;
                    state <= S_LEARN_NEXT;
                end

                // ---------------------------------------------------------
                // LEARN: Next chunk or row
                // ---------------------------------------------------------
                S_LEARN_NEXT: begin
                    if (learn_chunk_idx == CHUNKS_PER_ROW - 1) begin
                        if (learn_row_idx == ROWS - 1) begin
                            // All rows updated
                            plast_done_reg <= 1'b1;
                            learn_count <= learn_count + 1'b1;
                            state <= S_DONE;
                        end else begin
                            // Next row
                            learn_row_idx   <= learn_row_idx + 1'b1;
                            learn_chunk_idx <= '0;
                            state <= S_LEARN_READ;
                        end
                    end else begin
                        // Next chunk
                        learn_chunk_idx <= learn_chunk_idx + 1'b1;
                        state <= S_LEARN_READ;
                    end
                end

                // ---------------------------------------------------------
                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // =========================================================================
    // 9. HELPER FUNCTIONS
    // =========================================================================
    function automatic logic signed [ACC_WIDTH-1:0] sat_add_fn(
        input logic signed [ACC_WIDTH-1:0] val,
        input logic signed [ACC_WIDTH-1:0] inc
    );
        localparam int ACC_MAX = (1 << (ACC_WIDTH-1)) - 1;  // +63
        localparam int ACC_MIN = -(1 << (ACC_WIDTH-1));     // -64
        logic signed [ACC_WIDTH:0] sum;
        begin
            sum = $signed({val[ACC_WIDTH-1], val}) + $signed({inc[ACC_WIDTH-1], inc});
            if (sum > ACC_MAX)
                sat_add_fn = ACC_MAX[ACC_WIDTH-1:0];
            else if (sum < ACC_MIN)
                sat_add_fn = ACC_MIN[ACC_WIDTH-1:0];
            else
                sat_add_fn = sum[ACC_WIDTH-1:0];
        end
    endfunction

    // =========================================================================
    // 10. TIMING ASSERTIONS (Simulation only)
    // =========================================================================
    `ifdef SIMULATION
    initial begin
        $display("=======================================================");
        $display("AXIS Soul Core Instantiated");
        $display("  DIM        = %0d", DIM);
        $display("  CHUNK      = %0d", CHUNK);
        $display("  ROWS       = %0d", ROWS);
        $display("  PAR_LAYERS = %0d", PAR_LAYERS);
        $display("  ACC_WIDTH  = %0d", ACC_WIDTH);
        $display("-------------------------------------------------------");
        $display("  CHUNKS_PER_ROW = %0d", CHUNKS_PER_ROW);
        $display("  ROWS_PER_BANK  = %0d", ROWS_PER_BANK);
        $display("  BANK_DEPTH     = %0d", BANK_DEPTH);
        $display("  Query Latency  ~ %0d cycles", CHUNKS_PER_ROW * (ROWS/PAR_LAYERS) * 4);
        $display("  Learn Latency  ~ %0d cycles", ROWS * CHUNKS_PER_ROW * 4);
        $display("=======================================================");
    end
    `endif

endmodule

`default_nettype wire
