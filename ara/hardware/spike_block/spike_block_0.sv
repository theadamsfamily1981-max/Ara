/*
 * SpikingBrain-Style FPGA Tile
 * ============================
 *
 * A minimal neuromorphic processing tile implementing:
 * - Integrate-and-fire neurons with dynamic thresholds
 * - Sparse weight storage and multiply-accumulate
 * - Hebbian learning capability
 *
 * Target: Stratix-10 GX / VU9P class FPGAs
 *
 * Parameters:
 *   N_NEURONS   - Number of spiking neurons (default 512)
 *   D_EMBED     - Input embedding dimension (default 128)
 *   W_BITS      - Weight precision in bits (default 4)
 *   STATE_BITS  - Neuron state precision (default 8)
 */

module spike_block_0 #(
    parameter N_NEURONS   = 512,
    parameter D_EMBED     = 128,
    parameter W_BITS      = 4,
    parameter STATE_BITS  = 8,
    parameter ALPHA_FP    = 115,  // 0.9 in Q0.7
    parameter BETA_FP     = 1,    // 0.01 in Q0.7
    parameter TAU_FP      = 6     // 0.05 in Q0.7
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // AXI-Stream Input
    input  wire [D_EMBED*STATE_BITS-1:0] s_axis_tdata,
    input  wire                          s_axis_tvalid,
    output reg                           s_axis_tready,
    input  wire [STATE_BITS-1:0]        s_axis_trho,  // Neuromodulator

    // AXI-Stream Output
    output reg  [D_EMBED*STATE_BITS-1:0] m_axis_tdata,
    output reg                           m_axis_tvalid,
    input  wire                          m_axis_tready,
    output wire [N_NEURONS-1:0]         m_axis_tspikes,

    // Control
    input  wire                          enable,
    input  wire                          learn_enable,

    // Status
    output reg  [31:0]                   spike_count,
    output reg  [31:0]                   cycle_count,

    // Weight memory interface (external BRAM)
    output reg  [$clog2(N_NEURONS)-1:0]  weight_row_addr,
    input  wire [15:0]                   weight_row_ptr_cur,
    input  wire [15:0]                   weight_row_ptr_next,
    output reg  [15:0]                   weight_col_addr,
    input  wire [15:0]                   weight_col_idx,
    input  wire [W_BITS-1:0]            weight_value,
    output reg                           weight_rd_en,
    output reg                           weight_wr_en,
    output reg  [W_BITS-1:0]            weight_wr_data
);

    // ========================================================================
    // Local Parameters
    // ========================================================================

    localparam ACCUM_BITS = 24;  // Accumulator for MAV
    localparam FRAC_BITS  = 7;   // Fractional bits for fixed-point

    // FSM States
    localparam [3:0]
        ST_IDLE       = 4'd0,
        ST_LOAD_INPUT = 4'd1,
        ST_INIT_ROW   = 4'd2,
        ST_FETCH_PTR  = 4'd3,
        ST_MAV_LOOP   = 4'd4,
        ST_THRESHOLD  = 4'd5,
        ST_SPIKE      = 4'd6,
        ST_LEARN      = 4'd7,
        ST_OUTPUT     = 4'd8;

    // ========================================================================
    // Registers
    // ========================================================================

    // FSM
    reg [3:0] state, next_state;

    // Input buffer
    reg [STATE_BITS-1:0] input_embed [0:D_EMBED-1];

    // Neuron state
    reg signed [STATE_BITS-1:0] membrane [0:N_NEURONS-1];    // v[t]
    reg [STATE_BITS-1:0]        threshold [0:N_NEURONS-1];   // θ[t]
    reg [N_NEURONS-1:0]         spikes;

    // Processing
    reg [$clog2(N_NEURONS)-1:0] neuron_idx;
    reg [15:0]                   nnz_idx;
    reg [15:0]                   row_start, row_end;
    reg signed [ACCUM_BITS-1:0] accumulator;

    // Output buffer
    reg signed [STATE_BITS-1:0] output_embed [0:D_EMBED-1];

    // Learning
    reg signed [STATE_BITS-1:0] rho_reg;

    // ========================================================================
    // Input Unpacking
    // ========================================================================

    integer i;
    always @(posedge clk) begin
        if (state == ST_LOAD_INPUT && s_axis_tvalid) begin
            for (i = 0; i < D_EMBED; i = i + 1) begin
                input_embed[i] <= s_axis_tdata[i*STATE_BITS +: STATE_BITS];
            end
            rho_reg <= s_axis_trho;
        end
    end

    // ========================================================================
    // FSM
    // ========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= ST_IDLE;
        else
            state <= next_state;
    end

    always @(*) begin
        next_state = state;
        case (state)
            ST_IDLE: begin
                if (enable && s_axis_tvalid)
                    next_state = ST_LOAD_INPUT;
            end

            ST_LOAD_INPUT: begin
                next_state = ST_INIT_ROW;
            end

            ST_INIT_ROW: begin
                next_state = ST_FETCH_PTR;
            end

            ST_FETCH_PTR: begin
                next_state = ST_MAV_LOOP;
            end

            ST_MAV_LOOP: begin
                if (nnz_idx >= row_end)
                    next_state = ST_THRESHOLD;
            end

            ST_THRESHOLD: begin
                next_state = ST_SPIKE;
            end

            ST_SPIKE: begin
                if (learn_enable && rho_reg != 0 && spikes[neuron_idx])
                    next_state = ST_LEARN;
                else if (neuron_idx == N_NEURONS - 1)
                    next_state = ST_OUTPUT;
                else
                    next_state = ST_INIT_ROW;
            end

            ST_LEARN: begin
                // Hebbian update for active synapses
                if (nnz_idx >= row_end) begin
                    if (neuron_idx == N_NEURONS - 1)
                        next_state = ST_OUTPUT;
                    else
                        next_state = ST_INIT_ROW;
                end
            end

            ST_OUTPUT: begin
                if (m_axis_tready)
                    next_state = ST_IDLE;
            end
        endcase
    end

    // ========================================================================
    // Datapath
    // ========================================================================

    // Weight memory control
    always @(posedge clk) begin
        weight_rd_en <= 0;
        weight_wr_en <= 0;

        case (state)
            ST_INIT_ROW: begin
                weight_row_addr <= neuron_idx;
            end

            ST_FETCH_PTR: begin
                row_start <= weight_row_ptr_cur;
                row_end   <= weight_row_ptr_next;
                nnz_idx   <= weight_row_ptr_cur;
            end

            ST_MAV_LOOP: begin
                if (nnz_idx < row_end) begin
                    weight_col_addr <= nnz_idx;
                    weight_rd_en <= 1;
                    nnz_idx <= nnz_idx + 1;
                end
            end

            ST_LEARN: begin
                if (nnz_idx < row_end) begin
                    weight_col_addr <= nnz_idx;
                    weight_rd_en <= 1;
                    weight_wr_en <= 1;
                    nnz_idx <= nnz_idx + 1;
                end
            end
        endcase
    end

    // Accumulator for sparse MAV
    wire signed [ACCUM_BITS-1:0] mav_product;
    wire [15:0] col_idx_reg;
    reg  [15:0] col_idx_d1;

    always @(posedge clk) begin
        col_idx_d1 <= weight_col_idx;
    end

    // Product: weight * input[col_idx]
    wire signed [W_BITS-1:0]     w_signed = weight_value;
    wire signed [STATE_BITS-1:0] x_signed = input_embed[col_idx_d1];
    assign mav_product = w_signed * x_signed;

    always @(posedge clk) begin
        case (state)
            ST_INIT_ROW: begin
                accumulator <= 0;
                neuron_idx <= (state == ST_IDLE) ? 0 : neuron_idx;
            end

            ST_MAV_LOOP: begin
                // Pipeline: accumulate one cycle after read
                if (weight_rd_en)
                    accumulator <= accumulator + mav_product;
            end
        endcase
    end

    // Neuron update
    wire signed [ACCUM_BITS-1:0] leaked_potential;
    wire signed [ACCUM_BITS-1:0] new_potential;

    assign leaked_potential = (membrane[neuron_idx] * ALPHA_FP) >>> FRAC_BITS;
    assign new_potential = leaked_potential + accumulator - $signed({1'b0, threshold[neuron_idx]});

    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < N_NEURONS; i = i + 1) begin
                membrane[i] <= 0;
                threshold[i] <= 64;  // Initial threshold
            end
            spikes <= 0;
        end
        else if (state == ST_THRESHOLD) begin
            // Check for spike
            if (new_potential > 0) begin
                spikes[neuron_idx] <= 1;
                membrane[neuron_idx] <= 0;  // Reset
            end
            else begin
                spikes[neuron_idx] <= 0;
                membrane[neuron_idx] <= new_potential[STATE_BITS-1:0];
            end
        end
        else if (state == ST_SPIKE) begin
            // Threshold adaptation: θ += β * (spike - τ)
            if (spikes[neuron_idx])
                threshold[neuron_idx] <= threshold[neuron_idx] + BETA_FP;
            else if (threshold[neuron_idx] > TAU_FP)
                threshold[neuron_idx] <= threshold[neuron_idx] - 1;

            // Advance to next neuron
            if (neuron_idx < N_NEURONS - 1)
                neuron_idx <= neuron_idx + 1;
        end
    end

    // Hebbian learning
    wire signed [W_BITS+STATE_BITS+STATE_BITS-1:0] delta_w;
    assign delta_w = (rho_reg * x_signed) >>> (FRAC_BITS + 4);  // Scale down

    always @(posedge clk) begin
        if (state == ST_LEARN && weight_wr_en) begin
            // Δw = η * ρ * pre[j] * post[i]
            // Simplified: just use rho * input since post[i] = 1 (we're in ST_LEARN because spike)
            weight_wr_data <= weight_value + delta_w[W_BITS-1:0];
        end
    end

    // ========================================================================
    // Output
    // ========================================================================

    assign m_axis_tspikes = spikes;

    // Simple output: project spikes back to embedding space
    // (In full implementation, would use another sparse matrix)
    always @(posedge clk) begin
        if (state == ST_OUTPUT) begin
            for (i = 0; i < D_EMBED; i = i + 1) begin
                // Pass through input with spike modulation
                output_embed[i] <= input_embed[i] + (spikes[i % N_NEURONS] ? 8 : 0);
            end
        end
    end

    // Pack output
    genvar g;
    generate
        for (g = 0; g < D_EMBED; g = g + 1) begin : pack_output
            always @(posedge clk) begin
                m_axis_tdata[g*STATE_BITS +: STATE_BITS] <= output_embed[g];
            end
        end
    endgenerate

    // Handshaking
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axis_tready <= 1;
            m_axis_tvalid <= 0;
        end
        else begin
            s_axis_tready <= (state == ST_IDLE);
            m_axis_tvalid <= (state == ST_OUTPUT);
        end
    end

    // ========================================================================
    // Statistics
    // ========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_count <= 0;
            cycle_count <= 0;
        end
        else if (state == ST_OUTPUT) begin
            spike_count <= spike_count + $countones(spikes);
            cycle_count <= cycle_count + 1;
        end
    end

endmodule


/*
 * Testbench for spike_block_0
 */
`ifdef SIMULATION

module spike_block_tb;

    parameter N_NEURONS = 64;
    parameter D_EMBED = 32;

    reg clk, rst_n, enable, learn_enable;
    reg [D_EMBED*8-1:0] s_axis_tdata;
    reg s_axis_tvalid;
    wire s_axis_tready;
    reg [7:0] s_axis_trho;

    wire [D_EMBED*8-1:0] m_axis_tdata;
    wire m_axis_tvalid;
    reg m_axis_tready;
    wire [N_NEURONS-1:0] m_axis_tspikes;

    wire [31:0] spike_count, cycle_count;

    // Weight memory (simplified - would be external BRAM)
    reg [15:0] weight_row_ptr [0:N_NEURONS];
    reg [15:0] weight_col_idx [0:1023];
    reg [3:0]  weight_values [0:1023];

    wire [$clog2(N_NEURONS)-1:0] weight_row_addr;
    wire [15:0] weight_col_addr;
    wire weight_rd_en, weight_wr_en;
    wire [3:0] weight_wr_data;

    spike_block_0 #(
        .N_NEURONS(N_NEURONS),
        .D_EMBED(D_EMBED)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_trho(s_axis_trho),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tspikes(m_axis_tspikes),
        .enable(enable),
        .learn_enable(learn_enable),
        .spike_count(spike_count),
        .cycle_count(cycle_count),
        .weight_row_addr(weight_row_addr),
        .weight_row_ptr_cur(weight_row_ptr[weight_row_addr]),
        .weight_row_ptr_next(weight_row_ptr[weight_row_addr + 1]),
        .weight_col_addr(weight_col_addr),
        .weight_col_idx(weight_col_idx[weight_col_addr]),
        .weight_value(weight_values[weight_col_addr]),
        .weight_rd_en(weight_rd_en),
        .weight_wr_en(weight_wr_en),
        .weight_wr_data(weight_wr_data)
    );

    // Clock
    always #5 clk = ~clk;

    // Initialize weights (sparse random)
    integer i, j, nnz;
    initial begin
        nnz = 0;
        for (i = 0; i <= N_NEURONS; i = i + 1) begin
            weight_row_ptr[i] = nnz;
            if (i < N_NEURONS) begin
                for (j = 0; j < D_EMBED; j = j + 1) begin
                    if ($random % 10 == 0) begin  // 10% density
                        weight_col_idx[nnz] = j;
                        weight_values[nnz] = $random % 16 - 8;
                        nnz = nnz + 1;
                    end
                end
            end
        end
        $display("Initialized %0d non-zero weights", nnz);
    end

    // Test sequence
    initial begin
        clk = 0;
        rst_n = 0;
        enable = 0;
        learn_enable = 0;
        s_axis_tvalid = 0;
        s_axis_trho = 0;
        m_axis_tready = 1;

        #20 rst_n = 1;
        #10 enable = 1;

        // Send 10 embeddings
        repeat (10) begin
            // Random input
            for (i = 0; i < D_EMBED; i = i + 1)
                s_axis_tdata[i*8 +: 8] = $random % 256 - 128;

            s_axis_tvalid = 1;
            s_axis_trho = 8'd10;  // Small learning signal
            @(posedge clk);
            while (!s_axis_tready) @(posedge clk);
            s_axis_tvalid = 0;

            // Wait for output
            while (!m_axis_tvalid) @(posedge clk);
            $display("Cycle %0d: %0d spikes", cycle_count, $countones(m_axis_tspikes));
            @(posedge clk);
        end

        $display("Final spike count: %0d", spike_count);
        $finish;
    end

endmodule

`endif
