/*
 * HTC Weight Loader - Live BRAM Update via AXI-Stream
 * ====================================================
 *
 * Receives weight updates from CPU and writes to HTC BRAM.
 * Supports incremental updates (single attractor) or full reload.
 *
 * Weight Format:
 *   - D=173 binary HVs packed into 192 bits (padded)
 *   - R=2048 attractors total
 *   - Total: 2048 × 24 bytes = 48 KB
 *
 * Update Latency:
 *   - Full reload: ~5 µs @ 10 Gbps PCIe
 *   - Single attractor: ~50 ns
 */

module htc_weight_loader #(
    parameter int D = 173,              // HV dimension
    parameter int D_PADDED = 192,       // Padded to 64-bit boundary
    parameter int R = 2048,             // Number of attractors
    parameter int AXI_WIDTH = 512,      // AXI-Stream width
    parameter int CHUNKS_PER_HV = 1     // 192 bits fits in one 512-bit transfer
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // AXI-Stream slave interface (from PCIe DMA)
    input  logic [AXI_WIDTH-1:0]    s_axis_tdata,
    input  logic                    s_axis_tvalid,
    input  logic                    s_axis_tlast,
    input  logic [63:0]             s_axis_tuser,    // [15:0]=attractor_id, [16]=full_reload
    output logic                    s_axis_tready,

    // BRAM write port
    output logic [$clog2(R)-1:0]    bram_addr,
    output logic [D_PADDED-1:0]     bram_wdata,
    output logic                    bram_we,

    // Status
    output logic                    busy,
    output logic [31:0]             weights_loaded,
    output logic                    reload_complete
);

    // =============================================================================
    // State Machine
    // =============================================================================

    typedef enum logic [2:0] {
        IDLE,
        RECEIVE_HV,
        WRITE_BRAM,
        NEXT_ATTRACTOR,
        RELOAD_DONE
    } state_t;

    state_t state, next_state;

    // Registers
    logic [$clog2(R)-1:0] current_attractor;
    logic [$clog2(R)-1:0] target_attractor;
    logic [D_PADDED-1:0]  hv_buffer;
    logic                 full_reload_mode;
    logic [31:0]          load_count;

    // =============================================================================
    // State Transitions
    // =============================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end

    always_comb begin
        next_state = state;

        case (state)
            IDLE: begin
                if (s_axis_tvalid) begin
                    next_state = RECEIVE_HV;
                end
            end

            RECEIVE_HV: begin
                if (s_axis_tvalid && s_axis_tready) begin
                    next_state = WRITE_BRAM;
                end
            end

            WRITE_BRAM: begin
                next_state = NEXT_ATTRACTOR;
            end

            NEXT_ATTRACTOR: begin
                if (full_reload_mode && current_attractor < R - 1) begin
                    next_state = RECEIVE_HV;
                end else if (full_reload_mode) begin
                    next_state = RELOAD_DONE;
                end else begin
                    next_state = IDLE;
                end
            end

            RELOAD_DONE: begin
                next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // =============================================================================
    // Datapath
    // =============================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_attractor <= '0;
            target_attractor <= '0;
            hv_buffer <= '0;
            full_reload_mode <= 1'b0;
            load_count <= '0;
            bram_addr <= '0;
            bram_wdata <= '0;
            bram_we <= 1'b0;
            reload_complete <= 1'b0;
        end else begin
            // Default
            bram_we <= 1'b0;
            reload_complete <= 1'b0;

            case (state)
                IDLE: begin
                    if (s_axis_tvalid) begin
                        // Parse header from tuser
                        target_attractor <= s_axis_tuser[15:0];
                        full_reload_mode <= s_axis_tuser[16];

                        if (s_axis_tuser[16]) begin
                            // Full reload: start from attractor 0
                            current_attractor <= '0;
                        end else begin
                            // Single update
                            current_attractor <= s_axis_tuser[15:0];
                        end
                    end
                end

                RECEIVE_HV: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        // Capture HV data (D=173 bits)
                        hv_buffer <= s_axis_tdata[D_PADDED-1:0];
                    end
                end

                WRITE_BRAM: begin
                    // Write to BRAM
                    bram_addr <= current_attractor;
                    bram_wdata <= hv_buffer;
                    bram_we <= 1'b1;
                    load_count <= load_count + 1;
                end

                NEXT_ATTRACTOR: begin
                    if (full_reload_mode && current_attractor < R - 1) begin
                        current_attractor <= current_attractor + 1;
                    end
                end

                RELOAD_DONE: begin
                    reload_complete <= 1'b1;
                end
            endcase
        end
    end

    // =============================================================================
    // Outputs
    // =============================================================================

    assign s_axis_tready = (state == RECEIVE_HV);
    assign busy = (state != IDLE);
    assign weights_loaded = load_count;

endmodule


// =============================================================================
// Weight Checksum Module - Verify integrity after reload
// =============================================================================

module htc_weight_checksum #(
    parameter int D = 173,
    parameter int R = 2048
)(
    input  logic               clk,
    input  logic               rst_n,

    input  logic               start,
    input  logic [D-1:0]       bram_rdata,
    output logic [$clog2(R)-1:0] bram_raddr,
    output logic               done,
    output logic [63:0]        checksum
);

    logic [$clog2(R):0] row_idx;
    logic [63:0] running_sum;
    logic computing;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_idx <= '0;
            running_sum <= '0;
            computing <= 1'b0;
            done <= 1'b0;
            checksum <= '0;
        end else begin
            done <= 1'b0;

            if (start && !computing) begin
                computing <= 1'b1;
                row_idx <= '0;
                running_sum <= '0;
            end else if (computing) begin
                if (row_idx < R) begin
                    // XOR-fold the HV into checksum
                    running_sum <= running_sum ^ {bram_rdata, bram_rdata[D-1:D-64+D]};
                    row_idx <= row_idx + 1;
                end else begin
                    computing <= 1'b0;
                    done <= 1'b1;
                    checksum <= running_sum;
                end
            end
        end
    end

    assign bram_raddr = row_idx[$clog2(R)-1:0];

endmodule
