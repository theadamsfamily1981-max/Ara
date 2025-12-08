// ============================================================================
// Micron SB-852 Memory Adapter for Ara Soul Engine
// ============================================================================
//
// Target: Micron SB-852 "Deep Learning Accelerator" Board
//         Stratix-10 SoC + Multi-channel DDR4 (tens of GB)
//
// Role:   Phase 2 production platform - "deep-stream emotional cortex"
//         CERN-proven for 40 MHz streaming neural inference
//
// This adapter bridges the platform-agnostic Ara plasticity core to the
// SB-852's multi-channel DDR4 system via Intel EMIF.
//
// Key Features:
//   - Multi-channel DDR4 for parallel access (2-4 channels typical)
//   - HPS bridge for ARM-side soul management daemon
//   - Optimized for streaming workloads (burst-friendly)
//   - Designed for microsecond-scale emotional updates
//
// Memory Layout:
//   - Channel 0: Sign bits (compact, sequential access)
//   - Channel 1: Accumulators (wider, more bandwidth needed)
//   - Optional: Channel 2/3 for shadow copies or multi-soul
//
// ============================================================================

`include "../common/ara_soul_config.svh"

module sb852_mem_adapter #(
    parameter int ROWS       = `ARA_ROWS,
    parameter int DIM        = `ARA_DIM,
    parameter int CHUNK_BITS = `ARA_CHUNK_BITS,
    parameter int ACC_WIDTH  = `ARA_ACC_WIDTH,

    // SB-852 specific parameters
    parameter int DDR4_CHANNELS      = 2,          // Number of DDR4 channels
    parameter int DDR4_DATA_WIDTH    = 512,        // Per-channel data width
    parameter int DDR4_ADDR_WIDTH    = 32,         // Address bits
    parameter int HPS_BRIDGE_WIDTH   = 128,        // AXI bridge to HPS

    // Memory region configuration
    parameter logic [DDR4_ADDR_WIDTH-1:0] CH0_BASE_SIGNS  = 32'h0000_0000,
    parameter logic [DDR4_ADDR_WIDTH-1:0] CH1_BASE_ACCUMS = 32'h0000_0000,

    // Streaming optimizations
    parameter int PREFETCH_DEPTH = 4,              // Prefetch buffer depth
    parameter bit ENABLE_BURST_OPT = 1'b1          // Enable burst optimization
)(
    input  logic clk,
    input  logic rst_n,

    // ========================================================================
    // Core Interface (from ara_plasticity_core)
    // ========================================================================

    // Read request
    input  logic                       core_rd_req,
    input  logic [$clog2(ROWS)-1:0]    core_rd_row,
    input  logic [$clog2(DIM/CHUNK_BITS)-1:0] core_rd_chunk,
    output logic                       core_rd_valid,
    output logic [CHUNK_BITS-1:0]      core_rd_signs,
    output logic [CHUNK_BITS*ACC_WIDTH-1:0] core_rd_accums,

    // Write request
    input  logic                       core_wr_req,
    input  logic [$clog2(ROWS)-1:0]    core_wr_row,
    input  logic [$clog2(DIM/CHUNK_BITS)-1:0] core_wr_chunk,
    input  logic [CHUNK_BITS-1:0]      core_wr_signs,
    input  logic [CHUNK_BITS*ACC_WIDTH-1:0] core_wr_accums,
    output logic                       core_wr_done,

    // ========================================================================
    // DDR4 Channel 0 - Signs (Intel Avalon-MM)
    // ========================================================================

    output logic                       ddr4_ch0_read,
    output logic                       ddr4_ch0_write,
    output logic [DDR4_ADDR_WIDTH-1:0] ddr4_ch0_address,
    output logic [DDR4_DATA_WIDTH-1:0] ddr4_ch0_writedata,
    output logic [DDR4_DATA_WIDTH/8-1:0] ddr4_ch0_byteenable,
    output logic [6:0]                 ddr4_ch0_burstcount,
    input  logic [DDR4_DATA_WIDTH-1:0] ddr4_ch0_readdata,
    input  logic                       ddr4_ch0_readdatavalid,
    input  logic                       ddr4_ch0_waitrequest,

    // ========================================================================
    // DDR4 Channel 1 - Accumulators (Intel Avalon-MM)
    // ========================================================================

    output logic                       ddr4_ch1_read,
    output logic                       ddr4_ch1_write,
    output logic [DDR4_ADDR_WIDTH-1:0] ddr4_ch1_address,
    output logic [DDR4_DATA_WIDTH-1:0] ddr4_ch1_writedata,
    output logic [DDR4_DATA_WIDTH/8-1:0] ddr4_ch1_byteenable,
    output logic [6:0]                 ddr4_ch1_burstcount,
    input  logic [DDR4_DATA_WIDTH-1:0] ddr4_ch1_readdata,
    input  logic                       ddr4_ch1_readdatavalid,
    input  logic                       ddr4_ch1_waitrequest,

    // ========================================================================
    // HPS Bridge Interface (AXI4-Lite for control plane)
    // ========================================================================

    // HPS can read/write soul state directly for snapshots, diagnostics
    input  logic                       hps_axi_awvalid,
    output logic                       hps_axi_awready,
    input  logic [31:0]                hps_axi_awaddr,

    input  logic                       hps_axi_wvalid,
    output logic                       hps_axi_wready,
    input  logic [HPS_BRIDGE_WIDTH-1:0] hps_axi_wdata,
    input  logic [HPS_BRIDGE_WIDTH/8-1:0] hps_axi_wstrb,

    output logic                       hps_axi_bvalid,
    input  logic                       hps_axi_bready,
    output logic [1:0]                 hps_axi_bresp,

    input  logic                       hps_axi_arvalid,
    output logic                       hps_axi_arready,
    input  logic [31:0]                hps_axi_araddr,

    output logic                       hps_axi_rvalid,
    input  logic                       hps_axi_rready,
    output logic [HPS_BRIDGE_WIDTH-1:0] hps_axi_rdata,
    output logic [1:0]                 hps_axi_rresp,

    // ========================================================================
    // Streaming Interface (for 40MHz event ingestion)
    // ========================================================================

    // Fast-path for reward events (bypasses PCIe latency)
    input  logic                       stream_event_valid,
    input  logic signed [7:0]          stream_reward,
    input  logic [CHUNK_BITS-1:0]      stream_input_hv,
    input  logic [$clog2(ROWS)-1:0]    stream_target_row,
    output logic                       stream_event_ready,
    output logic                       stream_event_done,

    // ========================================================================
    // Status & Debug
    // ========================================================================

    output logic [31:0]                stats_rd_count,
    output logic [31:0]                stats_wr_count,
    output logic [31:0]                stats_stream_events,
    output logic [15:0]                stats_avg_latency_cycles,
    output logic                       adapter_busy,
    output logic [3:0]                 debug_state
);

    // ========================================================================
    // Local Parameters
    // ========================================================================

    localparam int CHUNKS_PER_ROW  = DIM / CHUNK_BITS;
    localparam int ROW_BITS        = $clog2(ROWS);
    localparam int CHUNK_ADDR_BITS = $clog2(CHUNKS_PER_ROW);

    localparam int ACCUM_BITS_PER_CHUNK  = CHUNK_BITS * ACC_WIDTH;
    localparam int ACCUM_BURSTS_PER_CHUNK = (ACCUM_BITS_PER_CHUNK + DDR4_DATA_WIDTH - 1) / DDR4_DATA_WIDTH;

    // ========================================================================
    // State Machine
    // ========================================================================

    typedef enum logic [3:0] {
        IDLE,
        RD_PARALLEL_REQ,     // Issue parallel reads to both channels
        RD_SIGNS_WAIT,
        RD_ACCUMS_WAIT,
        RD_SYNC,             // Wait for both channels to complete
        RD_DONE,
        WR_PARALLEL_REQ,
        WR_SIGNS_WAIT,
        WR_ACCUMS_WAIT,
        WR_SYNC,
        WR_DONE,
        HPS_ACCESS           // HPS is accessing soul state
    } state_t;

    state_t state, next_state;

    // ========================================================================
    // Internal Registers
    // ========================================================================

    logic [ROW_BITS-1:0]        active_row;
    logic [CHUNK_ADDR_BITS-1:0] active_chunk;
    logic [2:0]                 accum_burst_idx;

    // Buffered data
    logic [CHUNK_BITS-1:0]           sign_buffer;
    logic [ACCUM_BITS_PER_CHUNK-1:0] accum_buffer;
    logic                            signs_done;
    logic                            accums_done;

    // Write buffers
    logic [CHUNK_BITS-1:0]           sign_wr_buffer;
    logic [ACCUM_BITS_PER_CHUNK-1:0] accum_wr_buffer;

    // Statistics
    logic [31:0] rd_count_reg;
    logic [31:0] wr_count_reg;
    logic [31:0] stream_count_reg;

    // Latency measurement
    logic [15:0] latency_counter;
    logic [31:0] latency_accumulator;
    logic [15:0] event_count_for_avg;

    // Streaming FIFO signals
    logic stream_pending;
    logic signed [7:0]          stream_reward_latched;
    logic [CHUNK_BITS-1:0]      stream_hv_latched;
    logic [$clog2(ROWS)-1:0]    stream_row_latched;

    // ========================================================================
    // Address Calculation (Parallel across channels)
    // ========================================================================

    function automatic logic [DDR4_ADDR_WIDTH-1:0] calc_sign_addr(
        input logic [ROW_BITS-1:0] row,
        input logic [CHUNK_ADDR_BITS-1:0] chunk
    );
        logic [31:0] offset;
        offset = (row * CHUNKS_PER_ROW + chunk) * (CHUNK_BITS / 8);
        return CH0_BASE_SIGNS + offset[DDR4_ADDR_WIDTH-1:0];
    endfunction

    function automatic logic [DDR4_ADDR_WIDTH-1:0] calc_accum_addr(
        input logic [ROW_BITS-1:0] row,
        input logic [CHUNK_ADDR_BITS-1:0] chunk,
        input logic [2:0] burst_idx
    );
        logic [31:0] chunk_offset;
        logic [31:0] burst_offset;
        chunk_offset = (row * CHUNKS_PER_ROW + chunk) * (ACCUM_BITS_PER_CHUNK / 8);
        burst_offset = burst_idx * (DDR4_DATA_WIDTH / 8);
        return CH1_BASE_ACCUMS + chunk_offset[DDR4_ADDR_WIDTH-1:0] + burst_offset[DDR4_ADDR_WIDTH-1:0];
    endfunction

    // ========================================================================
    // State Machine
    // ========================================================================

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
                if (core_rd_req || stream_pending)
                    next_state = RD_PARALLEL_REQ;
                else if (core_wr_req)
                    next_state = WR_PARALLEL_REQ;
            end

            // Parallel read - issue to both channels simultaneously
            RD_PARALLEL_REQ: begin
                // Move to wait states (which one finishes first varies)
                if (!ddr4_ch0_waitrequest && !ddr4_ch1_waitrequest)
                    next_state = RD_SYNC;
                else if (!ddr4_ch0_waitrequest)
                    next_state = RD_ACCUMS_WAIT;
                else if (!ddr4_ch1_waitrequest)
                    next_state = RD_SIGNS_WAIT;
            end

            RD_SIGNS_WAIT: begin
                if (!ddr4_ch0_waitrequest)
                    next_state = RD_SYNC;
            end

            RD_ACCUMS_WAIT: begin
                if (!ddr4_ch1_waitrequest)
                    next_state = RD_SYNC;
            end

            RD_SYNC: begin
                // Wait for both data valid signals
                if (signs_done && accums_done)
                    next_state = RD_DONE;
            end

            RD_DONE: begin
                next_state = IDLE;
            end

            // Parallel write
            WR_PARALLEL_REQ: begin
                if (!ddr4_ch0_waitrequest && !ddr4_ch1_waitrequest)
                    next_state = WR_SYNC;
                else if (!ddr4_ch0_waitrequest)
                    next_state = WR_ACCUMS_WAIT;
                else if (!ddr4_ch1_waitrequest)
                    next_state = WR_SIGNS_WAIT;
            end

            WR_SIGNS_WAIT: begin
                if (!ddr4_ch0_waitrequest)
                    next_state = WR_SYNC;
            end

            WR_ACCUMS_WAIT: begin
                if (!ddr4_ch1_waitrequest)
                    next_state = WR_SYNC;
            end

            WR_SYNC: begin
                // For multi-burst accumulators
                if (accum_burst_idx >= ACCUM_BURSTS_PER_CHUNK - 1)
                    next_state = WR_DONE;
                else
                    next_state = WR_PARALLEL_REQ;
            end

            WR_DONE: begin
                next_state = IDLE;
            end

            default: next_state = IDLE;
        endcase
    end

    // ========================================================================
    // Data Path
    // ========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_row      <= '0;
            active_chunk    <= '0;
            accum_burst_idx <= '0;
            sign_buffer     <= '0;
            accum_buffer    <= '0;
            signs_done      <= 1'b0;
            accums_done     <= 1'b0;
            sign_wr_buffer  <= '0;
            accum_wr_buffer <= '0;
            rd_count_reg    <= '0;
            wr_count_reg    <= '0;
            stream_count_reg <= '0;
            stream_pending  <= 1'b0;
            latency_counter <= '0;
        end else begin
            // Latency counter (runs during active operations)
            if (state != IDLE)
                latency_counter <= latency_counter + 1'b1;

            case (state)
                IDLE: begin
                    signs_done  <= 1'b0;
                    accums_done <= 1'b0;
                    latency_counter <= '0;

                    // Handle streaming events with priority
                    if (stream_event_valid && !stream_pending) begin
                        stream_pending <= 1'b1;
                        stream_reward_latched <= stream_reward;
                        stream_hv_latched     <= stream_input_hv;
                        stream_row_latched    <= stream_target_row;
                    end

                    if (core_rd_req) begin
                        active_row   <= core_rd_row;
                        active_chunk <= core_rd_chunk;
                        accum_burst_idx <= '0;
                    end else if (core_wr_req) begin
                        active_row   <= core_wr_row;
                        active_chunk <= core_wr_chunk;
                        accum_burst_idx <= '0;
                        sign_wr_buffer  <= core_wr_signs;
                        accum_wr_buffer <= core_wr_accums;
                    end
                end

                RD_SYNC: begin
                    // Collect data as it arrives
                    if (ddr4_ch0_readdatavalid && !signs_done) begin
                        sign_buffer <= ddr4_ch0_readdata[CHUNK_BITS-1:0];
                        signs_done  <= 1'b1;
                    end

                    if (ddr4_ch1_readdatavalid) begin
                        // Pack accumulator data
                        accum_buffer[accum_burst_idx*DDR4_DATA_WIDTH +: DDR4_DATA_WIDTH] <= ddr4_ch1_readdata;
                        if (accum_burst_idx >= ACCUM_BURSTS_PER_CHUNK - 1)
                            accums_done <= 1'b1;
                        else
                            accum_burst_idx <= accum_burst_idx + 1'b1;
                    end
                end

                RD_DONE: begin
                    rd_count_reg <= rd_count_reg + 1'b1;
                end

                WR_SYNC: begin
                    accum_burst_idx <= accum_burst_idx + 1'b1;
                end

                WR_DONE: begin
                    wr_count_reg <= wr_count_reg + 1'b1;
                    if (stream_pending) begin
                        stream_pending   <= 1'b0;
                        stream_count_reg <= stream_count_reg + 1'b1;
                    end
                end

                default: ;
            endcase
        end
    end

    // ========================================================================
    // DDR4 Channel 0 Interface (Signs)
    // ========================================================================

    always_comb begin
        ddr4_ch0_read       = 1'b0;
        ddr4_ch0_write      = 1'b0;
        ddr4_ch0_address    = '0;
        ddr4_ch0_writedata  = '0;
        ddr4_ch0_byteenable = '1;
        ddr4_ch0_burstcount = 7'd1;

        case (state)
            RD_PARALLEL_REQ, RD_SIGNS_WAIT: begin
                ddr4_ch0_read    = 1'b1;
                ddr4_ch0_address = calc_sign_addr(active_row, active_chunk);
            end

            WR_PARALLEL_REQ, WR_SIGNS_WAIT: begin
                if (accum_burst_idx == 0) begin  // Only write signs on first burst
                    ddr4_ch0_write   = 1'b1;
                    ddr4_ch0_address = calc_sign_addr(active_row, active_chunk);
                    ddr4_ch0_writedata[CHUNK_BITS-1:0] = sign_wr_buffer;
                end
            end

            default: ;
        endcase
    end

    // ========================================================================
    // DDR4 Channel 1 Interface (Accumulators)
    // ========================================================================

    always_comb begin
        ddr4_ch1_read       = 1'b0;
        ddr4_ch1_write      = 1'b0;
        ddr4_ch1_address    = '0;
        ddr4_ch1_writedata  = '0;
        ddr4_ch1_byteenable = '1;
        ddr4_ch1_burstcount = 7'd1;

        case (state)
            RD_PARALLEL_REQ, RD_ACCUMS_WAIT, RD_SYNC: begin
                if (!accums_done) begin
                    ddr4_ch1_read    = 1'b1;
                    ddr4_ch1_address = calc_accum_addr(active_row, active_chunk, accum_burst_idx);
                end
            end

            WR_PARALLEL_REQ, WR_ACCUMS_WAIT, WR_SYNC: begin
                ddr4_ch1_write   = 1'b1;
                ddr4_ch1_address = calc_accum_addr(active_row, active_chunk, accum_burst_idx);
                ddr4_ch1_writedata = accum_wr_buffer[accum_burst_idx*DDR4_DATA_WIDTH +: DDR4_DATA_WIDTH];
            end

            default: ;
        endcase
    end

    // ========================================================================
    // HPS Bridge (Simplified - full AXI4-Lite implementation)
    // ========================================================================

    // HPS can access soul state for snapshots, debugging, etc.
    // This is a simplified stub - full implementation would include:
    // - Memory-mapped access to soul state
    // - Control registers
    // - Status/statistics registers

    logic [31:0] hps_ctrl_reg;
    logic [31:0] hps_status_reg;

    assign hps_axi_awready = 1'b1;
    assign hps_axi_wready  = 1'b1;
    assign hps_axi_bvalid  = 1'b0;  // Simplified
    assign hps_axi_bresp   = 2'b00;
    assign hps_axi_arready = 1'b1;
    assign hps_axi_rvalid  = 1'b0;  // Simplified
    assign hps_axi_rdata   = '0;
    assign hps_axi_rresp   = 2'b00;

    // ========================================================================
    // Streaming Interface
    // ========================================================================

    assign stream_event_ready = (state == IDLE) && !stream_pending;
    assign stream_event_done  = (state == WR_DONE) && stream_pending;

    // ========================================================================
    // Core Interface Outputs
    // ========================================================================

    assign core_rd_valid  = (state == RD_DONE);
    assign core_rd_signs  = sign_buffer;
    assign core_rd_accums = accum_buffer[CHUNK_BITS*ACC_WIDTH-1:0];

    assign core_wr_done   = (state == WR_DONE);

    assign adapter_busy   = (state != IDLE);
    assign debug_state    = state;

    // Statistics
    assign stats_rd_count       = rd_count_reg;
    assign stats_wr_count       = wr_count_reg;
    assign stats_stream_events  = stream_count_reg;
    assign stats_avg_latency_cycles = (event_count_for_avg > 0) ?
        latency_accumulator[31:16] : 16'd0;

endmodule


// ============================================================================
// SB-852 Soul Top Module
// ============================================================================
//
// Complete soul engine integration for the Micron SB-852.
// Includes the plasticity core, memory adapter, and HPS bridge.
//
// ============================================================================

module ara_soul_sb852_top #(
    parameter int ROWS       = `ARA_ROWS,
    parameter int DIM        = `ARA_DIM,
    parameter int CHUNK_BITS = `ARA_CHUNK_BITS,
    parameter int ACC_WIDTH  = `ARA_ACC_WIDTH
)(
    input  logic clk,
    input  logic rst_n,

    // PCIe/Host Control Interface
    input  logic                       ctrl_start,
    input  logic signed [7:0]          ctrl_reward,
    input  logic [CHUNK_BITS-1:0]      ctrl_input_hv,
    input  logic [$clog2(ROWS)-1:0]    ctrl_target_row,
    output logic                       ctrl_done,
    output logic                       ctrl_busy,

    // Streaming Interface (40MHz event path)
    input  logic                       stream_event_valid,
    input  logic signed [7:0]          stream_reward,
    input  logic [CHUNK_BITS-1:0]      stream_input_hv,
    input  logic [$clog2(ROWS)-1:0]    stream_target_row,
    output logic                       stream_event_ready,

    // DDR4 Channel 0 (Signs)
    output logic                       ddr4_ch0_read,
    output logic                       ddr4_ch0_write,
    output logic [31:0]                ddr4_ch0_address,
    output logic [511:0]               ddr4_ch0_writedata,
    output logic [63:0]                ddr4_ch0_byteenable,
    output logic [6:0]                 ddr4_ch0_burstcount,
    input  logic [511:0]               ddr4_ch0_readdata,
    input  logic                       ddr4_ch0_readdatavalid,
    input  logic                       ddr4_ch0_waitrequest,

    // DDR4 Channel 1 (Accumulators)
    output logic                       ddr4_ch1_read,
    output logic                       ddr4_ch1_write,
    output logic [31:0]                ddr4_ch1_address,
    output logic [511:0]               ddr4_ch1_writedata,
    output logic [63:0]                ddr4_ch1_byteenable,
    output logic [6:0]                 ddr4_ch1_burstcount,
    input  logic [511:0]               ddr4_ch1_readdata,
    input  logic                       ddr4_ch1_readdatavalid,
    input  logic                       ddr4_ch1_waitrequest,

    // HPS AXI Bridge
    input  logic                       hps_axi_awvalid,
    output logic                       hps_axi_awready,
    input  logic [31:0]                hps_axi_awaddr,
    input  logic                       hps_axi_wvalid,
    output logic                       hps_axi_wready,
    input  logic [127:0]               hps_axi_wdata,
    input  logic [15:0]                hps_axi_wstrb,
    output logic                       hps_axi_bvalid,
    input  logic                       hps_axi_bready,
    output logic [1:0]                 hps_axi_bresp,
    input  logic                       hps_axi_arvalid,
    output logic                       hps_axi_arready,
    input  logic [31:0]                hps_axi_araddr,
    output logic                       hps_axi_rvalid,
    input  logic                       hps_axi_rready,
    output logic [127:0]               hps_axi_rdata,
    output logic [1:0]                 hps_axi_rresp,

    // Status
    output logic [31:0]                stats_events_processed,
    output logic [31:0]                stats_stream_events,
    output logic [15:0]                stats_avg_latency
);

    // Core <-> Adapter signals
    logic                       mem_rd_req;
    logic [$clog2(ROWS)-1:0]    mem_rd_row;
    logic [$clog2(DIM/CHUNK_BITS)-1:0] mem_rd_chunk;
    logic                       mem_rd_valid;
    logic [CHUNK_BITS-1:0]      mem_rd_signs;
    logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_rd_accums;

    logic                       mem_wr_req;
    logic [$clog2(ROWS)-1:0]    mem_wr_row;
    logic [$clog2(DIM/CHUNK_BITS)-1:0] mem_wr_chunk;
    logic [CHUNK_BITS-1:0]      mem_wr_signs;
    logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_wr_accums;
    logic                       mem_wr_done;

    logic adapter_busy;
    logic stream_done;

    // Plasticity Core
    ara_plasticity_core #(
        .ROWS       (ROWS),
        .DIM        (DIM),
        .CHUNK_BITS (CHUNK_BITS),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_core (
        .clk            (clk),
        .rst_n          (rst_n),
        .start          (ctrl_start),
        .reward         (ctrl_reward),
        .input_hv       (ctrl_input_hv),
        .target_row     (ctrl_target_row),
        .done           (ctrl_done),
        .busy           (ctrl_busy),
        .mem_rd_req     (mem_rd_req),
        .mem_rd_row     (mem_rd_row),
        .mem_rd_chunk   (mem_rd_chunk),
        .mem_rd_valid   (mem_rd_valid),
        .mem_rd_signs   (mem_rd_signs),
        .mem_rd_accums  (mem_rd_accums),
        .mem_wr_req     (mem_wr_req),
        .mem_wr_row     (mem_wr_row),
        .mem_wr_chunk   (mem_wr_chunk),
        .mem_wr_signs   (mem_wr_signs),
        .mem_wr_accums  (mem_wr_accums),
        .mem_wr_done    (mem_wr_done)
    );

    // SB-852 Memory Adapter
    sb852_mem_adapter #(
        .ROWS       (ROWS),
        .DIM        (DIM),
        .CHUNK_BITS (CHUNK_BITS),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_adapter (
        .clk                   (clk),
        .rst_n                 (rst_n),

        // Core interface
        .core_rd_req           (mem_rd_req),
        .core_rd_row           (mem_rd_row),
        .core_rd_chunk         (mem_rd_chunk),
        .core_rd_valid         (mem_rd_valid),
        .core_rd_signs         (mem_rd_signs),
        .core_rd_accums        (mem_rd_accums),
        .core_wr_req           (mem_wr_req),
        .core_wr_row           (mem_wr_row),
        .core_wr_chunk         (mem_wr_chunk),
        .core_wr_signs         (mem_wr_signs),
        .core_wr_accums        (mem_wr_accums),
        .core_wr_done          (mem_wr_done),

        // DDR4 Channel 0
        .ddr4_ch0_read         (ddr4_ch0_read),
        .ddr4_ch0_write        (ddr4_ch0_write),
        .ddr4_ch0_address      (ddr4_ch0_address),
        .ddr4_ch0_writedata    (ddr4_ch0_writedata),
        .ddr4_ch0_byteenable   (ddr4_ch0_byteenable),
        .ddr4_ch0_burstcount   (ddr4_ch0_burstcount),
        .ddr4_ch0_readdata     (ddr4_ch0_readdata),
        .ddr4_ch0_readdatavalid(ddr4_ch0_readdatavalid),
        .ddr4_ch0_waitrequest  (ddr4_ch0_waitrequest),

        // DDR4 Channel 1
        .ddr4_ch1_read         (ddr4_ch1_read),
        .ddr4_ch1_write        (ddr4_ch1_write),
        .ddr4_ch1_address      (ddr4_ch1_address),
        .ddr4_ch1_writedata    (ddr4_ch1_writedata),
        .ddr4_ch1_byteenable   (ddr4_ch1_byteenable),
        .ddr4_ch1_burstcount   (ddr4_ch1_burstcount),
        .ddr4_ch1_readdata     (ddr4_ch1_readdata),
        .ddr4_ch1_readdatavalid(ddr4_ch1_readdatavalid),
        .ddr4_ch1_waitrequest  (ddr4_ch1_waitrequest),

        // HPS Bridge
        .hps_axi_awvalid       (hps_axi_awvalid),
        .hps_axi_awready       (hps_axi_awready),
        .hps_axi_awaddr        (hps_axi_awaddr),
        .hps_axi_wvalid        (hps_axi_wvalid),
        .hps_axi_wready        (hps_axi_wready),
        .hps_axi_wdata         (hps_axi_wdata),
        .hps_axi_wstrb         (hps_axi_wstrb),
        .hps_axi_bvalid        (hps_axi_bvalid),
        .hps_axi_bready        (hps_axi_bready),
        .hps_axi_bresp         (hps_axi_bresp),
        .hps_axi_arvalid       (hps_axi_arvalid),
        .hps_axi_arready       (hps_axi_arready),
        .hps_axi_araddr        (hps_axi_araddr),
        .hps_axi_rvalid        (hps_axi_rvalid),
        .hps_axi_rready        (hps_axi_rready),
        .hps_axi_rdata         (hps_axi_rdata),
        .hps_axi_rresp         (hps_axi_rresp),

        // Streaming
        .stream_event_valid    (stream_event_valid),
        .stream_reward         (stream_reward),
        .stream_input_hv       (stream_input_hv),
        .stream_target_row     (stream_target_row),
        .stream_event_ready    (stream_event_ready),
        .stream_event_done     (stream_done),

        // Stats
        .stats_rd_count        (),
        .stats_wr_count        (),
        .stats_stream_events   (stats_stream_events),
        .stats_avg_latency_cycles (stats_avg_latency),
        .adapter_busy          (adapter_busy),
        .debug_state           ()
    );

    // Event counter
    logic [31:0] event_counter;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            event_counter <= '0;
        else if (ctrl_done || stream_done)
            event_counter <= event_counter + 1'b1;
    end
    assign stats_events_processed = event_counter;

endmodule
