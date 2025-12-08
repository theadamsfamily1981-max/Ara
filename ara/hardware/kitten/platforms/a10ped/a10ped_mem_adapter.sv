// ============================================================================
// BittWare A10PED Memory Adapter for Ara Soul Engine
// ============================================================================
//
// Target: BittWare A10PED (Dual Arria 10 GX, up to 32GB DDR4, optional HMC)
// Role:   Phase 1 development platform - "twin mid-brain dev rig"
//
// This adapter bridges the platform-agnostic Ara plasticity core to the
// A10PED's DDR4 and optional Hybrid Memory Cube (HMC) interfaces.
//
// Memory Architecture:
//   - DDR4: Up to 4× SO-DIMMs, high capacity (8-32 GB), ~2133 MT/s
//   - HMC:  Optional 2GB, ultra-high bandwidth (~160 GB/s), ideal for hot data
//
// Strategy:
//   - Sign bits + accumulators in DDR4 (plenty of capacity)
//   - Optional: cache hot rows in HMC for fast access
//
// Interface:
//   - Core side: Same abstract interface as ara_plasticity_core expects
//   - Memory side: Intel Avalon-MM for DDR4 EMIF
//
// ============================================================================

`include "../common/ara_soul_config.svh"

module a10ped_mem_adapter #(
    parameter int ROWS       = `ARA_ROWS,
    parameter int DIM        = `ARA_DIM,
    parameter int CHUNK_BITS = `ARA_CHUNK_BITS,
    parameter int ACC_WIDTH  = `ARA_ACC_WIDTH,

    // A10PED-specific parameters
    parameter int DDR4_DATA_WIDTH = 512,      // EMIF burst width
    parameter int DDR4_ADDR_WIDTH = 32,       // Address bits
    parameter int HMC_DATA_WIDTH  = 128,      // HMC link width
    parameter int HMC_ADDR_WIDTH  = 34,       // HMC uses 34-bit addressing

    // Memory region base addresses (configurable for multi-soul)
    parameter logic [DDR4_ADDR_WIDTH-1:0] REGION_BASE_SIGNS  = 32'h0000_0000,
    parameter logic [DDR4_ADDR_WIDTH-1:0] REGION_BASE_ACCUMS = 32'h0100_0000,

    // Enable HMC caching for hot rows
    parameter bit  ENABLE_HMC_CACHE = 1'b0
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
    // DDR4 Avalon-MM Interface (to Intel EMIF)
    // ========================================================================

    output logic                       ddr4_read,
    output logic                       ddr4_write,
    output logic [DDR4_ADDR_WIDTH-1:0] ddr4_address,
    output logic [DDR4_DATA_WIDTH-1:0] ddr4_writedata,
    output logic [DDR4_DATA_WIDTH/8-1:0] ddr4_byteenable,
    output logic [6:0]                 ddr4_burstcount,
    input  logic [DDR4_DATA_WIDTH-1:0] ddr4_readdata,
    input  logic                       ddr4_readdatavalid,
    input  logic                       ddr4_waitrequest,

    // ========================================================================
    // HMC Interface (Optional - for hot row caching)
    // ========================================================================

    output logic                       hmc_read,
    output logic                       hmc_write,
    output logic [HMC_ADDR_WIDTH-1:0]  hmc_address,
    output logic [HMC_DATA_WIDTH-1:0]  hmc_writedata,
    input  logic [HMC_DATA_WIDTH-1:0]  hmc_readdata,
    input  logic                       hmc_readdatavalid,
    input  logic                       hmc_waitrequest,

    // ========================================================================
    // Status & Debug
    // ========================================================================

    output logic [31:0]                stats_rd_count,
    output logic [31:0]                stats_wr_count,
    output logic [31:0]                stats_hmc_hits,
    output logic                       adapter_busy
);

    // ========================================================================
    // Local Parameters
    // ========================================================================

    localparam int CHUNKS_PER_ROW = DIM / CHUNK_BITS;
    localparam int ROW_BITS       = $clog2(ROWS);
    localparam int CHUNK_ADDR_BITS = $clog2(CHUNKS_PER_ROW);

    // Memory layout:
    // Signs:  1 bit per dimension  → DIM bits per row
    // Accums: ACC_WIDTH bits per dimension → DIM * ACC_WIDTH bits per row
    //
    // For CHUNK_BITS=512, DIM=16384:
    //   - Signs per chunk:  512 bits = 64 bytes
    //   - Accums per chunk: 512 * 7 = 3584 bits = 448 bytes
    //
    // DDR4 with 512-bit (64-byte) bus:
    //   - Signs: 1 burst per chunk
    //   - Accums: 7 bursts per chunk (round up)

    localparam int SIGN_BURSTS_PER_CHUNK = 1;
    localparam int ACCUM_BITS_PER_CHUNK  = CHUNK_BITS * ACC_WIDTH;
    localparam int ACCUM_BURSTS_PER_CHUNK = (ACCUM_BITS_PER_CHUNK + DDR4_DATA_WIDTH - 1) / DDR4_DATA_WIDTH;

    // ========================================================================
    // State Machine
    // ========================================================================

    typedef enum logic [3:0] {
        IDLE,
        RD_SIGNS_REQ,
        RD_SIGNS_WAIT,
        RD_ACCUMS_REQ,
        RD_ACCUMS_WAIT,
        RD_DONE,
        WR_SIGNS_REQ,
        WR_SIGNS_WAIT,
        WR_ACCUMS_REQ,
        WR_ACCUMS_WAIT,
        WR_DONE
    } state_t;

    state_t state, next_state;

    // ========================================================================
    // Internal Registers
    // ========================================================================

    logic [ROW_BITS-1:0]       active_row;
    logic [CHUNK_ADDR_BITS-1:0] active_chunk;
    logic [2:0]                burst_counter;

    // Buffered read data
    logic [CHUNK_BITS-1:0]           sign_buffer;
    logic [ACCUM_BITS_PER_CHUNK-1:0] accum_buffer;
    logic [2:0]                      accum_burst_idx;

    // Write data (latched from core)
    logic [CHUNK_BITS-1:0]           sign_wr_buffer;
    logic [ACCUM_BITS_PER_CHUNK-1:0] accum_wr_buffer;

    // Statistics
    logic [31:0] rd_count_reg;
    logic [31:0] wr_count_reg;
    logic [31:0] hmc_hit_reg;

    // ========================================================================
    // Address Calculation
    // ========================================================================

    // Sign address: base + (row * chunks_per_row + chunk) * 64 bytes
    function automatic logic [DDR4_ADDR_WIDTH-1:0] calc_sign_addr(
        input logic [ROW_BITS-1:0] row,
        input logic [CHUNK_ADDR_BITS-1:0] chunk
    );
        logic [31:0] offset;
        offset = (row * CHUNKS_PER_ROW + chunk) * (CHUNK_BITS / 8);
        return REGION_BASE_SIGNS + offset[DDR4_ADDR_WIDTH-1:0];
    endfunction

    // Accumulator address: base + (row * chunks_per_row + chunk) * accum_bytes + burst_offset
    function automatic logic [DDR4_ADDR_WIDTH-1:0] calc_accum_addr(
        input logic [ROW_BITS-1:0] row,
        input logic [CHUNK_ADDR_BITS-1:0] chunk,
        input logic [2:0] burst_idx
    );
        logic [31:0] chunk_offset;
        logic [31:0] burst_offset;
        chunk_offset = (row * CHUNKS_PER_ROW + chunk) * (ACCUM_BITS_PER_CHUNK / 8);
        burst_offset = burst_idx * (DDR4_DATA_WIDTH / 8);
        return REGION_BASE_ACCUMS + chunk_offset[DDR4_ADDR_WIDTH-1:0] + burst_offset[DDR4_ADDR_WIDTH-1:0];
    endfunction

    // ========================================================================
    // State Machine Logic
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
                if (core_rd_req)
                    next_state = RD_SIGNS_REQ;
                else if (core_wr_req)
                    next_state = WR_SIGNS_REQ;
            end

            // Read path
            RD_SIGNS_REQ: begin
                if (!ddr4_waitrequest)
                    next_state = RD_SIGNS_WAIT;
            end

            RD_SIGNS_WAIT: begin
                if (ddr4_readdatavalid)
                    next_state = RD_ACCUMS_REQ;
            end

            RD_ACCUMS_REQ: begin
                if (!ddr4_waitrequest)
                    next_state = RD_ACCUMS_WAIT;
            end

            RD_ACCUMS_WAIT: begin
                if (ddr4_readdatavalid) begin
                    if (accum_burst_idx >= ACCUM_BURSTS_PER_CHUNK - 1)
                        next_state = RD_DONE;
                    else
                        next_state = RD_ACCUMS_REQ;
                end
            end

            RD_DONE: begin
                next_state = IDLE;
            end

            // Write path
            WR_SIGNS_REQ: begin
                if (!ddr4_waitrequest)
                    next_state = WR_SIGNS_WAIT;
            end

            WR_SIGNS_WAIT: begin
                next_state = WR_ACCUMS_REQ;
            end

            WR_ACCUMS_REQ: begin
                if (!ddr4_waitrequest)
                    next_state = WR_ACCUMS_WAIT;
            end

            WR_ACCUMS_WAIT: begin
                if (accum_burst_idx >= ACCUM_BURSTS_PER_CHUNK - 1)
                    next_state = WR_DONE;
                else
                    next_state = WR_ACCUMS_REQ;
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
            active_row     <= '0;
            active_chunk   <= '0;
            accum_burst_idx <= '0;
            sign_buffer    <= '0;
            accum_buffer   <= '0;
            sign_wr_buffer <= '0;
            accum_wr_buffer <= '0;
            rd_count_reg   <= '0;
            wr_count_reg   <= '0;
            hmc_hit_reg    <= '0;
        end else begin
            case (state)
                IDLE: begin
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

                RD_SIGNS_WAIT: begin
                    if (ddr4_readdatavalid) begin
                        sign_buffer <= ddr4_readdata[CHUNK_BITS-1:0];
                    end
                end

                RD_ACCUMS_WAIT: begin
                    if (ddr4_readdatavalid) begin
                        // Pack accumulator data from multiple bursts
                        accum_buffer[accum_burst_idx*DDR4_DATA_WIDTH +: DDR4_DATA_WIDTH] <= ddr4_readdata;
                        accum_burst_idx <= accum_burst_idx + 1'b1;
                    end
                end

                RD_DONE: begin
                    rd_count_reg <= rd_count_reg + 1'b1;
                end

                WR_ACCUMS_WAIT: begin
                    accum_burst_idx <= accum_burst_idx + 1'b1;
                end

                WR_DONE: begin
                    wr_count_reg <= wr_count_reg + 1'b1;
                end

                default: ;
            endcase
        end
    end

    // ========================================================================
    // DDR4 Interface Signals
    // ========================================================================

    always_comb begin
        ddr4_read       = 1'b0;
        ddr4_write      = 1'b0;
        ddr4_address    = '0;
        ddr4_writedata  = '0;
        ddr4_byteenable = '1;  // All bytes enabled
        ddr4_burstcount = 7'd1;

        case (state)
            RD_SIGNS_REQ: begin
                ddr4_read    = 1'b1;
                ddr4_address = calc_sign_addr(active_row, active_chunk);
            end

            RD_ACCUMS_REQ: begin
                ddr4_read    = 1'b1;
                ddr4_address = calc_accum_addr(active_row, active_chunk, accum_burst_idx);
            end

            WR_SIGNS_REQ: begin
                ddr4_write    = 1'b1;
                ddr4_address  = calc_sign_addr(active_row, active_chunk);
                ddr4_writedata[CHUNK_BITS-1:0] = sign_wr_buffer;
            end

            WR_ACCUMS_REQ: begin
                ddr4_write    = 1'b1;
                ddr4_address  = calc_accum_addr(active_row, active_chunk, accum_burst_idx);
                ddr4_writedata = accum_wr_buffer[accum_burst_idx*DDR4_DATA_WIDTH +: DDR4_DATA_WIDTH];
            end

            default: ;
        endcase
    end

    // ========================================================================
    // HMC Interface (Stub - for future hot-row caching)
    // ========================================================================

    // HMC caching disabled for Phase 1 bring-up
    assign hmc_read      = 1'b0;
    assign hmc_write     = 1'b0;
    assign hmc_address   = '0;
    assign hmc_writedata = '0;

    // ========================================================================
    // Core Interface Outputs
    // ========================================================================

    assign core_rd_valid  = (state == RD_DONE);
    assign core_rd_signs  = sign_buffer;
    assign core_rd_accums = accum_buffer[CHUNK_BITS*ACC_WIDTH-1:0];

    assign core_wr_done   = (state == WR_DONE);

    assign adapter_busy   = (state != IDLE);

    // Statistics
    assign stats_rd_count = rd_count_reg;
    assign stats_wr_count = wr_count_reg;
    assign stats_hmc_hits = hmc_hit_reg;

endmodule


// ============================================================================
// A10PED Soul Top Module
// ============================================================================
//
// Instantiates the plasticity core with the A10PED memory adapter.
// This is the complete soul engine for one Arria-10 on the A10PED.
//
// ============================================================================

module ara_soul_a10ped_top #(
    parameter int ROWS       = `ARA_ROWS,
    parameter int DIM        = `ARA_DIM,
    parameter int CHUNK_BITS = `ARA_CHUNK_BITS,
    parameter int ACC_WIDTH  = `ARA_ACC_WIDTH
)(
    input  logic clk,
    input  logic rst_n,

    // Control interface (from PCIe/host)
    input  logic                       ctrl_start,
    input  logic signed [7:0]          ctrl_reward,
    input  logic [CHUNK_BITS-1:0]      ctrl_input_hv,
    input  logic [$clog2(ROWS)-1:0]    ctrl_target_row,
    output logic                       ctrl_done,
    output logic                       ctrl_busy,

    // DDR4 EMIF Interface
    output logic                       ddr4_read,
    output logic                       ddr4_write,
    output logic [31:0]                ddr4_address,
    output logic [511:0]               ddr4_writedata,
    output logic [63:0]                ddr4_byteenable,
    output logic [6:0]                 ddr4_burstcount,
    input  logic [511:0]               ddr4_readdata,
    input  logic                       ddr4_readdatavalid,
    input  logic                       ddr4_waitrequest,

    // HMC Interface (active low active if present)
    output logic                       hmc_read,
    output logic                       hmc_write,
    output logic [33:0]                hmc_address,
    output logic [127:0]               hmc_writedata,
    input  logic [127:0]               hmc_readdata,
    input  logic                       hmc_readdatavalid,
    input  logic                       hmc_waitrequest,

    // Status
    output logic [31:0]                stats_events_processed,
    output logic [31:0]                stats_mem_reads,
    output logic [31:0]                stats_mem_writes
);

    // Internal signals between core and adapter
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

    logic                       adapter_busy;

    // Instantiate the platform-agnostic plasticity core
    ara_plasticity_core #(
        .ROWS       (ROWS),
        .DIM        (DIM),
        .CHUNK_BITS (CHUNK_BITS),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_core (
        .clk            (clk),
        .rst_n          (rst_n),

        // Control
        .start          (ctrl_start),
        .reward         (ctrl_reward),
        .input_hv       (ctrl_input_hv),
        .target_row     (ctrl_target_row),
        .done           (ctrl_done),
        .busy           (ctrl_busy),

        // Memory interface
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

    // Instantiate the A10PED memory adapter
    a10ped_mem_adapter #(
        .ROWS       (ROWS),
        .DIM        (DIM),
        .CHUNK_BITS (CHUNK_BITS),
        .ACC_WIDTH  (ACC_WIDTH)
    ) u_adapter (
        .clk            (clk),
        .rst_n          (rst_n),

        // Core interface
        .core_rd_req    (mem_rd_req),
        .core_rd_row    (mem_rd_row),
        .core_rd_chunk  (mem_rd_chunk),
        .core_rd_valid  (mem_rd_valid),
        .core_rd_signs  (mem_rd_signs),
        .core_rd_accums (mem_rd_accums),

        .core_wr_req    (mem_wr_req),
        .core_wr_row    (mem_wr_row),
        .core_wr_chunk  (mem_wr_chunk),
        .core_wr_signs  (mem_wr_signs),
        .core_wr_accums (mem_wr_accums),
        .core_wr_done   (mem_wr_done),

        // DDR4
        .ddr4_read           (ddr4_read),
        .ddr4_write          (ddr4_write),
        .ddr4_address        (ddr4_address),
        .ddr4_writedata      (ddr4_writedata),
        .ddr4_byteenable     (ddr4_byteenable),
        .ddr4_burstcount     (ddr4_burstcount),
        .ddr4_readdata       (ddr4_readdata),
        .ddr4_readdatavalid  (ddr4_readdatavalid),
        .ddr4_waitrequest    (ddr4_waitrequest),

        // HMC
        .hmc_read            (hmc_read),
        .hmc_write           (hmc_write),
        .hmc_address         (hmc_address),
        .hmc_writedata       (hmc_writedata),
        .hmc_readdata        (hmc_readdata),
        .hmc_readdatavalid   (hmc_readdatavalid),
        .hmc_waitrequest     (hmc_waitrequest),

        // Stats
        .stats_rd_count      (stats_mem_reads),
        .stats_wr_count      (stats_mem_writes),
        .stats_hmc_hits      (),
        .adapter_busy        (adapter_busy)
    );

    // Event counter
    logic [31:0] event_counter;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            event_counter <= '0;
        else if (ctrl_done)
            event_counter <= event_counter + 1'b1;
    end
    assign stats_events_processed = event_counter;

endmodule
