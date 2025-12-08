// =============================================================================
// Stratix-10 HBM Adapter - Intel S10 GX2800 Platform
// =============================================================================
//
// Bridges ara_plasticity_core's abstract memory interface to Intel HBM2.
//
// Platform: Intel Stratix-10 GX2800 (or similar with HBM)
//   - Uses Intel EMIF IP for HBM access
//   - Avalon-MM interface (different from Xilinx AXI)
//
// Memory Layout: Same as FK33
//   [0x0000_0000_0000_0000] - Sign bits region (4.2 MB)
//   [0x0000_0000_0100_0000] - Accumulator region (29.4 MB)
//
// =============================================================================

`timescale 1ns / 1ps

`include "../common/ara_soul_config.svh"

module s10_hbm_adapter #(
    parameter int ROWS        = ARA_ROWS,
    parameter int DIM         = ARA_DIM,
    parameter int CHUNK_BITS  = ARA_CHUNK_BITS,
    parameter int ACC_WIDTH   = ARA_ACC_WIDTH,

    // Avalon parameters (match Intel EMIF IP)
    parameter int AVL_ADDR_WIDTH = 32,
    parameter int AVL_DATA_WIDTH = 256,
    parameter int AVL_BE_WIDTH   = AVL_DATA_WIDTH / 8
)(
    input  logic clk_hbm,          // HBM EMIF clock
    input  logic clk_core,         // Core logic clock
    input  logic rst_n,

    // === From Plasticity Controller ===
    input  logic                           mem_req,
    output logic                           mem_ready,
    input  logic [$clog2(ROWS)-1:0]        mem_row_addr,
    input  logic [$clog2(DIM/CHUNK_BITS)-1:0] mem_chunk_addr,
    input  logic                           mem_we,
    input  logic [CHUNK_BITS-1:0]          mem_core_out,
    input  logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_out,
    output logic [CHUNK_BITS-1:0]          mem_core_in,
    output logic [CHUNK_BITS*ACC_WIDTH-1:0] mem_accum_in,

    // === Avalon-MM Interface to Intel HBM EMIF ===
    output logic [AVL_ADDR_WIDTH-1:0]  avl_address,
    output logic                       avl_read,
    output logic                       avl_write,
    output logic [AVL_DATA_WIDTH-1:0]  avl_writedata,
    output logic [AVL_BE_WIDTH-1:0]    avl_byteenable,
    output logic [6:0]                 avl_burstcount,
    input  logic [AVL_DATA_WIDTH-1:0]  avl_readdata,
    input  logic                       avl_readdatavalid,
    input  logic                       avl_waitrequest
);

    // =========================================================================
    // Memory Region Constants (Same as FK33)
    // =========================================================================

    localparam longint REGION_SIGNS  = 64'h0000_0000_0000_0000;
    localparam longint REGION_ACCUMS = 64'h0000_0000_0100_0000;

    localparam int SIGN_BYTES_PER_CHUNK  = CHUNK_BITS / 8;
    localparam int ACCUM_BYTES_PER_CHUNK = (CHUNK_BITS * ACC_WIDTH) / 8;
    localparam int AVL_BYTES = AVL_DATA_WIDTH / 8;

    localparam int SIGN_BEATS  = (SIGN_BYTES_PER_CHUNK + AVL_BYTES - 1) / AVL_BYTES;
    localparam int ACCUM_BEATS = (ACCUM_BYTES_PER_CHUNK + AVL_BYTES - 1) / AVL_BYTES;

    // =========================================================================
    // Address Calculation
    // =========================================================================

    logic [AVL_ADDR_WIDTH-1:0] sign_addr;
    logic [AVL_ADDR_WIDTH-1:0] accum_addr;

    always_comb begin
        sign_addr = REGION_SIGNS[AVL_ADDR_WIDTH-1:0] +
                    (mem_row_addr * (DIM / 8)) +
                    (mem_chunk_addr * SIGN_BYTES_PER_CHUNK);

        accum_addr = REGION_ACCUMS[AVL_ADDR_WIDTH-1:0] +
                     (mem_row_addr * (DIM * ACC_WIDTH / 8)) +
                     (mem_chunk_addr * ACCUM_BYTES_PER_CHUNK);
    end

    // =========================================================================
    // State Machine (Avalon-MM Protocol)
    // =========================================================================

    typedef enum logic [3:0] {
        A_IDLE,
        A_READ_SIGN,
        A_READ_SIGN_WAIT,
        A_READ_ACCUM,
        A_READ_ACCUM_WAIT,
        A_READ_DONE,
        A_WRITE_SIGN,
        A_WRITE_SIGN_WAIT,
        A_WRITE_ACCUM,
        A_WRITE_ACCUM_WAIT,
        A_WRITE_DONE
    } adapter_state_t;

    adapter_state_t state;
    logic [3:0] beat_cnt;

    logic [CHUNK_BITS-1:0]            sign_buffer;
    logic [CHUNK_BITS*ACC_WIDTH-1:0]  accum_buffer;

    // =========================================================================
    // Avalon-MM State Machine
    // =========================================================================

    always_ff @(posedge clk_hbm or negedge rst_n) begin
        if (!rst_n) begin
            state        <= A_IDLE;
            beat_cnt     <= '0;
            mem_ready    <= 1'b0;
            avl_read     <= 1'b0;
            avl_write    <= 1'b0;
            sign_buffer  <= '0;
            accum_buffer <= '0;
        end else begin
            mem_ready <= 1'b0;

            case (state)
                A_IDLE: begin
                    if (mem_req && !mem_we) begin
                        state <= A_READ_SIGN;
                    end else if (mem_req && mem_we) begin
                        sign_buffer  <= mem_core_out;
                        accum_buffer <= mem_accum_out;
                        state <= A_WRITE_SIGN;
                    end
                end

                // === READ SEQUENCE ===
                A_READ_SIGN: begin
                    avl_address    <= sign_addr;
                    avl_read       <= 1'b1;
                    avl_burstcount <= SIGN_BEATS[6:0];
                    beat_cnt       <= '0;

                    if (!avl_waitrequest) begin
                        avl_read <= 1'b0;
                        state    <= A_READ_SIGN_WAIT;
                    end
                end

                A_READ_SIGN_WAIT: begin
                    if (avl_readdatavalid) begin
                        sign_buffer[beat_cnt * AVL_DATA_WIDTH +: AVL_DATA_WIDTH] <= avl_readdata;

                        if (beat_cnt >= SIGN_BEATS - 1) begin
                            state <= A_READ_ACCUM;
                        end else begin
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                A_READ_ACCUM: begin
                    avl_address    <= accum_addr;
                    avl_read       <= 1'b1;
                    avl_burstcount <= ACCUM_BEATS[6:0];
                    beat_cnt       <= '0;

                    if (!avl_waitrequest) begin
                        avl_read <= 1'b0;
                        state    <= A_READ_ACCUM_WAIT;
                    end
                end

                A_READ_ACCUM_WAIT: begin
                    if (avl_readdatavalid) begin
                        accum_buffer[beat_cnt * AVL_DATA_WIDTH +: AVL_DATA_WIDTH] <= avl_readdata;

                        if (beat_cnt >= ACCUM_BEATS - 1) begin
                            state <= A_READ_DONE;
                        end else begin
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                A_READ_DONE: begin
                    mem_ready <= 1'b1;
                    state     <= A_IDLE;
                end

                // === WRITE SEQUENCE ===
                A_WRITE_SIGN: begin
                    avl_address    <= sign_addr;
                    avl_write      <= 1'b1;
                    avl_writedata  <= sign_buffer[beat_cnt * AVL_DATA_WIDTH +: AVL_DATA_WIDTH];
                    avl_byteenable <= {AVL_BE_WIDTH{1'b1}};
                    avl_burstcount <= SIGN_BEATS[6:0];

                    if (!avl_waitrequest) begin
                        if (beat_cnt >= SIGN_BEATS - 1) begin
                            avl_write <= 1'b0;
                            state     <= A_WRITE_ACCUM;
                            beat_cnt  <= '0;
                        end else begin
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                A_WRITE_ACCUM: begin
                    avl_address    <= accum_addr;
                    avl_write      <= 1'b1;
                    avl_writedata  <= accum_buffer[beat_cnt * AVL_DATA_WIDTH +: AVL_DATA_WIDTH];
                    avl_byteenable <= {AVL_BE_WIDTH{1'b1}};
                    avl_burstcount <= ACCUM_BEATS[6:0];

                    if (!avl_waitrequest) begin
                        if (beat_cnt >= ACCUM_BEATS - 1) begin
                            avl_write <= 1'b0;
                            state     <= A_WRITE_DONE;
                        end else begin
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                A_WRITE_DONE: begin
                    mem_ready <= 1'b1;
                    state     <= A_IDLE;
                end

                default: state <= A_IDLE;
            endcase
        end
    end

    // =========================================================================
    // Output to Plasticity Core
    // =========================================================================

    assign mem_core_in  = sign_buffer;
    assign mem_accum_in = accum_buffer;

endmodule
