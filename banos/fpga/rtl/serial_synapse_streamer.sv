/**
 * SERIAL SYNAPSE STREAMER
 * =======================
 *
 * Bio-Affective Neuromorphic Operating System
 * Feeds weights to bit-serial neurons from URAM/BRAM.
 *
 * THE BANDWIDTH CHALLENGE:
 *   URAM has 72-bit wide read ports.
 *   Bit-serial neurons need 1 bit per cycle.
 *   Solution: Read wide, stream narrow.
 *
 * ARCHITECTURE:
 *   - Reads 72-bit words from URAM (9 x 8-bit weights)
 *   - Unpacks into shift registers
 *   - Streams bits to neurons (1 bit/cycle or 1 byte/neuron)
 *   - Supports multiple stream modes for flexibility
 *
 * STREAM MODES:
 *   MODE_BIT_SERIAL:  Output 1 bit per neuron per cycle (for true bit-serial)
 *   MODE_BYTE_SERIAL: Output 1 byte per neuron per cycle (for byte-serial)
 *   MODE_PACKED:      Output 9 bytes simultaneously (for parallel bank)
 */

module serial_synapse_streamer #(
    parameter URAM_ADDR_WIDTH = 16,
    parameter WEIGHT_WIDTH    = 8,
    parameter WEIGHTS_PER_WORD = 9,    // 72 bits / 8 bits = 9 weights
    parameter NUM_STREAMS     = 9      // Parallel output streams
)(
    input  wire                         clk,
    input  wire                         rst_n,

    // =========================================================================
    // Control Interface
    // =========================================================================
    input  wire                         load_weights,   // Pulse to start new group
    input  wire [URAM_ADDR_WIDTH-1:0]   base_addr,      // Starting URAM address
    input  wire [7:0]                   num_words,      // Words to stream
    input  wire [1:0]                   stream_mode,    // 0=bit, 1=byte, 2=packed

    output reg                          busy,
    output reg                          stream_valid,
    output reg                          last_weight,

    // =========================================================================
    // URAM Interface (72-bit wide)
    // =========================================================================
    output reg  [URAM_ADDR_WIDTH-1:0]   uram_addr,
    output reg                          uram_rd_en,
    input  wire [71:0]                  uram_rdata,
    input  wire                         uram_rdata_valid,

    // =========================================================================
    // Output Streams (to neurons)
    // =========================================================================

    // Bit-serial mode: 1 bit per stream per cycle
    output wire [NUM_STREAMS-1:0]       bit_streams,

    // Byte-serial mode: 1 byte per stream per cycle
    output wire [NUM_STREAMS*WEIGHT_WIDTH-1:0] byte_streams,

    // Which stream is active
    output reg  [$clog2(NUM_STREAMS)-1:0] active_stream_idx
);

    // Stream mode encoding
    localparam MODE_BIT_SERIAL  = 2'd0;
    localparam MODE_BYTE_SERIAL = 2'd1;
    localparam MODE_PACKED      = 2'd2;

    // =========================================================================
    // Internal State
    // =========================================================================

    // Shift registers for weight storage
    reg [WEIGHT_WIDTH-1:0] weight_regs [NUM_STREAMS-1:0];

    // Counters
    reg [7:0]  word_count;
    reg [3:0]  bit_idx;
    reg [$clog2(NUM_STREAMS)-1:0] stream_idx;

    // FSM
    typedef enum logic [2:0] {
        IDLE,
        FETCH_WORD,
        WAIT_DATA,
        UNPACK,
        STREAM_BITS,
        STREAM_BYTES,
        NEXT_WORD
    } state_t;

    state_t state_r;

    // =========================================================================
    // Output Assignment
    // =========================================================================

    // Bit-serial: LSB of each shift register
    genvar i;
    generate
        for (i = 0; i < NUM_STREAMS; i = i + 1) begin : bit_out_gen
            assign bit_streams[i] = weight_regs[i][0];
        end
    endgenerate

    // Byte-serial: full byte from active stream
    generate
        for (i = 0; i < NUM_STREAMS; i = i + 1) begin : byte_out_gen
            assign byte_streams[(i+1)*WEIGHT_WIDTH-1 : i*WEIGHT_WIDTH] = weight_regs[i];
        end
    endgenerate

    // =========================================================================
    // Main FSM
    // =========================================================================

    integer j;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_r       <= IDLE;
            busy          <= 1'b0;
            stream_valid  <= 1'b0;
            last_weight   <= 1'b0;
            uram_rd_en    <= 1'b0;
            uram_addr     <= '0;
            word_count    <= '0;
            bit_idx       <= '0;
            stream_idx    <= '0;
            active_stream_idx <= '0;

            for (j = 0; j < NUM_STREAMS; j = j + 1) begin
                weight_regs[j] <= '0;
            end

        end else begin
            // Default signals
            uram_rd_en   <= 1'b0;
            stream_valid <= 1'b0;
            last_weight  <= 1'b0;

            case (state_r)
                // =============================================================
                IDLE: begin
                    busy <= 1'b0;
                    if (load_weights) begin
                        busy       <= 1'b1;
                        uram_addr  <= base_addr;
                        word_count <= num_words;
                        state_r    <= FETCH_WORD;
                    end
                end

                // =============================================================
                FETCH_WORD: begin
                    uram_rd_en <= 1'b1;
                    state_r    <= WAIT_DATA;
                end

                // =============================================================
                WAIT_DATA: begin
                    if (uram_rdata_valid) begin
                        state_r <= UNPACK;
                    end
                end

                // =============================================================
                UNPACK: begin
                    // Unpack 72-bit word into 9 x 8-bit weight registers
                    weight_regs[0] <= uram_rdata[7:0];
                    weight_regs[1] <= uram_rdata[15:8];
                    weight_regs[2] <= uram_rdata[23:16];
                    weight_regs[3] <= uram_rdata[31:24];
                    weight_regs[4] <= uram_rdata[39:32];
                    weight_regs[5] <= uram_rdata[47:40];
                    weight_regs[6] <= uram_rdata[55:48];
                    weight_regs[7] <= uram_rdata[63:56];
                    weight_regs[8] <= uram_rdata[71:64];

                    bit_idx    <= WEIGHT_WIDTH - 1;
                    stream_idx <= '0;

                    case (stream_mode)
                        MODE_BIT_SERIAL:  state_r <= STREAM_BITS;
                        MODE_BYTE_SERIAL: state_r <= STREAM_BYTES;
                        MODE_PACKED:      state_r <= NEXT_WORD;  // All bytes valid immediately
                        default:          state_r <= STREAM_BYTES;
                    endcase

                    // For packed mode, signal valid immediately
                    if (stream_mode == MODE_PACKED) begin
                        stream_valid <= 1'b1;
                    end
                end

                // =============================================================
                STREAM_BITS: begin
                    // Output 1 bit from all streams, then shift
                    stream_valid <= 1'b1;

                    // Shift all registers right
                    for (j = 0; j < NUM_STREAMS; j = j + 1) begin
                        weight_regs[j] <= {1'b0, weight_regs[j][WEIGHT_WIDTH-1:1]};
                    end

                    if (bit_idx == 0) begin
                        state_r <= NEXT_WORD;
                    end else begin
                        bit_idx <= bit_idx - 1;
                    end
                end

                // =============================================================
                STREAM_BYTES: begin
                    // Output 1 byte at a time (one stream per cycle)
                    stream_valid      <= 1'b1;
                    active_stream_idx <= stream_idx;

                    if (stream_idx == NUM_STREAMS - 1) begin
                        state_r <= NEXT_WORD;
                    end else begin
                        stream_idx <= stream_idx + 1;
                    end
                end

                // =============================================================
                NEXT_WORD: begin
                    if (word_count == 1) begin
                        // Last word complete
                        last_weight <= 1'b1;
                        state_r     <= IDLE;
                    end else begin
                        // Fetch next word
                        word_count <= word_count - 1;
                        uram_addr  <= uram_addr + 1;
                        state_r    <= FETCH_WORD;
                    end
                end

                default: state_r <= IDLE;
            endcase
        end
    end

endmodule


/**
 * SYNAPSE WEIGHT MEMORY
 * =====================
 *
 * URAM-based weight storage for bit-serial neurons.
 * Organized for efficient streaming to neuron banks.
 *
 * Memory Layout:
 *   Each URAM block: 4K x 72-bit words
 *   = 4096 * 9 = 36,864 weights per URAM
 *   = 4096 * 72 = 294,912 bits = 36 KB
 *
 * VU7P has 320 URAMs = 11.5 MB of weight storage
 *   = ~11.8 million 8-bit synapses
 */

module synapse_weight_memory #(
    parameter ADDR_WIDTH = 12,     // 4K words
    parameter DATA_WIDTH = 72      // 9 weights packed
)(
    input  wire                    clk,

    // Port A: Write (from host/learning)
    input  wire                    wr_en,
    input  wire [ADDR_WIDTH-1:0]   wr_addr,
    input  wire [DATA_WIDTH-1:0]   wr_data,

    // Port B: Read (to streamer)
    input  wire [ADDR_WIDTH-1:0]   rd_addr,
    output reg  [DATA_WIDTH-1:0]   rd_data,
    output reg                     rd_valid
);

    // URAM inference hint for Vivado
    (* ram_style = "ultra" *)
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    // Optional initialization (zeros or from file)
    initial begin
        integer i;
        for (i = 0; i < (1<<ADDR_WIDTH); i = i + 1) begin
            mem[i] = 72'd0;
        end
    end

    // Port A: Write
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // Port B: Read (1-cycle latency)
    always @(posedge clk) begin
        rd_data  <= mem[rd_addr];
        rd_valid <= 1'b1;  // Always valid after 1 cycle
    end

endmodule


/**
 * STATE MEMORY (BRAM)
 * ===================
 *
 * BRAM-based membrane potential storage.
 * Each neuron has a 16-bit state value.
 *
 * Memory Layout:
 *   Each BRAM18: 1K x 18-bit (16-bit state + 2 bits metadata)
 *   Each BRAM36: 2K x 18-bit
 *
 * VU7P has 1440 BRAM36 = 2,880K states = 2.9M neurons
 */

module neuron_state_memory #(
    parameter ADDR_WIDTH = 11,     // 2K neurons
    parameter DATA_WIDTH = 18      // 16-bit state + metadata
)(
    input  wire                    clk,

    // Port A: Read/Write (from neuron bank)
    input  wire                    a_we,
    input  wire [ADDR_WIDTH-1:0]   a_addr,
    input  wire [DATA_WIDTH-1:0]   a_wdata,
    output reg  [DATA_WIDTH-1:0]   a_rdata,

    // Port B: Read only (for monitoring/dreaming)
    input  wire [ADDR_WIDTH-1:0]   b_addr,
    output reg  [DATA_WIDTH-1:0]   b_rdata
);

    // BRAM inference
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    // Initialize to resting potential
    initial begin
        integer i;
        for (i = 0; i < (1<<ADDR_WIDTH); i = i + 1) begin
            mem[i] = 18'h2000;  // V_REST = 0.125 in Q8.8
        end
    end

    // Port A: Read/Write
    always @(posedge clk) begin
        if (a_we) begin
            mem[a_addr] <= a_wdata;
        end
        a_rdata <= mem[a_addr];
    end

    // Port B: Read only
    always @(posedge clk) begin
        b_rdata <= mem[b_addr];
    end

endmodule
