/**
 * NEURO-SYMBOLIC BRIDGE - The Bicameral Interface
 * ================================================
 *
 * Bio-Affective Neuromorphic Operating System
 * Bidirectional translation between Symbolic (LLM) and Spiking (SNN) worlds
 *
 * DESCENDING PATHWAY (Cortex -> Brainstem):
 *   LLM Hidden States (PCIe) -> Sparse Spikes (NoC)
 *   "Thought becomes reflex"
 *
 * ASCENDING PATHWAY (Brainstem -> Cortex):
 *   Aggregate Spike State (NoC) -> Somatic Vectors (PCIe)
 *   "Sensation becomes perception"
 *
 * This module implements:
 * 1. CSR Projection: Dense embedding -> Sparse spike pattern
 * 2. Spike Packetizer: Spike bits -> NoC flits to fabric tiles
 * 3. Spike Aggregator: NoC flits -> Dense somatic state vector
 * 4. Rate Coding: Continuous embedding -> Spike frequency modulation
 *
 * Memory Map (AXI-Lite Config):
 *   0x0000: Control register (start/reset/mode)
 *   0x0004: Status register (busy/done/overflow)
 *   0x0008: Descending threshold
 *   0x000C: Ascending gain
 *   0x0100: Projection matrix row_ptr base
 *   0x1000: Projection matrix col_idx base
 *   0x8000: Projection matrix values base
 */

module neuro_symbolic_bridge
    import kf_pkg::*;
#(
    parameter int DIM_EMBED   = 4096,  // LLM hidden dimension
    parameter int DIM_SPIKE   = 1024,  // Spike neurons (matches projection output)
    parameter int MESH_WIDTH  = 4,     // Kitten Fabric mesh width
    parameter int MESH_HEIGHT = 4      // Kitten Fabric mesh height
)(
    input  logic        clk,
    input  logic        rst_n,

    // =========================================================================
    // AXI Stream: Descending (LLM -> SNN)
    // Hidden state vectors from Threadripper via PCIe DMA
    // =========================================================================
    input  logic [511:0]  s_axis_desc_tdata,
    input  logic          s_axis_desc_tvalid,
    output logic          s_axis_desc_tready,
    input  logic          s_axis_desc_tlast,

    // =========================================================================
    // AXI Stream: Ascending (SNN -> LLM)
    // Somatic state vectors back to Threadripper
    // =========================================================================
    output logic [511:0]  m_axis_asc_tdata,
    output logic          m_axis_asc_tvalid,
    input  logic          m_axis_asc_tready,
    output logic          m_axis_asc_tlast,

    // =========================================================================
    // NoC Interface: To Kitten Fabric Edge Tiles
    // =========================================================================
    // Inject port (descending spikes)
    output logic          noc_inject_valid,
    input  logic          noc_inject_ready,
    output spike_flit_t   noc_inject_flit,

    // Eject port (ascending spikes)
    input  logic          noc_eject_valid,
    output logic          noc_eject_ready,
    input  spike_flit_t   noc_eject_flit,

    // =========================================================================
    // AXI-Lite Configuration Interface
    // =========================================================================
    input  logic [15:0]   cfg_addr,
    input  logic          cfg_we,
    input  logic          cfg_re,
    input  logic [31:0]   cfg_wdata,
    output logic [31:0]   cfg_rdata,
    output logic          cfg_ready
);

    // =========================================================================
    // CONFIGURATION REGISTERS
    // =========================================================================

    logic [31:0] ctrl_reg;        // [0]=enable, [1]=desc_start, [2]=asc_start
    logic [31:0] status_reg;      // [0]=desc_busy, [1]=asc_busy, [2]=overflow
    logic [31:0] desc_thresh;     // Spike threshold for descending
    logic [31:0] asc_gain;        // Gain for ascending aggregation

    // Control bits
    logic enable;
    logic desc_start;
    logic asc_start;
    assign enable = ctrl_reg[0];
    assign desc_start = ctrl_reg[1];
    assign asc_start = ctrl_reg[2];

    // =========================================================================
    // DESCENDING PATHWAY: LLM Hidden State -> Spike Injection
    // =========================================================================

    // CSR Projection instance
    logic [DIM_SPIKE-1:0] spike_vector;
    logic projection_done;

    // Config interface for projection matrix
    logic proj_cfg_we;
    logic [1:0] proj_cfg_sel;
    logic [19:0] proj_cfg_addr;
    logic [31:0] proj_cfg_wdata;

    csr_projection #(
        .INPUT_DIM  (DIM_EMBED),
        .OUTPUT_DIM (DIM_SPIKE)
    ) u_projection (
        .clk            (clk),
        .rst_n          (rst_n),

        .s_axis_tdata   (s_axis_desc_tdata),
        .s_axis_tvalid  (s_axis_desc_tvalid && enable),
        .s_axis_tready  (s_axis_desc_tready),
        .s_axis_tlast   (s_axis_desc_tlast),

        .spikes_out     (spike_vector),
        .done           (projection_done),

        .cfg_we         (proj_cfg_we),
        .cfg_sel        (proj_cfg_sel),
        .cfg_addr       (proj_cfg_addr),
        .cfg_wdata      (proj_cfg_wdata)
    );

    // =========================================================================
    // SPIKE PACKETIZER: Convert spike bits to NoC flits
    // =========================================================================

    // Distribute spikes across mesh tiles (round-robin by neuron ID)
    logic [$clog2(DIM_SPIKE)-1:0] inject_idx;
    logic inject_active;

    // Map neuron index to tile coordinates
    // Simple distribution: neuron N goes to tile (N/256 % MESH_X, N/256 / MESH_X)
    wire [7:0] tile_x = (inject_idx / KF_NEURONS_PER_TILE) % MESH_WIDTH;
    wire [7:0] tile_y = (inject_idx / KF_NEURONS_PER_TILE) / MESH_WIDTH;
    wire [7:0] neuron_in_tile = inject_idx % KF_NEURONS_PER_TILE;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            inject_idx <= 0;
            inject_active <= 0;
            noc_inject_valid <= 0;
        end else if (projection_done && !inject_active) begin
            // Start injection sequence
            inject_active <= 1;
            inject_idx <= 0;
        end else if (inject_active) begin
            if (inject_idx < DIM_SPIKE) begin
                if (spike_vector[inject_idx]) begin
                    // This neuron fired - send flit
                    noc_inject_valid <= 1;
                    noc_inject_flit.dest_x <= tile_x;
                    noc_inject_flit.dest_y <= tile_y;
                    noc_inject_flit.neuron_id <= neuron_in_tile[KF_NEURON_ID_BITS-1:0];
                    noc_inject_flit.payload <= 8'hFF;  // Max stimulus

                    if (noc_inject_ready) begin
                        inject_idx <= inject_idx + 1;
                        noc_inject_valid <= 0;
                    end
                end else begin
                    // No spike, skip
                    inject_idx <= inject_idx + 1;
                end
            end else begin
                // Done injecting
                inject_active <= 0;
                noc_inject_valid <= 0;
            end
        end else begin
            noc_inject_valid <= 0;
        end
    end

    // =========================================================================
    // ASCENDING PATHWAY: Spike Aggregation -> Somatic Vector
    // =========================================================================

    // Spike counters per region (aggregated from ejected flits)
    localparam int NUM_REGIONS = 16;  // 4x4 mesh = 16 regions
    logic [15:0] spike_count [NUM_REGIONS];

    // Aggregate ejected spikes
    assign noc_eject_ready = 1'b1;  // Always accept

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_REGIONS; i++) begin
                spike_count[i] <= 0;
            end
        end else if (asc_start) begin
            // Clear counters on start
            for (int i = 0; i < NUM_REGIONS; i++) begin
                spike_count[i] <= 0;
            end
        end else if (noc_eject_valid && noc_eject_ready) begin
            // Increment counter for the source region
            automatic int region = noc_eject_flit.dest_y * MESH_WIDTH + noc_eject_flit.dest_x;
            if (region < NUM_REGIONS) begin
                spike_count[region] <= spike_count[region] + 1;
            end
        end
    end

    // =========================================================================
    // SOMATIC VECTOR OUTPUT
    // =========================================================================

    // Pack spike counts into output stream
    // Format: 16 x 32-bit counts = 512 bits per beat
    logic asc_busy;
    logic [1:0] asc_beat;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axis_asc_tvalid <= 0;
            m_axis_asc_tlast <= 0;
            asc_busy <= 0;
            asc_beat <= 0;
        end else if (asc_start && !asc_busy) begin
            asc_busy <= 1;
            asc_beat <= 0;
        end else if (asc_busy) begin
            // Output somatic vector (16 regions, 2 beats of 8 regions each)
            m_axis_asc_tvalid <= 1;

            case (asc_beat)
                2'd0: begin
                    for (int i = 0; i < 8; i++) begin
                        m_axis_asc_tdata[i*32 +: 32] <= {16'b0, spike_count[i]};
                    end
                    m_axis_asc_tlast <= 0;
                end
                2'd1: begin
                    for (int i = 0; i < 8; i++) begin
                        m_axis_asc_tdata[i*32 +: 32] <= {16'b0, spike_count[8+i]};
                    end
                    m_axis_asc_tlast <= 1;
                end
                default: ;
            endcase

            if (m_axis_asc_tready) begin
                if (asc_beat == 1) begin
                    asc_busy <= 0;
                    m_axis_asc_tvalid <= 0;
                    m_axis_asc_tlast <= 0;
                end else begin
                    asc_beat <= asc_beat + 1;
                end
            end
        end else begin
            m_axis_asc_tvalid <= 0;
        end
    end

    // =========================================================================
    // CONFIGURATION INTERFACE
    // =========================================================================

    assign status_reg = {29'b0, 1'b0, asc_busy, inject_active};
    assign cfg_ready = 1'b1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ctrl_reg <= 32'h0;
            desc_thresh <= 32'h0000_1000;  // Default threshold
            asc_gain <= 32'h0000_0100;     // Default gain
            proj_cfg_we <= 0;
        end else begin
            proj_cfg_we <= 0;  // Default

            if (cfg_we) begin
                case (cfg_addr[15:12])
                    4'h0: begin  // Control registers
                        case (cfg_addr[7:0])
                            8'h00: ctrl_reg <= cfg_wdata;
                            8'h08: desc_thresh <= cfg_wdata;
                            8'h0C: asc_gain <= cfg_wdata;
                            default: ;
                        endcase
                    end
                    4'h1: begin  // Projection row_ptr
                        proj_cfg_we <= 1;
                        proj_cfg_sel <= 2'b00;
                        proj_cfg_addr <= cfg_addr[11:0];
                        proj_cfg_wdata <= cfg_wdata;
                    end
                    4'h2, 4'h3, 4'h4, 4'h5, 4'h6, 4'h7: begin  // Projection col_idx
                        proj_cfg_we <= 1;
                        proj_cfg_sel <= 2'b01;
                        proj_cfg_addr <= {cfg_addr[14:12], cfg_addr[11:0]};
                        proj_cfg_wdata <= cfg_wdata;
                    end
                    4'h8, 4'h9, 4'hA, 4'hB, 4'hC, 4'hD, 4'hE, 4'hF: begin  // Projection values
                        proj_cfg_we <= 1;
                        proj_cfg_sel <= 2'b10;
                        proj_cfg_addr <= {cfg_addr[14:12], cfg_addr[11:0]};
                        proj_cfg_wdata <= cfg_wdata;
                    end
                endcase
            end

            // Auto-clear start bits
            if (ctrl_reg[1]) ctrl_reg[1] <= 0;
            if (ctrl_reg[2]) ctrl_reg[2] <= 0;
        end
    end

    // Read path
    always_comb begin
        cfg_rdata = 32'h0;
        if (cfg_re) begin
            case (cfg_addr[7:0])
                8'h00: cfg_rdata = ctrl_reg;
                8'h04: cfg_rdata = status_reg;
                8'h08: cfg_rdata = desc_thresh;
                8'h0C: cfg_rdata = asc_gain;
                default: cfg_rdata = 32'hDEAD_BEEF;
            endcase
        end
    end

endmodule : neuro_symbolic_bridge
