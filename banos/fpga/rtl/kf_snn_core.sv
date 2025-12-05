/**
 * Kitten Fabric (FK33) - Spiking Neural Network Core
 * ===================================================
 *
 * Event-driven LIF (Leaky Integrate-and-Fire) neuron array.
 *
 * Architecture:
 * - N_NEURONS neurons with membrane potential in BRAM
 * - N_SYNAPSES synapses stored in synapse_ram
 * - Index table maps presynaptic neuron → synapse range
 * - On incoming spike: walk synapse range, accumulate to postsynaptic neurons
 * - Threshold crossing → emit output spike
 *
 * Event-driven property:
 * - Only active when spikes arrive
 * - Complexity ~ number of active synapses, not N²
 * - Zero dynamic power when quiescent
 *
 * Interface:
 * - spike_in_*: incoming spike from NoC router
 * - spike_out_*: outgoing spike to NoC router
 * - cfg_*: configuration interface for loading weights/indices
 */

module kf_snn_core
    import kf_pkg::*;
#(
    parameter int N_NEURONS  = KF_NEURONS_PER_TILE,
    parameter int N_SYNAPSES = KF_SYNAPSES_PER_TILE
)(
    input  logic                          clk,
    input  logic                          rst_n,

    // =========================================================================
    // Spike Input (from NoC router)
    // =========================================================================
    input  logic                          spike_in_valid,
    output logic                          spike_in_ready,
    input  logic [KF_NEURON_ID_BITS-1:0]  spike_in_pre_id,   // Presynaptic neuron
    input  logic [7:0]                    spike_in_payload,  // Context/timestamp

    // =========================================================================
    // Spike Output (to NoC router)
    // =========================================================================
    output logic                          spike_out_valid,
    input  logic                          spike_out_ready,
    output logic [KF_NEURON_ID_BITS-1:0]  spike_out_post_id, // Postsynaptic neuron
    output logic [7:0]                    spike_out_payload,

    // =========================================================================
    // Configuration Interface (for loading weights)
    // =========================================================================
    input  logic                          cfg_we,
    input  logic [1:0]                    cfg_sel,           // 0=synapse, 1=index, 2=vmem
    input  logic [KF_SYNAPSE_ID_BITS-1:0] cfg_addr,
    input  logic [31:0]                   cfg_wdata,

    // =========================================================================
    // Debug/Status
    // =========================================================================
    output logic [7:0]                    active_neuron_count,
    output logic                          core_busy
);

    // =========================================================================
    // MEMORY ARRAYS
    // =========================================================================

    // Membrane potentials (BRAM)
    logic signed [V_WIDTH-1:0] vmem [N_NEURONS];

    // Synapse table (BRAM)
    synapse_t syn_ram [N_SYNAPSES];

    // Index table: synapse range per presynaptic neuron (BRAM)
    syn_index_t idx_ram [N_NEURONS];

    // Spike pending FIFO (small, for multiple firings per walk)
    localparam int SPIKE_FIFO_DEPTH = 8;
    logic [KF_NEURON_ID_BITS-1:0] spike_fifo [SPIKE_FIFO_DEPTH];
    logic [7:0]                   spike_fifo_payload [SPIKE_FIFO_DEPTH];
    logic [$clog2(SPIKE_FIFO_DEPTH):0] fifo_wr_ptr, fifo_rd_ptr;
    logic fifo_empty, fifo_full;

    assign fifo_empty = (fifo_wr_ptr == fifo_rd_ptr);
    assign fifo_full  = (fifo_wr_ptr[$clog2(SPIKE_FIFO_DEPTH)] !=
                         fifo_rd_ptr[$clog2(SPIKE_FIFO_DEPTH)]) &&
                        (fifo_wr_ptr[$clog2(SPIKE_FIFO_DEPTH)-1:0] ==
                         fifo_rd_ptr[$clog2(SPIKE_FIFO_DEPTH)-1:0]);

    // =========================================================================
    // FSM STATE
    // =========================================================================

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_LOAD_RANGE,
        ST_WALK_SYNAPSES,
        ST_CHECK_FIRE,
        ST_EMIT_SPIKE
    } state_t;

    state_t state, state_next;

    // Walk state
    logic [KF_SYNAPSE_ID_BITS-1:0] walk_idx;
    logic [KF_SYNAPSE_ID_BITS-1:0] walk_end;
    logic [KF_NEURON_ID_BITS-1:0]  cur_pre_id;
    logic [7:0]                    cur_payload;

    // Pipeline registers for synapse read
    synapse_t                      syn_rd;
    logic                          syn_rd_valid;
    logic [KF_NEURON_ID_BITS-1:0]  pending_post_id;

    // Fire detection
    logic                          fire_detected;
    logic [KF_NEURON_ID_BITS-1:0]  fire_neuron_id;

    // =========================================================================
    // CONFIGURATION WRITE
    // =========================================================================

    always_ff @(posedge clk) begin
        if (cfg_we) begin
            case (cfg_sel)
                2'b00: syn_ram[cfg_addr] <= synapse_t'(cfg_wdata);
                2'b01: idx_ram[cfg_addr[KF_NEURON_ID_BITS-1:0]] <= syn_index_t'(cfg_wdata[SYN_INDEX_WIDTH-1:0]);
                2'b10: vmem[cfg_addr[KF_NEURON_ID_BITS-1:0]] <= cfg_wdata[V_WIDTH-1:0];
                default: ;
            endcase
        end
    end

    // =========================================================================
    // MAIN FSM
    // =========================================================================

    assign spike_in_ready = (state == ST_IDLE) && !fifo_full;
    assign core_busy = (state != ST_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            spike_out_valid <= 1'b0;
            syn_rd_valid <= 1'b0;
            fifo_wr_ptr <= '0;
            fifo_rd_ptr <= '0;
            active_neuron_count <= '0;
        end else begin
            // Default: clear one-cycle signals
            syn_rd_valid <= 1'b0;
            fire_detected <= 1'b0;

            case (state)

                // ---------------------------------------------------------
                ST_IDLE: begin
                    spike_out_valid <= 1'b0;

                    if (spike_in_valid && spike_in_ready) begin
                        // Latch incoming spike
                        cur_pre_id  <= spike_in_pre_id;
                        cur_payload <= spike_in_payload;
                        state <= ST_LOAD_RANGE;
                    end else if (!fifo_empty && spike_out_ready) begin
                        // Emit pending spike from FIFO
                        spike_out_valid   <= 1'b1;
                        spike_out_post_id <= spike_fifo[fifo_rd_ptr[$clog2(SPIKE_FIFO_DEPTH)-1:0]];
                        spike_out_payload <= spike_fifo_payload[fifo_rd_ptr[$clog2(SPIKE_FIFO_DEPTH)-1:0]];
                        fifo_rd_ptr <= fifo_rd_ptr + 1;
                    end
                end

                // ---------------------------------------------------------
                ST_LOAD_RANGE: begin
                    // Read index table for this presynaptic neuron
                    walk_idx <= idx_ram[cur_pre_id].start_idx;
                    walk_end <= idx_ram[cur_pre_id].end_idx;
                    state <= ST_WALK_SYNAPSES;
                end

                // ---------------------------------------------------------
                ST_WALK_SYNAPSES: begin
                    // Check if range is valid (end >= start)
                    if (walk_idx > walk_end) begin
                        // No synapses for this presyn, done
                        state <= ST_IDLE;
                    end else begin
                        // Read synapse from BRAM
                        syn_rd <= syn_ram[walk_idx];
                        syn_rd_valid <= 1'b1;
                        state <= ST_CHECK_FIRE;
                    end
                end

                // ---------------------------------------------------------
                ST_CHECK_FIRE: begin
                    if (syn_rd_valid) begin
                        // Accumulate weight to postsynaptic neuron
                        logic signed [V_WIDTH-1:0] new_vmem;
                        logic signed [V_WIDTH-1:0] weight_ext;

                        weight_ext = weight_to_vmem(syn_rd.weight);
                        new_vmem = vmem_add_sat(vmem[syn_rd.post_id], weight_ext);
                        vmem[syn_rd.post_id] <= new_vmem;

                        // Check threshold
                        if (new_vmem >= V_THRESH) begin
                            // Fire! Reset membrane and queue spike
                            vmem[syn_rd.post_id] <= V_RESET;
                            fire_detected <= 1'b1;
                            fire_neuron_id <= syn_rd.post_id;

                            // Push to FIFO if not full
                            if (!fifo_full) begin
                                spike_fifo[fifo_wr_ptr[$clog2(SPIKE_FIFO_DEPTH)-1:0]] <= syn_rd.post_id;
                                spike_fifo_payload[fifo_wr_ptr[$clog2(SPIKE_FIFO_DEPTH)-1:0]] <= cur_payload;
                                fifo_wr_ptr <= fifo_wr_ptr + 1;
                                active_neuron_count <= active_neuron_count + 1;
                            end
                        end
                    end

                    // Advance to next synapse or finish
                    if (walk_idx == walk_end) begin
                        state <= ST_IDLE;
                    end else begin
                        walk_idx <= walk_idx + 1;
                        state <= ST_WALK_SYNAPSES;
                    end
                end

                default: state <= ST_IDLE;

            endcase
        end
    end

    // =========================================================================
    // LEAK PROCESS (optional: run when idle for biological realism)
    // =========================================================================

    // In a full implementation, we'd have a separate slow clock domain
    // or a counter that triggers leak on all neurons periodically.
    // For v1, we omit leak to keep it purely event-driven.

    // =========================================================================
    // MEMBRANE INITIALIZATION
    // =========================================================================

    // Initialize membranes to zero on reset
    integer i;
    initial begin
        for (i = 0; i < N_NEURONS; i++) begin
            vmem[i] = '0;
        end
    end

endmodule : kf_snn_core
