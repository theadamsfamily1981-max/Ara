/**
 * Kitten Fabric (FK33) - 2x2 Mesh Testbench
 * ==========================================
 *
 * Basic simulation testbench for verifying:
 * 1. Spike injection and routing
 * 2. SNN core synapse processing
 * 3. Multi-hop routing across mesh
 *
 * Test sequence:
 * 1. Load synapses into tile (0,0)
 * 2. Inject spike targeting (0,0)
 * 3. Verify spike processing triggers output
 * 4. Inject spike targeting (1,1) - tests routing
 */

`timescale 1ns / 1ps

module tb_kf_mesh_2x2;
    import kf_pkg::*;

    // =========================================================================
    // CLOCK AND RESET
    // =========================================================================

    logic clk;
    logic rst_n;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100 MHz
    end

    initial begin
        rst_n = 0;
        repeat (10) @(posedge clk);
        rst_n = 1;
    end

    // =========================================================================
    // DUT SIGNALS
    // =========================================================================

    logic        ext_spike_in_valid;
    logic        ext_spike_in_ready;
    spike_flit_t ext_spike_in_flit;

    logic        ext_spike_out_valid;
    logic        ext_spike_out_ready;
    spike_flit_t ext_spike_out_flit;

    logic                          cfg_we;
    logic [1:0]                    cfg_tile_sel;
    logic [1:0]                    cfg_sel;
    logic [KF_SYNAPSE_ID_BITS-1:0] cfg_addr;
    logic [31:0]                   cfg_wdata;

    logic [7:0]  tile_active_count [4];
    logic [3:0]  tile_busy;

    // =========================================================================
    // DUT INSTANCE
    // =========================================================================

    kf_mesh_2x2 dut (
        .clk                (clk),
        .rst_n              (rst_n),

        .ext_spike_in_valid (ext_spike_in_valid),
        .ext_spike_in_ready (ext_spike_in_ready),
        .ext_spike_in_flit  (ext_spike_in_flit),

        .ext_spike_out_valid(ext_spike_out_valid),
        .ext_spike_out_ready(ext_spike_out_ready),
        .ext_spike_out_flit (ext_spike_out_flit),

        .cfg_we             (cfg_we),
        .cfg_tile_sel       (cfg_tile_sel),
        .cfg_sel            (cfg_sel),
        .cfg_addr           (cfg_addr),
        .cfg_wdata          (cfg_wdata),

        .tile_active_count  (tile_active_count),
        .tile_busy          (tile_busy)
    );

    // =========================================================================
    // HELPER TASKS
    // =========================================================================

    // Write to synapse RAM
    task write_synapse(
        input [1:0] tile,
        input [KF_SYNAPSE_ID_BITS-1:0] addr,
        input [KF_NEURON_ID_BITS-1:0] pre_id,
        input [KF_NEURON_ID_BITS-1:0] post_id,
        input signed [W_WIDTH-1:0] weight,
        input [7:0] flags
    );
        @(posedge clk);
        cfg_we <= 1;
        cfg_tile_sel <= tile;
        cfg_sel <= 2'b00;  // synapse RAM
        cfg_addr <= addr;
        cfg_wdata <= {pre_id, post_id, weight, flags};
        @(posedge clk);
        cfg_we <= 0;
    endtask

    // Write to index RAM
    task write_index(
        input [1:0] tile,
        input [KF_NEURON_ID_BITS-1:0] neuron,
        input [KF_SYNAPSE_ID_BITS-1:0] start_idx,
        input [KF_SYNAPSE_ID_BITS-1:0] end_idx
    );
        @(posedge clk);
        cfg_we <= 1;
        cfg_tile_sel <= tile;
        cfg_sel <= 2'b01;  // index RAM
        cfg_addr <= {{(KF_SYNAPSE_ID_BITS-KF_NEURON_ID_BITS){1'b0}}, neuron};
        cfg_wdata <= {8'b0, start_idx, end_idx};
        @(posedge clk);
        cfg_we <= 0;
    endtask

    // Inject a spike
    task inject_spike(
        input [KF_COORD_BITS-1:0] dest_x,
        input [KF_COORD_BITS-1:0] dest_y,
        input [KF_NEURON_ID_BITS-1:0] neuron_id,
        input [7:0] payload
    );
        @(posedge clk);
        ext_spike_in_valid <= 1;
        ext_spike_in_flit.dest_x <= dest_x;
        ext_spike_in_flit.dest_y <= dest_y;
        ext_spike_in_flit.neuron_id <= neuron_id;
        ext_spike_in_flit.payload <= payload;

        // Wait for ready
        wait (ext_spike_in_ready);
        @(posedge clk);
        ext_spike_in_valid <= 0;
    endtask

    // =========================================================================
    // TEST SEQUENCE
    // =========================================================================

    int spike_count;

    initial begin
        // Initialize
        ext_spike_in_valid = 0;
        ext_spike_in_flit = '0;
        ext_spike_out_ready = 1;
        cfg_we = 0;
        cfg_tile_sel = 0;
        cfg_sel = 0;
        cfg_addr = 0;
        cfg_wdata = 0;
        spike_count = 0;

        // Wait for reset
        @(posedge rst_n);
        repeat (5) @(posedge clk);

        $display("=== Kitten Fabric 2x2 Mesh Test ===");
        $display("Time: %0t - Starting configuration", $time);

        // -----------------------------------------------------------------
        // TEST 1: Configure tile (0,0) with simple synapse chain
        // -----------------------------------------------------------------

        // Create synapse: neuron 0 -> neuron 1 with strong positive weight
        // Weight = +0.5 in Q1.7 = 64 = 8'h40
        write_synapse(
            .tile(2'b00),
            .addr(0),
            .pre_id(8'd0),
            .post_id(8'd1),
            .weight(8'sh40),  // +0.5
            .flags(8'h02)     // Excitatory
        );

        // Multiple synapses to reach threshold (V_THRESH = 16.0)
        // Need 32 * 0.5 = 16.0
        for (int i = 1; i < 32; i++) begin
            write_synapse(
                .tile(2'b00),
                .addr(i),
                .pre_id(8'd0),
                .post_id(8'd1),
                .weight(8'sh40),
                .flags(8'h02)
            );
        end

        // Set index for neuron 0: synapses 0-31
        write_index(
            .tile(2'b00),
            .neuron(8'd0),
            .start_idx(12'd0),
            .end_idx(12'd31)
        );

        $display("Time: %0t - Configuration complete", $time);
        repeat (5) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 2: Inject spike to tile (0,0), neuron 0
        // -----------------------------------------------------------------

        $display("Time: %0t - Injecting spike to (0,0) neuron 0", $time);
        inject_spike(
            .dest_x(8'd0),
            .dest_y(8'd0),
            .neuron_id(8'd0),
            .payload(8'h00)  // Local destination
        );

        // Wait for processing
        repeat (100) @(posedge clk);

        // -----------------------------------------------------------------
        // TEST 3: Inject spike to tile (1,1) - test routing
        // -----------------------------------------------------------------

        $display("Time: %0t - Injecting spike to (1,1) neuron 5", $time);
        inject_spike(
            .dest_x(8'd1),
            .dest_y(8'd1),
            .neuron_id(8'd5),
            .payload(8'h11)  // Will route to (1,1)
        );

        // Wait for routing
        repeat (50) @(posedge clk);

        // -----------------------------------------------------------------
        // Done
        // -----------------------------------------------------------------

        $display("=== Test Complete ===");
        $display("Output spikes captured: %0d", spike_count);
        $display("Tile 0 active: %0d", tile_active_count[0]);
        $display("Tile 1 active: %0d", tile_active_count[1]);
        $display("Tile 2 active: %0d", tile_active_count[2]);
        $display("Tile 3 active: %0d", tile_active_count[3]);

        #100;
        $finish;
    end

    // =========================================================================
    // OUTPUT SPIKE MONITOR
    // =========================================================================

    always @(posedge clk) begin
        if (ext_spike_out_valid && ext_spike_out_ready) begin
            spike_count <= spike_count + 1;
            $display("Time: %0t - OUTPUT SPIKE: dest=(%0d,%0d) neuron=%0d payload=%h",
                $time,
                ext_spike_out_flit.dest_x,
                ext_spike_out_flit.dest_y,
                ext_spike_out_flit.neuron_id,
                ext_spike_out_flit.payload);
        end
    end

    // =========================================================================
    // TIMEOUT WATCHDOG
    // =========================================================================

    initial begin
        #10000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule : tb_kf_mesh_2x2
