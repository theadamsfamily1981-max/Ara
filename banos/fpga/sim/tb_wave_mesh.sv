/**
 * ARA-WAVE Field Engine - Wave Mesh Testbench
 * ============================================
 *
 * Verify Schrödinger equation evolution:
 * 1. Initialize Gaussian wavepacket
 * 2. Set potential barrier
 * 3. Evolve and observe interference
 *
 * Test scenarios:
 * - Free particle spreading
 * - Barrier reflection/tunneling
 * - Harmonic oscillator (if V = x²)
 */

`timescale 1ns / 1ps

module tb_wave_mesh;
    import wave_pkg::*;

    // =========================================================================
    // PARAMETERS
    // =========================================================================

    localparam int MESH_X = 8;
    localparam int MESH_Y = 8;
    localparam int SIM_FRAMES = 100;

    // =========================================================================
    // SIGNALS
    // =========================================================================

    logic clk;
    logic rst_n;
    logic enable;
    logic [DT_WIDTH-1:0] dt;

    logic pot_we;
    logic [$clog2(MESH_X)-1:0] pot_x;
    logic [$clog2(MESH_Y)-1:0] pot_y;
    logic signed [V_WIDTH-1:0] pot_data;

    logic psi_we;
    logic [$clog2(MESH_X)-1:0] psi_x;
    logic [$clog2(MESH_Y)-1:0] psi_y;
    complex_t psi_data;

    logic [$clog2(MESH_X)-1:0] read_x;
    logic [$clog2(MESH_Y)-1:0] read_y;
    complex_t read_psi;
    logic [PSI_WIDTH-1:0] read_magnitude;

    logic busy;
    logic frame_done;

    // =========================================================================
    // DUT
    // =========================================================================

    wave_mesh #(
        .MESH_X   (MESH_X),
        .MESH_Y   (MESH_Y),
        .TOROIDAL (1'b0)  // Absorbing boundaries
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .enable         (enable),
        .dt             (dt),
        .pot_we         (pot_we),
        .pot_x          (pot_x),
        .pot_y          (pot_y),
        .pot_data       (pot_data),
        .psi_we         (psi_we),
        .psi_x          (psi_x),
        .psi_y          (psi_y),
        .psi_data       (psi_data),
        .read_x         (read_x),
        .read_y         (read_y),
        .read_psi       (read_psi),
        .read_magnitude (read_magnitude),
        .busy           (busy),
        .frame_done     (frame_done)
    );

    // =========================================================================
    // CLOCK
    // =========================================================================

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100 MHz
    end

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    // Gaussian wavepacket in Q8.8
    function automatic complex_t gaussian(
        input int x, input int y,
        input int cx, input int cy,  // Center
        input real sigma
    );
        real dx, dy, r2, amp;
        complex_t result;

        dx = real'(x - cx);
        dy = real'(y - cy);
        r2 = dx*dx + dy*dy;
        amp = $exp(-r2 / (2.0 * sigma * sigma));

        // Scale to Q8.8: max amplitude 64.0
        result.re = $rtoi(amp * 64.0 * 256.0);  // Q8.8
        result.im = 16'sd0;  // Real Gaussian, no imaginary part

        return result;
    endfunction

    // =========================================================================
    // TEST SEQUENCE
    // =========================================================================

    initial begin
        // Initialize signals
        rst_n = 0;
        enable = 0;
        dt = DEFAULT_DT;  // 0.01 in Q0.16
        pot_we = 0;
        psi_we = 0;
        read_x = 0;
        read_y = 0;

        // Reset
        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        $display("=== ARA-WAVE Field Engine Test ===");
        $display("Mesh size: %0d x %0d", MESH_X, MESH_Y);

        // -----------------------------------------------------------------
        // Initialize potential field (flat, V=0 everywhere)
        // -----------------------------------------------------------------
        $display("Initializing potential field...");
        for (int x = 0; x < MESH_X; x++) begin
            for (int y = 0; y < MESH_Y; y++) begin
                @(posedge clk);
                pot_we <= 1;
                pot_x <= x;
                pot_y <= y;

                // Add a small barrier in the middle
                if (x == MESH_X/2 && y >= MESH_Y/4 && y < 3*MESH_Y/4) begin
                    pot_data <= 16'sd1024;  // Barrier: V = 0.25 in Q4.12
                end else begin
                    pot_data <= 16'sd0;  // Free space
                end
            end
        end
        @(posedge clk);
        pot_we <= 0;

        // -----------------------------------------------------------------
        // Initialize wavefunction (Gaussian wavepacket)
        // -----------------------------------------------------------------
        $display("Initializing wavefunction (Gaussian)...");
        for (int x = 0; x < MESH_X; x++) begin
            for (int y = 0; y < MESH_Y; y++) begin
                @(posedge clk);
                psi_we <= 1;
                psi_x <= x;
                psi_y <= y;
                psi_data <= gaussian(x, y, MESH_X/4, MESH_Y/2, 1.5);
            end
        end
        @(posedge clk);
        psi_we <= 0;

        // -----------------------------------------------------------------
        // Evolve the field
        // -----------------------------------------------------------------
        $display("Starting evolution...");

        for (int frame = 0; frame < SIM_FRAMES; frame++) begin
            // Trigger one evolution step
            @(posedge clk);
            enable <= 1;
            @(posedge clk);
            enable <= 0;

            // Wait for completion
            wait (frame_done);
            @(posedge clk);

            // Print magnitude at center and edges
            if (frame % 10 == 0) begin
                // Sample a few points
                read_x <= MESH_X/4;
                read_y <= MESH_Y/2;
                @(posedge clk);
                $display("Frame %3d: Psi[%d,%d] = (%d, %d) |Psi|² = %d",
                    frame, read_x, read_y,
                    read_psi.re, read_psi.im, read_magnitude);

                read_x <= MESH_X/2;
                @(posedge clk);
                $display("           Psi[%d,%d] = (%d, %d) |Psi|² = %d",
                    read_x, read_y,
                    read_psi.re, read_psi.im, read_magnitude);

                read_x <= 3*MESH_X/4;
                @(posedge clk);
                $display("           Psi[%d,%d] = (%d, %d) |Psi|² = %d",
                    read_x, read_y,
                    read_psi.re, read_psi.im, read_magnitude);
            end
        end

        // -----------------------------------------------------------------
        // Final field dump
        // -----------------------------------------------------------------
        $display("\n=== Final Field State ===");
        for (int y = MESH_Y-1; y >= 0; y--) begin
            $write("y=%d: ", y);
            for (int x = 0; x < MESH_X; x++) begin
                read_x <= x;
                read_y <= y;
                @(posedge clk);
                // Print magnitude as ASCII density
                if (read_magnitude > 1000)
                    $write("#");
                else if (read_magnitude > 500)
                    $write("*");
                else if (read_magnitude > 100)
                    $write(".");
                else
                    $write(" ");
            end
            $write("\n");
        end

        $display("\n=== Test Complete ===");
        #100;
        $finish;
    end

    // =========================================================================
    // TIMEOUT
    // =========================================================================

    initial begin
        #100000;
        $display("ERROR: Test timeout!");
        $finish;
    end

endmodule : tb_wave_mesh
