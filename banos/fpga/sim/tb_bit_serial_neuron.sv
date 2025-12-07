// -----------------------------------------------------------------------------
// tb_bit_serial_neuron.sv
// Testbench for bit-serial neuron with HDC mode
//
// Tests:
//   1. HDC Mode: XOR binding of two hypervector bit streams
//   2. SNN Mode: Integrate-and-fire with threshold
// -----------------------------------------------------------------------------
`timescale 1ns / 1ps

module tb_bit_serial_neuron;

    // Parameters
    localparam WEIGHT_WIDTH  = 8;
    localparam ACC_WIDTH     = 16;
    localparam THRESH_WIDTH  = 16;
    localparam HV_DIM        = 64;  // Small hypervector for testing

    // Signals
    reg                      clk;
    reg                      rst_n;
    reg                      mode_hdc;
    reg                      in_valid;
    reg                      weight_bit;
    reg                      state_bit_in;
    reg                      start;
    reg  [THRESH_WIDTH-1:0]  threshold;
    wire                     state_bit_out;
    wire                     fire_event;

    // Test vectors
    reg [HV_DIM-1:0] hv_a;
    reg [HV_DIM-1:0] hv_b;
    reg [HV_DIM-1:0] hv_c_expected;
    reg [HV_DIM-1:0] hv_c_actual;

    // Counters
    integer i;
    integer errors;
    integer test_num;

    // DUT
    bit_serial_neuron #(
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .THRESH_WIDTH(THRESH_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .mode_hdc(mode_hdc),
        .in_valid(in_valid),
        .weight_bit(weight_bit),
        .state_bit_in(state_bit_in),
        .start(start),
        .threshold(threshold),
        .state_bit_out(state_bit_out),
        .fire_event(fire_event)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100 MHz
    end

    // =========================================================================
    // TEST SEQUENCES
    // =========================================================================

    initial begin
        $display("============================================");
        $display("TB: Bit-Serial Neuron with HDC Mode");
        $display("============================================");

        // Initialize
        rst_n        = 0;
        mode_hdc     = 0;
        in_valid     = 0;
        weight_bit   = 0;
        state_bit_in = 0;
        start        = 0;
        threshold    = 16'd100;
        errors       = 0;
        test_num     = 0;

        // Reset
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(2) @(posedge clk);

        // ---------------------------------------------------------------------
        // TEST 1: HDC Mode - Basic XOR Truth Table
        // ---------------------------------------------------------------------
        test_num = 1;
        $display("\n--- TEST %0d: HDC Mode - XOR Truth Table ---", test_num);

        mode_hdc = 1;
        @(posedge clk);

        // Test all 4 combinations
        test_hdc_xor(0, 0, 0);
        test_hdc_xor(0, 1, 1);
        test_hdc_xor(1, 0, 1);
        test_hdc_xor(1, 1, 0);

        // ---------------------------------------------------------------------
        // TEST 2: HDC Mode - Full Hypervector Binding
        // ---------------------------------------------------------------------
        test_num = 2;
        $display("\n--- TEST %0d: HDC Mode - Full Hypervector Bind ---", test_num);

        // Generate random hypervectors
        hv_a = {$random, $random};  // 64 random bits
        hv_b = {$random, $random};
        hv_c_expected = hv_a ^ hv_b;  // Expected bound result

        $display("HV_A = %h", hv_a);
        $display("HV_B = %h", hv_b);
        $display("Expected C = A XOR B = %h", hv_c_expected);

        // Stream hypervectors through neuron
        mode_hdc = 1;
        hv_c_actual = 0;

        for (i = 0; i < HV_DIM; i = i + 1) begin
            @(posedge clk);
            in_valid     = 1;
            weight_bit   = hv_a[i];      // A bit
            state_bit_in = hv_b[i];      // B bit
            @(posedge clk);
            in_valid     = 0;
            @(posedge clk);
            // Capture output (1 cycle latency)
            hv_c_actual[i] = state_bit_out;
        end

        @(posedge clk);
        in_valid = 0;

        $display("Actual   C = %h", hv_c_actual);

        if (hv_c_actual == hv_c_expected) begin
            $display("PASS: Hypervector binding correct!");
        end else begin
            $display("FAIL: Hypervector binding mismatch!");
            errors = errors + 1;
        end

        // ---------------------------------------------------------------------
        // TEST 3: SNN Mode - Accumulation
        // ---------------------------------------------------------------------
        test_num = 3;
        $display("\n--- TEST %0d: SNN Mode - Accumulation ---", test_num);

        mode_hdc  = 0;
        threshold = 16'd10;

        // Reset accumulator
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // Feed spikes with positive weight
        $display("Feeding spikes with weight_bit=1 (positive)...");

        for (i = 0; i < 20; i = i + 1) begin
            @(posedge clk);
            in_valid     = 1;
            weight_bit   = 1;  // Positive weight
            state_bit_in = 1;  // Spike present
            @(posedge clk);
            in_valid     = 0;

            if (fire_event) begin
                $display("  Cycle %0d: FIRE EVENT (threshold crossed)", i);
            end
        end

        // ---------------------------------------------------------------------
        // TEST 4: Mode Switching
        // ---------------------------------------------------------------------
        test_num = 4;
        $display("\n--- TEST %0d: Mode Switching ---", test_num);

        // Start in SNN mode
        mode_hdc = 0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // Do some SNN work
        for (i = 0; i < 5; i = i + 1) begin
            @(posedge clk);
            in_valid     = 1;
            weight_bit   = 1;
            state_bit_in = 1;
            @(posedge clk);
            in_valid     = 0;
        end

        // Switch to HDC mode instantly
        $display("Switching to HDC mode...");
        mode_hdc = 1;

        // Do HDC binding
        @(posedge clk);
        in_valid     = 1;
        weight_bit   = 1;
        state_bit_in = 0;
        @(posedge clk);
        in_valid     = 0;
        @(posedge clk);

        if (state_bit_out == 1) begin
            $display("PASS: HDC mode active after switch (1 XOR 0 = 1)");
        end else begin
            $display("FAIL: HDC mode not working after switch");
            errors = errors + 1;
        end

        // Switch back to SNN mode
        $display("Switching back to SNN mode...");
        mode_hdc = 0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        $display("SNN mode restored.");

        // ---------------------------------------------------------------------
        // SUMMARY
        // ---------------------------------------------------------------------
        repeat(10) @(posedge clk);

        $display("\n============================================");
        if (errors == 0) begin
            $display("ALL TESTS PASSED!");
        end else begin
            $display("TESTS COMPLETED WITH %0d ERROR(S)", errors);
        end
        $display("============================================\n");

        $finish;
    end

    // =========================================================================
    // HELPER TASKS
    // =========================================================================

    task test_hdc_xor;
        input a, b, expected;
        begin
            @(posedge clk);
            in_valid     = 1;
            weight_bit   = a;
            state_bit_in = b;
            @(posedge clk);
            in_valid     = 0;
            @(posedge clk);  // Wait for output

            if (state_bit_out == expected && fire_event == expected) begin
                $display("  PASS: %0b XOR %0b = %0b", a, b, state_bit_out);
            end else begin
                $display("  FAIL: %0b XOR %0b = %0b (expected %0b)", a, b, state_bit_out, expected);
                errors = errors + 1;
            end
        end
    endtask

endmodule
