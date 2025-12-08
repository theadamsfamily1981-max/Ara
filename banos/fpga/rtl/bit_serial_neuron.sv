// -----------------------------------------------------------------------------
// bit_serial_neuron.sv
// Bit-serial neuron with dual mode:
//   - SNN mode: integrate-and-fire
//   - HDC mode: bitwise XOR "binding" between two input bit streams
//
// KEY INSIGHT: The XOR gate is SHARED between modes.
//   - SNN Addition:  sum_bit = a ^ b ^ carry
//   - HDC Binding:   bind_bit = a ^ b
//
// In HDC mode:
//   - weight_bit     : acts as Hypervector A bit stream
//   - state_bit_in   : acts as Hypervector B bit stream
//   - state_bit_out  : bound bit (A XOR B)
//   - fire_event     : same as bound bit (for easy routing)
//
// The same 2M neurons that recognize patterns (SNN)
// can bind concepts logically (HDC) on the next clock cycle.
// -----------------------------------------------------------------------------
module bit_serial_neuron #(
    parameter integer WEIGHT_WIDTH   = 8,   // bits per weight (for SNN mode)
    parameter integer ACC_WIDTH      = 16,  // accumulator width
    parameter integer THRESH_WIDTH   = 16   // threshold width
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Mode select
    // 0 = SNN integrate&fire
    // 1 = HDC binding (XOR)
    input  wire                     mode_hdc,

    // --- Common streaming interface ---
    input  wire                     in_valid,      // input bit valid
    input  wire                     weight_bit,    // SNN: weight bit, HDC: Hypervector A bit
    input  wire                     state_bit_in,  // SNN: previous state bit / spike, HDC: Hypervector B bit

    // Control for start-of-word (e.g. new timestep / new input vector)
    input  wire                     start,         // pulse at start of accumulation window
    input  wire [THRESH_WIDTH-1:0]  threshold,     // SNN mode threshold

    // Outputs
    output reg                      state_bit_out, // SNN: accumulated state MSB / next state bit
                                                   // HDC: A XOR B
    output reg                      fire_event     // SNN: spike event
                                                   // HDC: same as state_bit_out
);

    // -------------------------------------------------------------------------
    // SNN MODE STATE
    // -------------------------------------------------------------------------
    reg  [ACC_WIDTH-1:0] acc;          // integration accumulator
    wire [ACC_WIDTH-1:0] acc_next;

    // For very simple demo, interpret state_bit_in as incoming spike bit
    wire spike_in = state_bit_in & in_valid;

    // Example "weight" reconstruction from single bit.
    // In your real design, this will come from a weight shift register / RAM.
    // Here we map weight_bit to +/-1.
    wire signed [1:0] w_signed = weight_bit ? 2'sd1 : -2'sd1;

    // Signed accumulator update (very simplified)
    assign acc_next = acc + {{(ACC_WIDTH-2){w_signed[1]}}, w_signed};

    // Threshold detect in SNN mode
    wire snn_fire = (acc_next >= threshold);

    // -------------------------------------------------------------------------
    // HDC MODE LOGIC
    // -------------------------------------------------------------------------
    // Bind two Hypervectors: A XOR B
    // This is the SAME XOR gate that exists in the SNN adder!
    wire hdc_bind_bit = weight_bit ^ state_bit_in;

    // -------------------------------------------------------------------------
    // SEQUENTIAL STATE UPDATE
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc           <= '0;
            state_bit_out <= 1'b0;
            fire_event    <= 1'b0;
        end else begin
            // Default no spike/result
            fire_event <= 1'b0;

            if (mode_hdc) begin
                // -------------------------------------------------------------
                // HDC MODE
                // - no accumulator used
                // - no thresholding
                // - just stream XOR result
                // -------------------------------------------------------------
                if (in_valid) begin
                    state_bit_out <= hdc_bind_bit;
                    fire_event    <= hdc_bind_bit;  // use as "data" pulse
                end
                // You *could* add accumulation / bundling here if you want
                // a more complex HDC primitive at the neuron level.
            end else begin
                // -------------------------------------------------------------
                // SNN MODE
                // -------------------------------------------------------------
                if (start) begin
                    // Start of new integration window / timestep
                    acc <= '0;
                end else if (in_valid && spike_in) begin
                    acc <= acc_next;
                end

                // state_bit_out could be, e.g., MSB of accumulator
                state_bit_out <= acc[ACC_WIDTH-1];

                // Spike when threshold crossed
                if (in_valid && snn_fire) begin
                    fire_event <= 1'b1;
                end
            end
        end
    end

endmodule


// -----------------------------------------------------------------------------
// bit_serial_neuron_array.sv
// Array of dual-mode neurons for parallel processing
// -----------------------------------------------------------------------------
module bit_serial_neuron_array #(
    parameter integer N_NEURONS     = 256,
    parameter integer WEIGHT_WIDTH  = 8,
    parameter integer ACC_WIDTH     = 16,
    parameter integer THRESH_WIDTH  = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // Global mode (all neurons same mode)
    input  wire                     mode_hdc,

    // Parallel bit inputs (one per neuron)
    input  wire [N_NEURONS-1:0]     weight_bits,    // A bits (HDC) / weight bits (SNN)
    input  wire [N_NEURONS-1:0]     state_bits_in,  // B bits (HDC) / spike bits (SNN)
    input  wire                     in_valid,
    input  wire                     start,
    input  wire [THRESH_WIDTH-1:0]  threshold,

    // Parallel outputs
    output wire [N_NEURONS-1:0]     state_bits_out,
    output wire [N_NEURONS-1:0]     fire_events
);

    genvar i;
    generate
        for (i = 0; i < N_NEURONS; i = i + 1) begin : gen_neurons
            bit_serial_neuron #(
                .WEIGHT_WIDTH(WEIGHT_WIDTH),
                .ACC_WIDTH(ACC_WIDTH),
                .THRESH_WIDTH(THRESH_WIDTH)
            ) neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .mode_hdc(mode_hdc),
                .in_valid(in_valid),
                .weight_bit(weight_bits[i]),
                .state_bit_in(state_bits_in[i]),
                .start(start),
                .threshold(threshold),
                .state_bit_out(state_bits_out[i]),
                .fire_event(fire_events[i])
            );
        end
    endgenerate

endmodule
