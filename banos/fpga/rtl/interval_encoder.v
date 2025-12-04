//============================================================================
// BANOS - Bio-Affective Neuromorphic Operating System
// Interval Encoder - Converts continuous values to single-spike patterns
//============================================================================
// Implements the Interval Coding scheme from Vacuum Spiker:
// - Partitions input domain into k non-overlapping intervals
// - Each interval maps to a unique neuron
// - Exactly ONE spike per input sample (energy efficient)
//
// This enables non-linear separability while maintaining O(1) spike budget.
//============================================================================

`timescale 1ns / 1ps

module interval_encoder #(
    parameter DATA_WIDTH     = 16,          // Input data width
    parameter NUM_INTERVALS  = 16,          // Number of discretization intervals
    parameter INPUT_MIN      = 16'h0000,    // Minimum input value (e.g., 40°C)
    parameter INPUT_MAX      = 16'hFFFF     // Maximum input value (e.g., 90°C)
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Input value (continuous)
    input  wire [DATA_WIDTH-1:0]   input_value,
    input  wire                    input_valid,

    // Output spikes (one-hot encoded)
    output reg  [NUM_INTERVALS-1:0] spike_pattern,
    output reg                     output_valid,

    // Interval that fired (for debugging)
    output reg  [$clog2(NUM_INTERVALS)-1:0] active_interval,

    // Status
    output wire                    clamped_low,   // Input was below minimum
    output wire                    clamped_high   // Input was above maximum
);

    //------------------------------------------------------------------------
    // Constants
    //------------------------------------------------------------------------

    localparam [DATA_WIDTH-1:0] RANGE = INPUT_MAX - INPUT_MIN;
    localparam [DATA_WIDTH-1:0] INTERVAL_WIDTH = RANGE / NUM_INTERVALS;

    //------------------------------------------------------------------------
    // Internal Signals
    //------------------------------------------------------------------------

    reg [DATA_WIDTH-1:0] clamped_value;
    reg [$clog2(NUM_INTERVALS)-1:0] interval_index;
    reg input_below_min, input_above_max;

    //------------------------------------------------------------------------
    // Clamping Logic
    //------------------------------------------------------------------------

    assign clamped_low  = input_below_min;
    assign clamped_high = input_above_max;

    always @(*) begin
        input_below_min = (input_value < INPUT_MIN);
        input_above_max = (input_value > INPUT_MAX);

        if (input_below_min) begin
            clamped_value = INPUT_MIN;
        end else if (input_above_max) begin
            clamped_value = INPUT_MAX;
        end else begin
            clamped_value = input_value;
        end
    end

    //------------------------------------------------------------------------
    // Interval Mapping
    // index = floor((value - INPUT_MIN) / INTERVAL_WIDTH)
    //------------------------------------------------------------------------

    wire [DATA_WIDTH-1:0] offset_value;
    wire [2*DATA_WIDTH-1:0] division_result;

    assign offset_value = clamped_value - INPUT_MIN;

    // Division by constant (synthesizes to shift/multiply)
    // For power-of-2 NUM_INTERVALS, this becomes a simple shift
    generate
        if ((NUM_INTERVALS & (NUM_INTERVALS - 1)) == 0) begin : power_of_two
            // Power of 2: use shift
            localparam SHIFT_BITS = $clog2(NUM_INTERVALS);
            assign division_result = offset_value >> (DATA_WIDTH - SHIFT_BITS);
        end else begin : general_case
            // General case: multiply by reciprocal
            // interval_index = offset_value * NUM_INTERVALS / RANGE
            assign division_result = (offset_value * NUM_INTERVALS) / RANGE;
        end
    endgenerate

    //------------------------------------------------------------------------
    // Pipeline Stage 1: Compute interval index
    //------------------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            interval_index <= 0;
            output_valid <= 1'b0;
        end else begin
            output_valid <= input_valid;

            if (input_valid) begin
                // Clamp index to valid range
                if (division_result >= NUM_INTERVALS) begin
                    interval_index <= NUM_INTERVALS - 1;
                end else begin
                    interval_index <= division_result[$clog2(NUM_INTERVALS)-1:0];
                end
            end
        end
    end

    //------------------------------------------------------------------------
    // Pipeline Stage 2: Generate one-hot spike pattern
    //------------------------------------------------------------------------

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_pattern <= {NUM_INTERVALS{1'b0}};
            active_interval <= 0;
        end else if (output_valid) begin
            // One-hot encoding: exactly one neuron fires
            for (i = 0; i < NUM_INTERVALS; i = i + 1) begin
                spike_pattern[i] <= (i == interval_index);
            end
            active_interval <= interval_index;
        end else begin
            spike_pattern <= {NUM_INTERVALS{1'b0}};
        end
    end

endmodule


//============================================================================
// Multi-Channel Interval Encoder
// Encodes multiple sensor streams in parallel
//============================================================================

module multi_channel_encoder #(
    parameter NUM_CHANNELS   = 4,           // Number of input channels
    parameter DATA_WIDTH     = 16,
    parameter NUM_INTERVALS  = 16,
    parameter INPUT_MIN      = 16'h0000,
    parameter INPUT_MAX      = 16'hFFFF
)(
    input  wire                              clk,
    input  wire                              rst_n,

    // Input values (packed array)
    input  wire [NUM_CHANNELS*DATA_WIDTH-1:0] input_values,
    input  wire [NUM_CHANNELS-1:0]           input_valid,

    // Output spike patterns (one-hot per channel, concatenated)
    output wire [NUM_CHANNELS*NUM_INTERVALS-1:0] spike_patterns,
    output wire [NUM_CHANNELS-1:0]           output_valid,

    // Total neurons that fired this cycle
    output wire [$clog2(NUM_CHANNELS)+1:0]   total_spikes
);

    genvar ch;
    wire [NUM_CHANNELS-1:0] channel_fired;

    generate
        for (ch = 0; ch < NUM_CHANNELS; ch = ch + 1) begin : encoder_gen
            interval_encoder #(
                .DATA_WIDTH(DATA_WIDTH),
                .NUM_INTERVALS(NUM_INTERVALS),
                .INPUT_MIN(INPUT_MIN),
                .INPUT_MAX(INPUT_MAX)
            ) encoder_inst (
                .clk(clk),
                .rst_n(rst_n),
                .input_value(input_values[(ch+1)*DATA_WIDTH-1:ch*DATA_WIDTH]),
                .input_valid(input_valid[ch]),
                .spike_pattern(spike_patterns[(ch+1)*NUM_INTERVALS-1:ch*NUM_INTERVALS]),
                .output_valid(output_valid[ch]),
                .active_interval(),
                .clamped_low(),
                .clamped_high()
            );

            // Track if this channel produced a spike
            assign channel_fired[ch] = output_valid[ch];
        end
    endgenerate

    // Count total spikes (one per active channel due to one-hot encoding)
    integer i;
    reg [$clog2(NUM_CHANNELS)+1:0] spike_sum;

    always @(*) begin
        spike_sum = 0;
        for (i = 0; i < NUM_CHANNELS; i = i + 1) begin
            spike_sum = spike_sum + channel_fired[i];
        end
    end

    assign total_spikes = spike_sum;

endmodule


//============================================================================
// Thermal Interval Encoder - Specialized for CPU/GPU temperature
//============================================================================

module thermal_encoder #(
    parameter NUM_INTERVALS  = 16,
    // Temperature range: 30°C to 100°C mapped to 0x0000 to 0xFFFF
    // Each LSB = (100-30)/65536 = 0.00107°C
    parameter TEMP_MIN_RAW   = 16'h0000,    // 30°C
    parameter TEMP_MAX_RAW   = 16'hFFFF,    // 100°C
    // Critical thresholds (in raw units)
    parameter TEMP_WARNING   = 16'hB333,    // ~80°C
    parameter TEMP_CRITICAL  = 16'hD999     // ~90°C
)(
    input  wire        clk,
    input  wire        rst_n,

    // Raw temperature from sensor (ADC value)
    input  wire [15:0] temp_raw,
    input  wire        temp_valid,

    // Spike output
    output wire [NUM_INTERVALS-1:0] spike_pattern,
    output wire        output_valid,
    output wire [$clog2(NUM_INTERVALS)-1:0] active_interval,

    // Thermal status flags (directly to reflex controller)
    output reg         warning_flag,
    output reg         critical_flag,
    output reg         emergency_flag   // Above critical for extended period
);

    //------------------------------------------------------------------------
    // Interval Encoder Instance
    //------------------------------------------------------------------------

    interval_encoder #(
        .DATA_WIDTH(16),
        .NUM_INTERVALS(NUM_INTERVALS),
        .INPUT_MIN(TEMP_MIN_RAW),
        .INPUT_MAX(TEMP_MAX_RAW)
    ) encoder (
        .clk(clk),
        .rst_n(rst_n),
        .input_value(temp_raw),
        .input_valid(temp_valid),
        .spike_pattern(spike_pattern),
        .output_valid(output_valid),
        .active_interval(active_interval),
        .clamped_low(),
        .clamped_high()
    );

    //------------------------------------------------------------------------
    // Thermal Status Detection
    //------------------------------------------------------------------------

    reg [7:0] critical_counter;  // Counts cycles above critical

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            warning_flag <= 1'b0;
            critical_flag <= 1'b0;
            emergency_flag <= 1'b0;
            critical_counter <= 8'b0;
        end else if (temp_valid) begin
            // Warning: above 80°C
            warning_flag <= (temp_raw >= TEMP_WARNING);

            // Critical: above 90°C
            critical_flag <= (temp_raw >= TEMP_CRITICAL);

            // Emergency: above critical for 255 cycles (~255ms at 1kHz sampling)
            if (temp_raw >= TEMP_CRITICAL) begin
                if (critical_counter < 8'hFF) begin
                    critical_counter <= critical_counter + 1;
                end
                if (critical_counter >= 8'hFF) begin
                    emergency_flag <= 1'b1;
                end
            end else begin
                critical_counter <= 8'b0;
                emergency_flag <= 1'b0;
            end
        end
    end

endmodule
