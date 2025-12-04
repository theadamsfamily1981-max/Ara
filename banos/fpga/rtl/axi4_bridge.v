//============================================================================
// BANOS - Bio-Affective Neuromorphic Operating System
// AXI4-Lite Bridge - CPU-FPGA Neural State Interface
//============================================================================
// This module maps the FPGA's neural state to memory addresses that the
// CPU can read. It is the "synapse" connecting the hindbrain (FPGA) to
// the brainstem (kernel).
//
// Memory Map:
// 0x00: NEURAL_STATE    - Bitmap of currently firing neurons (RO)
// 0x04: PAIN_LEVEL      - Integrated spike count = System suffering (RO)
// 0x08: REFLEX_LOG      - What actions did the FPGA take? (RO)
// 0x0C: AROUSAL_LEVEL   - Current system arousal (RO)
// 0x10: DOMINANCE_LEVEL - Available resources metric (RO)
// 0x14: PLEASURE_LEVEL  - Inverse of thermal stress + errors (RO)
// 0x18: ALERT_STATUS    - Current alert state (RO)
// 0x1C: TIMESTAMP       - FPGA clock counter (RO)
// 0x20: CONTROL         - Enable/disable, learning mode (RW)
// 0x24: CONFIG          - Threshold configuration (RW)
// 0x28: IRQ_ENABLE      - Interrupt enable mask (RW)
// 0x2C: IRQ_STATUS      - Interrupt status (R/W1C)
//============================================================================

`timescale 1ns / 1ps

module axi4_lite_bridge #(
    parameter C_S_AXI_DATA_WIDTH = 32,
    parameter C_S_AXI_ADDR_WIDTH = 8,
    parameter NUM_RECURRENT      = 32
)(
    // AXI4-Lite Slave Interface
    input  wire                                S_AXI_ACLK,
    input  wire                                S_AXI_ARESETN,

    // Write address channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       S_AXI_AWADDR,
    input  wire [2:0]                          S_AXI_AWPROT,
    input  wire                                S_AXI_AWVALID,
    output wire                                S_AXI_AWREADY,

    // Write data channel
    input  wire [C_S_AXI_DATA_WIDTH-1:0]       S_AXI_WDATA,
    input  wire [(C_S_AXI_DATA_WIDTH/8)-1:0]   S_AXI_WSTRB,
    input  wire                                S_AXI_WVALID,
    output wire                                S_AXI_WREADY,

    // Write response channel
    output wire [1:0]                          S_AXI_BRESP,
    output wire                                S_AXI_BVALID,
    input  wire                                S_AXI_BREADY,

    // Read address channel
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]       S_AXI_ARADDR,
    input  wire [2:0]                          S_AXI_ARPROT,
    input  wire                                S_AXI_ARVALID,
    output wire                                S_AXI_ARREADY,

    // Read data channel
    output wire [C_S_AXI_DATA_WIDTH-1:0]       S_AXI_RDATA,
    output wire [1:0]                          S_AXI_RRESP,
    output wire                                S_AXI_RVALID,
    input  wire                                S_AXI_RREADY,

    // Neural State Inputs (from Vacuum Spiker)
    input  wire [NUM_RECURRENT-1:0]            neural_state,
    input  wire [15:0]                         pain_level,
    input  wire [31:0]                         reflex_log,
    input  wire                                alert_active,
    input  wire                                vacuum_state,
    input  wire [31:0]                         total_spikes,

    // PAD Inputs (from thermal/system monitors)
    input  wire [15:0]                         pleasure_raw,
    input  wire [15:0]                         arousal_raw,
    input  wire [15:0]                         dominance_raw,

    // Reflex Inputs
    input  wire                                prochot_active,
    input  wire [7:0]                          fan_pwm_current,
    input  wire                                emergency_active,

    // Control Outputs
    output reg                                 snn_enable,
    output reg                                 learn_enable,
    output reg                                 force_inhibit,
    output reg  [15:0]                         alert_threshold,

    // Interrupt Output
    output wire                                irq
);

    //------------------------------------------------------------------------
    // AXI4-Lite Signals
    //------------------------------------------------------------------------

    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    reg                          axi_awready;
    reg                          axi_wready;
    reg [1:0]                    axi_bresp;
    reg                          axi_bvalid;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    reg                          axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    reg [1:0]                    axi_rresp;
    reg                          axi_rvalid;

    // Internal Registers
    reg [31:0] control_reg;       // 0x20
    reg [31:0] config_reg;        // 0x24
    reg [31:0] irq_enable_reg;    // 0x28
    reg [31:0] irq_status_reg;    // 0x2C

    // Timestamp counter
    reg [31:0] timestamp;

    //------------------------------------------------------------------------
    // AXI Signal Assignments
    //------------------------------------------------------------------------

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    //------------------------------------------------------------------------
    // Timestamp Counter
    //------------------------------------------------------------------------

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            timestamp <= 32'b0;
        end else begin
            timestamp <= timestamp + 1;
        end
    end

    //------------------------------------------------------------------------
    // Control Register Decoding
    //------------------------------------------------------------------------

    always @(*) begin
        snn_enable     = control_reg[0];
        learn_enable   = control_reg[1];
        force_inhibit  = control_reg[2];
        alert_threshold = config_reg[15:0];
    end

    //------------------------------------------------------------------------
    // Write Address Channel
    //------------------------------------------------------------------------

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 1'b0;
            axi_awaddr  <= 0;
        end else begin
            if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID) begin
                axi_awready <= 1'b1;
                axi_awaddr  <= S_AXI_AWADDR;
            end else begin
                axi_awready <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Write Data Channel
    //------------------------------------------------------------------------

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_wready <= 1'b0;
        end else begin
            if (~axi_wready && S_AXI_WVALID && S_AXI_AWVALID) begin
                axi_wready <= 1'b1;
            end else begin
                axi_wready <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Write Response Channel
    //------------------------------------------------------------------------

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_bvalid <= 1'b0;
            axi_bresp  <= 2'b0;
        end else begin
            if (axi_awready && S_AXI_AWVALID && axi_wready && S_AXI_WVALID && ~axi_bvalid) begin
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b00;  // OKAY
            end else if (S_AXI_BREADY && axi_bvalid) begin
                axi_bvalid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Register Write Logic
    //------------------------------------------------------------------------

    wire wr_en;
    assign wr_en = axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            control_reg    <= 32'h0000_0001;  // Enable by default
            config_reg     <= 32'h0000_0005;  // Alert threshold = 5
            irq_enable_reg <= 32'h0000_0000;
            irq_status_reg <= 32'h0000_0000;
        end else if (wr_en) begin
            case (axi_awaddr[7:2])
                6'h08: control_reg    <= S_AXI_WDATA;  // 0x20
                6'h09: config_reg     <= S_AXI_WDATA;  // 0x24
                6'h0A: irq_enable_reg <= S_AXI_WDATA;  // 0x28
                6'h0B: irq_status_reg <= irq_status_reg & ~S_AXI_WDATA;  // 0x2C (W1C)
                default: ;  // Ignore writes to read-only registers
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Read Address Channel
    //------------------------------------------------------------------------

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_arready <= 1'b0;
            axi_araddr  <= 0;
        end else begin
            if (~axi_arready && S_AXI_ARVALID) begin
                axi_arready <= 1'b1;
                axi_araddr  <= S_AXI_ARADDR;
            end else begin
                axi_arready <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Read Data Channel
    //------------------------------------------------------------------------

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rvalid <= 1'b0;
            axi_rresp  <= 2'b0;
        end else begin
            if (axi_arready && S_AXI_ARVALID && ~axi_rvalid) begin
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b00;  // OKAY
            end else if (axi_rvalid && S_AXI_RREADY) begin
                axi_rvalid <= 1'b0;
            end
        end
    end

    //------------------------------------------------------------------------
    // Register Read Logic
    //------------------------------------------------------------------------

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_rdata <= 32'b0;
        end else if (axi_arready && S_AXI_ARVALID && ~axi_rvalid) begin
            case (axi_araddr[7:2])
                // Read-only registers
                6'h00: axi_rdata <= {{(32-NUM_RECURRENT){1'b0}}, neural_state};  // 0x00: NEURAL_STATE
                6'h01: axi_rdata <= {16'b0, pain_level};                         // 0x04: PAIN_LEVEL
                6'h02: axi_rdata <= reflex_log;                                  // 0x08: REFLEX_LOG
                6'h03: axi_rdata <= {16'b0, arousal_raw};                        // 0x0C: AROUSAL_LEVEL
                6'h04: axi_rdata <= {16'b0, dominance_raw};                      // 0x10: DOMINANCE_LEVEL
                6'h05: axi_rdata <= {16'b0, pleasure_raw};                       // 0x14: PLEASURE_LEVEL
                6'h06: begin                                                      // 0x18: ALERT_STATUS
                    axi_rdata <= {
                        24'b0,
                        emergency_active,      // [7]
                        prochot_active,        // [6]
                        3'b0,
                        vacuum_state,          // [2]
                        alert_active,          // [1]
                        snn_enable             // [0]
                    };
                end
                6'h07: axi_rdata <= timestamp;                                   // 0x1C: TIMESTAMP

                // Read-write registers
                6'h08: axi_rdata <= control_reg;                                 // 0x20: CONTROL
                6'h09: axi_rdata <= config_reg;                                  // 0x24: CONFIG
                6'h0A: axi_rdata <= irq_enable_reg;                              // 0x28: IRQ_ENABLE
                6'h0B: axi_rdata <= irq_status_reg;                              // 0x2C: IRQ_STATUS

                // Statistics
                6'h0C: axi_rdata <= total_spikes;                                // 0x30: TOTAL_SPIKES
                6'h0D: axi_rdata <= {24'b0, fan_pwm_current};                    // 0x34: FAN_PWM

                default: axi_rdata <= 32'hDEAD_BEEF;  // Invalid address marker
            endcase
        end
    end

    //------------------------------------------------------------------------
    // Interrupt Logic
    //------------------------------------------------------------------------

    // Interrupt sources
    wire irq_alert     = alert_active    && irq_enable_reg[0];
    wire irq_emergency = emergency_active && irq_enable_reg[1];
    wire irq_prochot   = prochot_active  && irq_enable_reg[2];

    // Update IRQ status on rising edges
    reg alert_prev, emergency_prev, prochot_prev;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            alert_prev <= 1'b0;
            emergency_prev <= 1'b0;
            prochot_prev <= 1'b0;
        end else begin
            alert_prev <= alert_active;
            emergency_prev <= emergency_active;
            prochot_prev <= prochot_active;

            // Set status on rising edge
            if (alert_active && !alert_prev)
                irq_status_reg[0] <= 1'b1;
            if (emergency_active && !emergency_prev)
                irq_status_reg[1] <= 1'b1;
            if (prochot_active && !prochot_prev)
                irq_status_reg[2] <= 1'b1;
        end
    end

    // Combined IRQ output
    assign irq = |(irq_status_reg & irq_enable_reg);

endmodule


//============================================================================
// Top-Level BANOS Hindbrain Module
// Instantiates all Phase 1 components
//============================================================================

module banos_hindbrain #(
    parameter NUM_SENSORS    = 4,           // Temperature, voltage, etc.
    parameter NUM_INTERVALS  = 16,          // Interval coding resolution
    parameter NUM_RECURRENT  = 32,          // SNN recurrent neurons
    parameter AXI_DATA_WIDTH = 32,
    parameter AXI_ADDR_WIDTH = 8
)(
    // Clock and Reset
    input  wire        clk,
    input  wire        rst_n,

    // Sensor Inputs (raw ADC values)
    input  wire [NUM_SENSORS*16-1:0] sensor_values,
    input  wire [NUM_SENSORS-1:0]    sensor_valid,

    // Hardware Reflex Outputs
    output wire        prochot,
    output wire [7:0]  fan_pwm,
    output wire        emergency_shutdown,

    // AXI4-Lite Slave Interface
    input  wire [AXI_ADDR_WIDTH-1:0]  S_AXI_AWADDR,
    input  wire [2:0]                 S_AXI_AWPROT,
    input  wire                       S_AXI_AWVALID,
    output wire                       S_AXI_AWREADY,
    input  wire [AXI_DATA_WIDTH-1:0]  S_AXI_WDATA,
    input  wire [3:0]                 S_AXI_WSTRB,
    input  wire                       S_AXI_WVALID,
    output wire                       S_AXI_WREADY,
    output wire [1:0]                 S_AXI_BRESP,
    output wire                       S_AXI_BVALID,
    input  wire                       S_AXI_BREADY,
    input  wire [AXI_ADDR_WIDTH-1:0]  S_AXI_ARADDR,
    input  wire [2:0]                 S_AXI_ARPROT,
    input  wire                       S_AXI_ARVALID,
    output wire                       S_AXI_ARREADY,
    output wire [AXI_DATA_WIDTH-1:0]  S_AXI_RDATA,
    output wire [1:0]                 S_AXI_RRESP,
    output wire                       S_AXI_RVALID,
    input  wire                       S_AXI_RREADY,

    // Interrupt
    output wire        irq
);

    //------------------------------------------------------------------------
    // Internal Wires
    //------------------------------------------------------------------------

    // Encoder outputs
    wire [NUM_SENSORS*NUM_INTERVALS-1:0] spike_patterns;
    wire [NUM_SENSORS-1:0] encoder_valid;

    // Vacuum Spiker outputs
    wire [NUM_RECURRENT-1:0] snn_spikes;
    wire [$clog2(NUM_RECURRENT):0] spike_count;
    wire snn_alert;
    wire [15:0] pain_level;
    wire vacuum_state;
    wire [31:0] total_spikes;

    // Reflex controller outputs
    wire prochot_internal;
    wire [7:0] fan_pwm_internal;
    wire emergency_internal;
    wire [31:0] reflex_log;
    wire [3:0] kill_priority;
    wire kill_request;

    // AXI bridge control outputs
    wire snn_enable;
    wire learn_enable;
    wire force_inhibit;
    wire [15:0] alert_threshold;

    // Thermal flags (simplified - first sensor assumed to be CPU temp)
    wire thermal_warning, thermal_critical, thermal_emergency;

    //------------------------------------------------------------------------
    // Multi-Channel Interval Encoder
    //------------------------------------------------------------------------

    multi_channel_encoder #(
        .NUM_CHANNELS(NUM_SENSORS),
        .DATA_WIDTH(16),
        .NUM_INTERVALS(NUM_INTERVALS),
        .INPUT_MIN(16'h0000),
        .INPUT_MAX(16'hFFFF)
    ) encoder (
        .clk(clk),
        .rst_n(rst_n),
        .input_values(sensor_values),
        .input_valid(sensor_valid),
        .spike_patterns(spike_patterns),
        .output_valid(encoder_valid),
        .total_spikes()
    );

    //------------------------------------------------------------------------
    // Thermal Encoder (Channel 0)
    //------------------------------------------------------------------------

    thermal_encoder #(
        .NUM_INTERVALS(NUM_INTERVALS),
        .TEMP_MIN_RAW(16'h0000),
        .TEMP_MAX_RAW(16'hFFFF),
        .TEMP_WARNING(16'hB333),   // ~80°C
        .TEMP_CRITICAL(16'hD999)   // ~90°C
    ) thermal_enc (
        .clk(clk),
        .rst_n(rst_n),
        .temp_raw(sensor_values[15:0]),
        .temp_valid(sensor_valid[0]),
        .spike_pattern(),  // Handled by multi_channel_encoder
        .output_valid(),
        .active_interval(),
        .warning_flag(thermal_warning),
        .critical_flag(thermal_critical),
        .emergency_flag(thermal_emergency)
    );

    //------------------------------------------------------------------------
    // Vacuum Spiker SNN
    //------------------------------------------------------------------------

    vacuum_spiker #(
        .NUM_INPUTS(NUM_SENSORS * NUM_INTERVALS),
        .NUM_RECURRENT(NUM_RECURRENT),
        .DATA_WIDTH(16),
        .FRAC_BITS(8),
        .WEIGHT_BITS(8),
        .ALERT_THRESHOLD(5),
        .WINDOW_CYCLES(256)
    ) snn (
        .clk(clk),
        .rst_n(rst_n),
        .enable(snn_enable),
        .input_spikes(spike_patterns),
        .input_valid(|encoder_valid),
        .learn_enable(learn_enable),
        .force_inhibit(force_inhibit),
        .weight_we(1'b0),
        .weight_addr(0),
        .weight_wdata(0),
        .weight_rdata(),
        .output_spikes(snn_spikes),
        .spike_count(spike_count),
        .alert(snn_alert),
        .pain_level(pain_level),
        .vacuum_state(vacuum_state),
        .total_spikes_lifetime(total_spikes)
    );

    //------------------------------------------------------------------------
    // Reflex Controller
    //------------------------------------------------------------------------

    reflex_controller #(
        .NUM_CHANNELS(NUM_SENSORS)
    ) reflex (
        .clk(clk),
        .rst_n(rst_n),
        .enable(snn_enable),
        .alerts({NUM_SENSORS{snn_alert}}),
        .pain_levels({NUM_SENSORS{pain_level}}),
        .thermal_critical({NUM_SENSORS{thermal_critical}}),
        .thermal_emergency({NUM_SENSORS{thermal_emergency}}),
        .prochot_assert(prochot_internal),
        .fan_pwm_override(fan_pwm_internal),
        .emergency_shutdown(emergency_internal),
        .kill_priority(kill_priority),
        .kill_request(kill_request),
        .reflex_action_log(reflex_log),
        .reflex_timestamp()
    );

    //------------------------------------------------------------------------
    // AXI4-Lite Bridge
    //------------------------------------------------------------------------

    // Placeholder PAD values (would come from additional sensors)
    wire [15:0] pleasure_raw = 16'hFFFF - pain_level;  // Inverse of pain
    wire [15:0] arousal_raw  = {8'b0, spike_count, 3'b0};  // Based on activity
    wire [15:0] dominance_raw = 16'h8000;  // Placeholder

    axi4_lite_bridge #(
        .C_S_AXI_DATA_WIDTH(AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .NUM_RECURRENT(NUM_RECURRENT)
    ) axi_bridge (
        .S_AXI_ACLK(clk),
        .S_AXI_ARESETN(rst_n),
        .S_AXI_AWADDR(S_AXI_AWADDR),
        .S_AXI_AWPROT(S_AXI_AWPROT),
        .S_AXI_AWVALID(S_AXI_AWVALID),
        .S_AXI_AWREADY(S_AXI_AWREADY),
        .S_AXI_WDATA(S_AXI_WDATA),
        .S_AXI_WSTRB(S_AXI_WSTRB),
        .S_AXI_WVALID(S_AXI_WVALID),
        .S_AXI_WREADY(S_AXI_WREADY),
        .S_AXI_BRESP(S_AXI_BRESP),
        .S_AXI_BVALID(S_AXI_BVALID),
        .S_AXI_BREADY(S_AXI_BREADY),
        .S_AXI_ARADDR(S_AXI_ARADDR),
        .S_AXI_ARPROT(S_AXI_ARPROT),
        .S_AXI_ARVALID(S_AXI_ARVALID),
        .S_AXI_ARREADY(S_AXI_ARREADY),
        .S_AXI_RDATA(S_AXI_RDATA),
        .S_AXI_RRESP(S_AXI_RRESP),
        .S_AXI_RVALID(S_AXI_RVALID),
        .S_AXI_RREADY(S_AXI_RREADY),
        .neural_state(snn_spikes),
        .pain_level(pain_level),
        .reflex_log(reflex_log),
        .alert_active(snn_alert),
        .vacuum_state(vacuum_state),
        .total_spikes(total_spikes),
        .pleasure_raw(pleasure_raw),
        .arousal_raw(arousal_raw),
        .dominance_raw(dominance_raw),
        .prochot_active(prochot_internal),
        .fan_pwm_current(fan_pwm_internal),
        .emergency_active(emergency_internal),
        .snn_enable(snn_enable),
        .learn_enable(learn_enable),
        .force_inhibit(force_inhibit),
        .alert_threshold(alert_threshold),
        .irq(irq)
    );

    //------------------------------------------------------------------------
    // Output Assignments
    //------------------------------------------------------------------------

    assign prochot = prochot_internal;
    assign fan_pwm = fan_pwm_internal;
    assign emergency_shutdown = emergency_internal;

endmodule
