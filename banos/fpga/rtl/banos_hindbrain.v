/**
 * BANOS HINDBRAIN - Top-Level FPGA Module
 * ========================================
 *
 * Bio-Affective Neuromorphic Operating System
 * VU7P Target (SB-852 Carrier Board)
 *
 * This is the "brainstem" - the autonomous regulatory center that:
 * 1. Manages the Circadian Clock (sleep/wake cycles)
 * 2. Hosts the Spiking Neural Network (vacuum_spiker)
 * 3. Routes data between PCIe host and neural fabric
 * 4. Implements reflexive responses (faster than CPU can react)
 *
 * The hindbrain operates autonomously. The cerebral cortex (Threadripper)
 * can observe and influence, but the hindbrain protects the body even
 * when the cortex is asleep or overloaded.
 *
 * Memory Map (via PCIe BAR0):
 *   0x0000-0x00FF : Status registers
 *   0x0100-0x01FF : Circadian controller
 *   0x0200-0x02FF : SNN control
 *   0x0300-0x03FF : Neuro-symbolic bridge
 *   0x1000-0x1FFF : Spike injection FIFO
 *   0x2000-0x2FFF : Spike readout FIFO
 */

module banos_hindbrain #(
    parameter CLK_FREQ_MHZ = 300,
    parameter NUM_NEURONS  = 1024,
    parameter GRID_SIZE    = 32
)(
    // Clock & Reset
    input  wire        clk,
    input  wire        rst_n,

    // PCIe AXI-Lite Interface (from XDMA)
    input  wire [31:0] s_axil_awaddr,
    input  wire        s_axil_awvalid,
    output wire        s_axil_awready,
    input  wire [31:0] s_axil_wdata,
    input  wire [3:0]  s_axil_wstrb,
    input  wire        s_axil_wvalid,
    output wire        s_axil_wready,
    output wire [1:0]  s_axil_bresp,
    output wire        s_axil_bvalid,
    input  wire        s_axil_bready,
    input  wire [31:0] s_axil_araddr,
    input  wire        s_axil_arvalid,
    output wire        s_axil_arready,
    output wire [31:0] s_axil_rdata,
    output wire [1:0]  s_axil_rresp,
    output wire        s_axil_rvalid,
    input  wire        s_axil_rready,

    // External Sensors
    input  wire [15:0] temperature_mc,   // Die temperature in milli-Celsius
    input  wire [15:0] vccint_mv,        // Core voltage in millivolts
    input  wire [15:0] fan_rpm,          // Cooling fan RPM

    // Host Interrupts
    output wire        irq_alert,        // Alert to host (pain spike, thermal, etc.)
    output wire        irq_sleep,        // Sleep request to host

    // Debug LEDs (directly drive 8 LEDs on carrier)
    output wire [7:0]  debug_leds
);

    // ==========================================================================
    // Internal Signals
    // ==========================================================================

    // Spike network
    wire [31:0] total_spikes;
    wire [15:0] spike_rate;
    wire [NUM_NEURONS-1:0] spike_vector;

    // Circadian signals
    wire        sleep_mode;
    wire        dream_active;
    wire [7:0]  melatonin_level;
    wire [7:0]  circadian_phase;
    wire [31:0] fatigue_level;
    wire [1:0]  sleep_state;

    // Host control
    reg         force_sleep_r;
    reg         force_wake_r;
    reg         learn_enable_r;

    // Pain/entropy signals
    wire [15:0] current_entropy;
    wire [31:0] pain_level;

    // Neuro-symbolic bridge signals
    wire [1023:0] descending_spikes;  // LLM attention -> SNN
    wire [255:0]  ascending_vector;   // SNN aggregate -> LLM hormone

    // ==========================================================================
    // Register Decoding
    // ==========================================================================

    // Simple register file for host access
    reg [31:0] status_reg;
    reg [31:0] control_reg;
    reg [31:0] rdata_reg;
    reg        rvalid_reg;

    // Address decode
    wire [11:0] addr = s_axil_awvalid ? s_axil_awaddr[11:0] :
                       s_axil_arvalid ? s_axil_araddr[11:0] : 12'd0;

    wire addr_status    = (addr[11:8] == 4'h0);  // 0x0xx
    wire addr_circadian = (addr[11:8] == 4'h1);  // 0x1xx
    wire addr_snn       = (addr[11:8] == 4'h2);  // 0x2xx
    wire addr_bridge    = (addr[11:8] == 4'h3);  // 0x3xx

    // ==========================================================================
    // Status Register (0x000)
    // ==========================================================================
    //
    // [7:0]   melatonin_level  - Tiredness (0=alert, 255=exhausted)
    // [15:8]  circadian_phase  - Cycle position (0-127=wake, 128-255=sleep)
    // [17:16] sleep_state      - 0=AWAKE, 1=DROWSY, 2=REM, 3=DEEP
    // [18]    sleep_mode       - 1=Sleep requested
    // [19]    dream_active     - 1=STDP learning active
    // [31:20] reserved

    always @(posedge clk) begin
        status_reg <= {
            12'd0,                      // [31:20] reserved
            dream_active,               // [19]
            sleep_mode,                 // [18]
            sleep_state,                // [17:16]
            circadian_phase,            // [15:8]
            melatonin_level             // [7:0]
        };
    end

    // ==========================================================================
    // Control Register (0x004)
    // ==========================================================================
    //
    // [0] force_sleep  - Write 1 to force sleep
    // [1] force_wake   - Write 1 to force wake
    // [2] learn_enable - Enable STDP learning manually
    // [3] reset_fatigue - Reset fatigue counter

    always @(posedge clk) begin
        if (!rst_n) begin
            control_reg <= 32'd0;
            force_sleep_r <= 1'b0;
            force_wake_r <= 1'b0;
            learn_enable_r <= 1'b0;
        end else begin
            // Self-clearing bits
            force_sleep_r <= 1'b0;
            force_wake_r <= 1'b0;

            // Handle writes
            if (s_axil_awvalid && s_axil_wvalid && addr == 12'h004) begin
                control_reg <= s_axil_wdata;
                force_sleep_r <= s_axil_wdata[0];
                force_wake_r <= s_axil_wdata[1];
                learn_enable_r <= s_axil_wdata[2];
            end
        end
    end

    // ==========================================================================
    // Read Logic
    // ==========================================================================

    always @(posedge clk) begin
        if (!rst_n) begin
            rdata_reg <= 32'd0;
            rvalid_reg <= 1'b0;
        end else begin
            rvalid_reg <= s_axil_arvalid;

            if (s_axil_arvalid) begin
                case (addr)
                    12'h000: rdata_reg <= status_reg;
                    12'h004: rdata_reg <= control_reg;
                    12'h008: rdata_reg <= fatigue_level;
                    12'h00C: rdata_reg <= total_spikes;
                    12'h010: rdata_reg <= {16'd0, current_entropy};
                    12'h014: rdata_reg <= pain_level;
                    12'h018: rdata_reg <= {16'd0, temperature_mc};
                    12'h01C: rdata_reg <= {16'd0, spike_rate};
                    12'h100: rdata_reg <= {24'd0, melatonin_level};
                    12'h104: rdata_reg <= {24'd0, circadian_phase};
                    12'h108: rdata_reg <= {30'd0, sleep_state};
                    default: rdata_reg <= 32'hDEAD_BEEF;
                endcase
            end
        end
    end

    // AXI-Lite response
    assign s_axil_awready = 1'b1;
    assign s_axil_wready  = 1'b1;
    assign s_axil_bresp   = 2'b00;
    assign s_axil_bvalid  = s_axil_awvalid && s_axil_wvalid;
    assign s_axil_arready = 1'b1;
    assign s_axil_rdata   = rdata_reg;
    assign s_axil_rresp   = 2'b00;
    assign s_axil_rvalid  = rvalid_reg;

    // ==========================================================================
    // Entropy Calculation
    // ==========================================================================

    // Combine temperature and spike variance into entropy metric
    // Higher temp + higher spike variance = higher entropy = more fatigue
    wire [15:0] thermal_component = (temperature_mc > 16'd50000) ?
                                    (temperature_mc - 16'd50000) >> 4 : 16'd0;
    wire [15:0] activity_component = spike_rate;

    assign current_entropy = thermal_component + activity_component;

    // Pain level from high temperature or high entropy
    assign pain_level = {16'd0, current_entropy} +
                       ((temperature_mc > 16'd80000) ? {16'd0, temperature_mc - 16'd80000} : 32'd0);

    // ==========================================================================
    // Circadian Controller
    // ==========================================================================

    circadian_controller #(
        .CLK_FREQ_MHZ(CLK_FREQ_MHZ),
        .FATIGUE_LIMIT(32'hF000_0000),
        .FATIGUE_WAKE(32'h4000_0000)
    ) u_circadian (
        .clk(clk),
        .rst_n(rst_n),

        // Somatic inputs
        .total_spikes(total_spikes),
        .current_entropy(current_entropy),

        // Host overrides
        .force_sleep(force_sleep_r),
        .force_wake(force_wake_r),

        // Outputs
        .sleep_mode(sleep_mode),
        .dream_active(dream_active),
        .melatonin_level(melatonin_level),
        .circadian_phase(circadian_phase),
        .fatigue_level(fatigue_level),
        .sleep_state(sleep_state)
    );

    // ==========================================================================
    // Spiking Neural Network (Vacuum Spiker)
    // ==========================================================================

    // Learning enable: manual OR during dreaming
    wire snn_learn_enable = learn_enable_r | dream_active;

    // Input inhibition during sleep (sensory deprivation)
    wire snn_input_inhibit = sleep_mode;

    vacuum_spiker #(
        .NUM_NEURONS(NUM_NEURONS),
        .GRID_SIZE(GRID_SIZE)
    ) u_snn (
        .clk(clk),
        .rst_n(rst_n),

        // Learning control
        .learn_enable(snn_learn_enable),
        .input_inhibit(snn_input_inhibit),

        // Descending input (from LLM attention during wake)
        .inject_spikes(descending_spikes[NUM_NEURONS-1:0] & {NUM_NEURONS{~sleep_mode}}),

        // Outputs
        .spike_vector(spike_vector),
        .total_spikes(total_spikes),
        .spike_rate(spike_rate)
    );

    // ==========================================================================
    // Interrupts
    // ==========================================================================

    // Alert interrupt: high pain or temperature
    assign irq_alert = (pain_level > 32'h0001_0000) | (temperature_mc > 16'd85000);

    // Sleep interrupt: notify host when entering/exiting sleep
    reg sleep_mode_d;
    always @(posedge clk) sleep_mode_d <= sleep_mode;
    assign irq_sleep = (sleep_mode != sleep_mode_d);

    // ==========================================================================
    // Debug LEDs
    // ==========================================================================

    // LED[7:6] = sleep_state
    // LED[5]   = dream_active
    // LED[4]   = sleep_mode
    // LED[3:0] = melatonin_level[7:4]
    assign debug_leds = {
        sleep_state,        // [7:6]
        dream_active,       // [5]
        sleep_mode,         // [4]
        melatonin_level[7:4] // [3:0]
    };

    // ==========================================================================
    // Placeholder for Ascending Vector (SNN -> LLM hormone)
    // ==========================================================================

    // Aggregate spike activity into hormone-like vector
    // This feeds back to the neuro-symbolic bridge
    assign ascending_vector = {
        spike_rate,         // Activity level
        melatonin_level,    // Tiredness
        circadian_phase,    // Cycle phase
        {8{dream_active}},  // Dream state
        {8{sleep_mode}},    // Sleep state
        {200{1'b0}}         // Reserved
    };

    // Placeholder for descending spikes (LLM -> SNN)
    // In full implementation, this comes from the neuro-symbolic bridge
    assign descending_spikes = {1024{1'b0}};

endmodule
