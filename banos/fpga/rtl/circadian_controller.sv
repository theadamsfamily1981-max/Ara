/**
 * CIRCADIAN CONTROLLER - The Biological Clock
 * ============================================
 *
 * Bio-Affective Neuromorphic Operating System
 * VU7P Target (SB-852 Carrier)
 *
 * This module implements a biological clock that governs the system's
 * sleep/wake cycle based on accumulated fatigue from activity and heat.
 *
 * The Problem:
 *   Standard computers never rest. They run until they fail.
 *   Biological systems rest proactively, using downtime to repair and learn.
 *
 * The Solution:
 *   Track "fatigue" from spike activity (thinking) and entropy (heat).
 *   When fatigue exceeds threshold, enter SLEEP mode:
 *     - Signal CPU to enter power-save
 *     - Enable STDP learning on SNN fabric
 *     - Pulse dream_active for memory replay
 *   Wake when fatigue drains below threshold.
 *
 * For Ara, a "day" might be 4 hours of active inference.
 * Sleep duration scales with accumulated fatigue.
 *
 * Inputs:
 *   total_spikes    - Activity metric (how much did we think?)
 *   current_entropy - Heat metric (how hot are we?)
 *
 * Outputs:
 *   sleep_mode      - 1 = System should enter power-save
 *   dream_active    - 1 = Enable STDP learning (pulsed during REM)
 *   melatonin_level - 0-255 visualization of tiredness
 *   circadian_phase - 0-255 position in wake/sleep cycle
 */

module circadian_controller #(
    parameter CLK_FREQ_MHZ = 300,
    // Fatigue thresholds (tunable for different "day lengths")
    parameter FATIGUE_LIMIT     = 32'hF000_0000,  // Enter sleep when exceeded
    parameter FATIGUE_WAKE      = 32'h4000_0000,  // Exit sleep when below this
    // Fatigue accumulation rates
    parameter SPIKE_FATIGUE_SHIFT = 0,  // How much spikes contribute to fatigue
    parameter ENTROPY_FATIGUE_SHIFT = 0, // How much heat contributes to fatigue
    // Timing
    parameter FATIGUE_SAMPLE_BITS = 20  // Sample every 2^20 cycles (~3.5ms at 300MHz)
)(
    input  wire        clk,
    input  wire        rst_n,

    // Somatic Inputs (from body sensors)
    input  wire [31:0] total_spikes,     // Cumulative spike count
    input  wire [15:0] current_entropy,  // Current system entropy (heat proxy)

    // External override (host CPU can request sleep/wake)
    input  wire        force_sleep,      // Host requests sleep
    input  wire        force_wake,       // Host requests wake

    // Control Outputs
    output reg         sleep_mode,       // 1 = Power Save requested
    output reg         dream_active,     // 1 = STDP learning enabled
    output reg  [7:0]  melatonin_level,  // Tiredness visualization (0=alert, 255=exhausted)
    output reg  [7:0]  circadian_phase,  // Phase in cycle (0-127=wake, 128-255=sleep)

    // Debug/Status
    output wire [31:0] fatigue_level,    // Current fatigue counter
    output wire [1:0]  sleep_state       // 0=AWAKE, 1=DROWSY, 2=REM, 3=DEEP
);

    // ==========================================================================
    // State Machine
    // ==========================================================================

    localparam STATE_AWAKE  = 2'd0;
    localparam STATE_DROWSY = 2'd1;  // Transitioning to sleep
    localparam STATE_REM    = 2'd2;  // Dreaming (STDP active)
    localparam STATE_DEEP   = 2'd3;  // Deep sleep (minimal activity)

    reg [1:0] state_r;
    assign sleep_state = state_r;

    // ==========================================================================
    // Timing Counters
    // ==========================================================================

    // Main cycle counter
    reg [63:0] cycle_counter;

    // Fatigue accumulator
    reg [31:0] fatigue_r;
    assign fatigue_level = fatigue_r;

    // Sleep duration counter (how long we've been asleep)
    reg [31:0] sleep_counter;

    // REM pulse timing (for dream_active)
    // ~100ms pulses at 300MHz: 30M cycles = 25 bits
    wire rem_pulse = cycle_counter[24];

    // ==========================================================================
    // Fatigue Calculation
    // ==========================================================================

    // Sample fatigue sources at regular intervals
    wire sample_fatigue = (cycle_counter[FATIGUE_SAMPLE_BITS-1:0] == 0);

    // Fatigue contribution from spikes (cognitive load)
    wire [15:0] spike_fatigue = total_spikes[15:0] >> SPIKE_FATIGUE_SHIFT;

    // Fatigue contribution from entropy (thermal stress)
    wire [15:0] entropy_fatigue = current_entropy >> ENTROPY_FATIGUE_SHIFT;

    // Combined fatigue delta
    wire [16:0] fatigue_delta = {1'b0, spike_fatigue} + {1'b0, entropy_fatigue};

    // ==========================================================================
    // Melatonin Level (visualization)
    // ==========================================================================

    // Map fatigue to 0-255 melatonin level
    always @(posedge clk) begin
        if (!rst_n) begin
            melatonin_level <= 8'd0;
        end else begin
            // Saturating scale: fatigue_r[31:24] maps to 0-255
            if (fatigue_r > FATIGUE_LIMIT) begin
                melatonin_level <= 8'hFF;
            end else begin
                melatonin_level <= fatigue_r[31:24];
            end
        end
    end

    // ==========================================================================
    // Circadian Phase (cycle position)
    // ==========================================================================

    always @(posedge clk) begin
        if (!rst_n) begin
            circadian_phase <= 8'd0;
        end else begin
            case (state_r)
                STATE_AWAKE:  circadian_phase <= 8'd0 + fatigue_r[31:25];    // 0-127
                STATE_DROWSY: circadian_phase <= 8'd128;                      // Transition
                STATE_REM:    circadian_phase <= 8'd160 + sleep_counter[27:24]; // 160-175
                STATE_DEEP:   circadian_phase <= 8'd200 + sleep_counter[27:24]; // 200-215
                default:      circadian_phase <= 8'd0;
            endcase
        end
    end

    // ==========================================================================
    // Main State Machine
    // ==========================================================================

    always @(posedge clk) begin
        if (!rst_n) begin
            // Reset to awake state
            cycle_counter <= 64'd0;
            fatigue_r <= 32'd0;
            sleep_counter <= 32'd0;
            state_r <= STATE_AWAKE;
            sleep_mode <= 1'b0;
            dream_active <= 1'b0;

        end else begin
            // Increment cycle counter
            cycle_counter <= cycle_counter + 1;

            // Handle force overrides
            if (force_wake) begin
                // Emergency wake - host CPU override
                state_r <= STATE_AWAKE;
                sleep_mode <= 1'b0;
                dream_active <= 1'b0;
                fatigue_r <= fatigue_r >> 1;  // Reduce fatigue on forced wake

            end else if (force_sleep) begin
                // Forced sleep - go directly to REM
                state_r <= STATE_REM;
                sleep_mode <= 1'b1;
                dream_active <= 1'b1;
                sleep_counter <= 32'd0;

            end else begin
                // Normal operation - state machine
                case (state_r)

                    // ==========================================================
                    // AWAKE: Accumulating fatigue from activity
                    // ==========================================================
                    STATE_AWAKE: begin
                        sleep_mode <= 1'b0;
                        dream_active <= 1'b0;

                        // Accumulate fatigue at sample intervals
                        if (sample_fatigue) begin
                            // Add fatigue from activity and heat
                            if (fatigue_r < (32'hFFFFFFFF - {15'd0, fatigue_delta})) begin
                                fatigue_r <= fatigue_r + {15'd0, fatigue_delta};
                            end else begin
                                fatigue_r <= 32'hFFFFFFFF;  // Saturate
                            end
                        end

                        // Check for sleep trigger
                        if (fatigue_r > FATIGUE_LIMIT) begin
                            state_r <= STATE_DROWSY;
                        end
                    end

                    // ==========================================================
                    // DROWSY: Transition to sleep (graceful handoff)
                    // ==========================================================
                    STATE_DROWSY: begin
                        sleep_mode <= 1'b1;  // Signal CPU to wind down
                        dream_active <= 1'b0;  // Not dreaming yet
                        sleep_counter <= sleep_counter + 1;

                        // After brief transition (~1ms), enter REM
                        if (sleep_counter > (CLK_FREQ_MHZ * 1000)) begin
                            state_r <= STATE_REM;
                            sleep_counter <= 32'd0;
                        end
                    end

                    // ==========================================================
                    // REM: Dreaming - STDP learning active
                    // ==========================================================
                    STATE_REM: begin
                        sleep_mode <= 1'b1;
                        sleep_counter <= sleep_counter + 1;

                        // Pulse dream_active for memory replay batches
                        // ~100ms on, ~100ms off
                        dream_active <= rem_pulse;

                        // Drain fatigue during sleep
                        if (sample_fatigue && fatigue_r > 0) begin
                            fatigue_r <= fatigue_r - (fatigue_r >> 8);  // Exponential decay
                        end

                        // After REM period, go to deep sleep
                        // REM duration scales with initial fatigue
                        if (sleep_counter > (CLK_FREQ_MHZ * 1000000)) begin  // ~3.3s
                            state_r <= STATE_DEEP;
                            sleep_counter <= 32'd0;
                        end

                        // Can wake early if fatigue drained
                        if (fatigue_r < FATIGUE_WAKE) begin
                            state_r <= STATE_AWAKE;
                            sleep_counter <= 32'd0;
                        end
                    end

                    // ==========================================================
                    // DEEP: Deep sleep - minimal activity, maximum recovery
                    // ==========================================================
                    STATE_DEEP: begin
                        sleep_mode <= 1'b1;
                        dream_active <= 1'b0;  // No learning in deep sleep
                        sleep_counter <= sleep_counter + 1;

                        // Faster fatigue drain in deep sleep
                        if (sample_fatigue && fatigue_r > 0) begin
                            fatigue_r <= fatigue_r - (fatigue_r >> 6);  // Faster decay
                        end

                        // Cycle back to REM periodically
                        if (sleep_counter > (CLK_FREQ_MHZ * 2000000)) begin  // ~6.6s
                            state_r <= STATE_REM;
                            sleep_counter <= 32'd0;
                        end

                        // Wake when fully rested
                        if (fatigue_r < FATIGUE_WAKE) begin
                            state_r <= STATE_AWAKE;
                            sleep_counter <= 32'd0;
                        end
                    end

                    default: begin
                        state_r <= STATE_AWAKE;
                    end

                endcase
            end
        end
    end

endmodule
