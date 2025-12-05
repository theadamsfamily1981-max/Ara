/**
 * ARA-WAVE Field Engine - WaveSite Processing Element
 * =====================================================
 *
 * Single cell of the quantum field lattice.
 * Implements discretized Schrödinger equation in fixed-point.
 *
 * Physics:
 *   Ψ_new = Ψ + i * dt * H * Ψ
 *   H = -∇² + V
 *
 * The 5-point Laplacian stencil:
 *   ∇²Ψ = Ψ_N + Ψ_S + Ψ_E + Ψ_W - 4*Ψ_C
 *
 * Complex rotation by i*dt:
 *   Re_new = Re - dt * Im_term
 *   Im_new = Im + dt * Re_term
 *
 * Where term = -∇²Ψ + V*Ψ (Hamiltonian applied to Ψ)
 *
 * Pipeline: 4 stages
 *   1. Laplacian computation
 *   2. Potential multiply
 *   3. Hamiltonian combine
 *   4. Time evolution (i*dt rotation)
 */

module wave_site
    import wave_pkg::*;
(
    input  logic clk,
    input  logic rst_n,
    input  logic enable,           // Update enable (for sync)

    // Neighbor inputs (from adjacent sites)
    input  complex_t psi_north,
    input  complex_t psi_south,
    input  complex_t psi_east,
    input  complex_t psi_west,

    // This site's current state
    input  complex_t psi_center,
    input  logic signed [V_WIDTH-1:0] potential,

    // Control
    input  logic [DT_WIDTH-1:0] dt,  // Time step

    // Output: updated wavefunction
    output complex_t psi_next,
    output logic     valid_out
);

    // =========================================================================
    // PIPELINE REGISTERS
    // =========================================================================

    // Stage 1: Laplacian
    logic signed [PSI_WIDTH+2:0] lap_re_s1;  // Extra bits for sum
    logic signed [PSI_WIDTH+2:0] lap_im_s1;
    complex_t psi_c_s1;
    logic signed [V_WIDTH-1:0] pot_s1;
    logic valid_s1;

    // Stage 2: Potential term
    logic signed [PSI_WIDTH-1:0] lap_re_s2;
    logic signed [PSI_WIDTH-1:0] lap_im_s2;
    logic signed [PSI_WIDTH-1:0] v_psi_re_s2;
    logic signed [PSI_WIDTH-1:0] v_psi_im_s2;
    logic valid_s2;

    // Stage 3: Hamiltonian
    logic signed [PSI_WIDTH-1:0] H_psi_re_s3;  // -∇²Ψ + V*Ψ
    logic signed [PSI_WIDTH-1:0] H_psi_im_s3;
    complex_t psi_c_s3;
    logic valid_s3;

    // Stage 4: Output
    logic valid_s4;

    // =========================================================================
    // STAGE 1: LAPLACIAN (∇²Ψ = N + S + E + W - 4C)
    // =========================================================================

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lap_re_s1 <= '0;
            lap_im_s1 <= '0;
            psi_c_s1  <= '0;
            pot_s1    <= '0;
            valid_s1  <= 1'b0;
        end else if (enable) begin
            // Sum neighbors
            lap_re_s1 <= $signed(psi_north.re) + $signed(psi_south.re) +
                         $signed(psi_east.re)  + $signed(psi_west.re) -
                         ($signed(psi_center.re) <<< 2);  // -4*center

            lap_im_s1 <= $signed(psi_north.im) + $signed(psi_south.im) +
                         $signed(psi_east.im)  + $signed(psi_west.im) -
                         ($signed(psi_center.im) <<< 2);

            psi_c_s1 <= psi_center;
            pot_s1   <= potential;
            valid_s1 <= 1'b1;
        end else begin
            valid_s1 <= 1'b0;
        end
    end

    // =========================================================================
    // STAGE 2: POTENTIAL TERM (V * Ψ)
    // =========================================================================

    // V is Q4.12, Ψ is Q8.8
    // Product is Q12.20, shift right by 12 to get Q8.8
    logic signed [PSI_WIDTH+V_WIDTH-1:0] v_psi_re_full;
    logic signed [PSI_WIDTH+V_WIDTH-1:0] v_psi_im_full;

    always_comb begin
        v_psi_re_full = pot_s1 * psi_c_s1.re;
        v_psi_im_full = pot_s1 * psi_c_s1.im;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lap_re_s2   <= '0;
            lap_im_s2   <= '0;
            v_psi_re_s2 <= '0;
            v_psi_im_s2 <= '0;
            valid_s2    <= 1'b0;
        end else begin
            // Truncate Laplacian to PSI_WIDTH
            lap_re_s2 <= lap_re_s1[PSI_WIDTH-1:0];
            lap_im_s2 <= lap_im_s1[PSI_WIDTH-1:0];

            // Shift V*Ψ result
            v_psi_re_s2 <= v_psi_re_full[V_FRAC_BITS +: PSI_WIDTH];
            v_psi_im_s2 <= v_psi_im_full[V_FRAC_BITS +: PSI_WIDTH];

            valid_s2 <= valid_s1;
        end
    end

    // =========================================================================
    // STAGE 3: HAMILTONIAN (H*Ψ = -∇²Ψ + V*Ψ)
    // =========================================================================

    // Pipeline psi_center through to stage 3
    complex_t psi_c_s2;
    always_ff @(posedge clk) begin
        psi_c_s2 <= psi_c_s1;
        psi_c_s3 <= psi_c_s2;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            H_psi_re_s3 <= '0;
            H_psi_im_s3 <= '0;
            valid_s3    <= 1'b0;
        end else begin
            // H*Ψ = -∇²Ψ + V*Ψ
            // Note: we negate the Laplacian here
            H_psi_re_s3 <= sat_add(-lap_re_s2, v_psi_re_s2);
            H_psi_im_s3 <= sat_add(-lap_im_s2, v_psi_im_s2);

            valid_s3 <= valid_s2;
        end
    end

    // =========================================================================
    // STAGE 4: TIME EVOLUTION (Ψ_new = Ψ + i*dt*H*Ψ)
    // =========================================================================

    // Complex multiply by i:
    //   i * (Re + i*Im) = -Im + i*Re
    //
    // So: i * dt * H*Ψ = dt * (-H_im + i*H_re)
    //
    // Ψ_new.re = Ψ.re + dt * (-H_im) = Ψ.re - dt * H_im
    // Ψ_new.im = Ψ.im + dt * (H_re)  = Ψ.im + dt * H_re

    logic signed [PSI_WIDTH-1:0] delta_re;
    logic signed [PSI_WIDTH-1:0] delta_im;

    always_comb begin
        // dt * H_im and dt * H_re
        delta_re = fp_mul(-H_psi_im_s3, dt);  // -dt * H_im
        delta_im = fp_mul(H_psi_re_s3, dt);   //  dt * H_re
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            psi_next  <= '0;
            valid_out <= 1'b0;
        end else begin
            psi_next.re <= sat_add(psi_c_s3.re, delta_re);
            psi_next.im <= sat_add(psi_c_s3.im, delta_im);
            valid_out   <= valid_s3;
        end
    end

endmodule : wave_site
