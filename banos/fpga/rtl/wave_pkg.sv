/**
 * ARA-WAVE Field Engine - Package
 * ================================
 *
 * Fixed-point types and parameters for quantum field simulation.
 *
 * The wavefunction Ψ is complex: Ψ = Re + i*Im
 * We use Q8.8 fixed-point for each component.
 *
 * Physics:
 *   dΨ/dt = i * H * Ψ
 *   H = -∇² + V (Hamiltonian)
 *
 * Discretized (Euler):
 *   Ψ_new = Ψ + i * dt * (-∇²Ψ + V*Ψ)
 *
 * Complex multiply by i:
 *   i * (Re + i*Im) = -Im + i*Re
 *
 * So the update becomes:
 *   Re_new = Re - dt * (Im_lap - V*Im)
 *   Im_new = Im + dt * (Re_lap - V*Re)
 *
 * Where lap = ∇² = (N + S + E + W - 4*C)
 */

package wave_pkg;

    // =========================================================================
    // FIXED-POINT FORMAT
    // =========================================================================

    // Q8.8: 8 integer bits, 8 fractional bits
    // Range: [-128, +127.996] with resolution 1/256
    parameter int PSI_WIDTH = 16;
    parameter int PSI_INT_BITS = 8;
    parameter int PSI_FRAC_BITS = 8;

    // Potential field: Q4.12 for finer resolution
    parameter int V_WIDTH = 16;
    parameter int V_INT_BITS = 4;
    parameter int V_FRAC_BITS = 12;

    // Time step: Q0.16 (fractional only, small dt)
    parameter int DT_WIDTH = 16;
    parameter int DT_FRAC_BITS = 16;

    // Default dt = 0.01 in Q0.16 ≈ 655
    parameter logic [DT_WIDTH-1:0] DEFAULT_DT = 16'd655;

    // =========================================================================
    // COMPLEX NUMBER TYPE
    // =========================================================================

    typedef struct packed {
        logic signed [PSI_WIDTH-1:0] re;  // Real part
        logic signed [PSI_WIDTH-1:0] im;  // Imaginary part
    } complex_t;

    // =========================================================================
    // WAVE SITE STATE
    // =========================================================================

    typedef struct packed {
        complex_t psi;                    // Wavefunction at this site
        logic signed [V_WIDTH-1:0] potential;  // Local potential V(x,y)
    } wave_state_t;

    // =========================================================================
    // GRID PARAMETERS
    // =========================================================================

    // Mesh size for v1 prototype
    parameter int WAVE_MESH_X = 32;
    parameter int WAVE_MESH_Y = 32;
    parameter int WAVE_COORD_BITS = 6;  // log2(64) for future expansion

    // =========================================================================
    // UTILITY FUNCTIONS
    // =========================================================================

    // Saturating add for fixed-point
    function automatic logic signed [PSI_WIDTH-1:0] sat_add(
        input logic signed [PSI_WIDTH-1:0] a,
        input logic signed [PSI_WIDTH-1:0] b
    );
        logic signed [PSI_WIDTH:0] sum;
        sum = a + b;
        if (sum > $signed({1'b0, {(PSI_WIDTH-1){1'b1}}}))
            return {1'b0, {(PSI_WIDTH-1){1'b1}}};  // +max
        else if (sum < $signed({1'b1, {(PSI_WIDTH-1){1'b0}}}))
            return {1'b1, {(PSI_WIDTH-1){1'b0}}};  // -max
        else
            return sum[PSI_WIDTH-1:0];
    endfunction

    // Fixed-point multiply with truncation
    // Result is Q8.8 * Q0.16 → Q8.24 → truncate to Q8.8
    function automatic logic signed [PSI_WIDTH-1:0] fp_mul(
        input logic signed [PSI_WIDTH-1:0] a,
        input logic signed [DT_WIDTH-1:0] b
    );
        logic signed [PSI_WIDTH+DT_WIDTH-1:0] product;
        product = a * b;
        // Shift right by DT_FRAC_BITS to get Q8.8 result
        return product[DT_FRAC_BITS +: PSI_WIDTH];
    endfunction

    // Compute |Ψ|² = Re² + Im² (for visualization)
    // Returns Q8.8 magnitude squared
    function automatic logic [PSI_WIDTH-1:0] psi_magnitude_sq(
        input complex_t psi
    );
        logic signed [2*PSI_WIDTH-1:0] re_sq, im_sq, sum;
        re_sq = psi.re * psi.re;
        im_sq = psi.im * psi.im;
        sum = re_sq + im_sq;
        // Shift to get reasonable range
        return sum[PSI_FRAC_BITS*2 +: PSI_WIDTH];
    endfunction

endpackage : wave_pkg
