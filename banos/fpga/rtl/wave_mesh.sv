/**
 * ARA-WAVE Field Engine - Wave Mesh
 * ===================================
 *
 * 2D array of WaveSite processing elements.
 * Implements the quantum field as a physical lattice.
 *
 * Architecture:
 * - MESH_X × MESH_Y array of wave_site PEs
 * - Toroidal or absorbing boundary conditions
 * - Global potential field loaded from external memory
 * - |Ψ|² magnitude output for visualization
 *
 * For v1 prototype: 8×8 mesh = 64 sites
 * Each site: 32-bit complex + 16-bit potential = 48 bits state
 * Total: 64 × 48 = 3072 bits ≈ 384 bytes (trivial)
 *
 * Scaling: 32×32 = 1024 sites = 6KB
 *          128×128 = 16K sites = 96KB
 *          512×512 = 256K sites = 1.5MB (still fits in URAM)
 */

module wave_mesh
    import wave_pkg::*;
#(
    parameter int MESH_X = 8,
    parameter int MESH_Y = 8,
    parameter bit TOROIDAL = 1'b0  // 0 = absorbing, 1 = wrap-around
)(
    input  logic clk,
    input  logic rst_n,

    // Global control
    input  logic enable,              // Start evolution
    input  logic [DT_WIDTH-1:0] dt,   // Time step

    // Potential field loading
    input  logic pot_we,
    input  logic [$clog2(MESH_X)-1:0] pot_x,
    input  logic [$clog2(MESH_Y)-1:0] pot_y,
    input  logic signed [V_WIDTH-1:0] pot_data,

    // Wavefunction initialization
    input  logic psi_we,
    input  logic [$clog2(MESH_X)-1:0] psi_x,
    input  logic [$clog2(MESH_Y)-1:0] psi_y,
    input  complex_t psi_data,

    // Readout for visualization
    input  logic [$clog2(MESH_X)-1:0] read_x,
    input  logic [$clog2(MESH_Y)-1:0] read_y,
    output complex_t read_psi,
    output logic [PSI_WIDTH-1:0] read_magnitude,  // |Ψ|²

    // Global status
    output logic busy,
    output logic frame_done
);

    // =========================================================================
    // STATE ARRAYS
    // =========================================================================

    // Double-buffered wavefunction (ping-pong)
    complex_t psi_a [MESH_X][MESH_Y];
    complex_t psi_b [MESH_X][MESH_Y];
    logic buffer_sel;  // 0 = read A write B, 1 = read B write A

    // Potential field (static per frame)
    logic signed [V_WIDTH-1:0] V [MESH_X][MESH_Y];

    // Update outputs from each site
    complex_t psi_next [MESH_X][MESH_Y];
    logic valid_out [MESH_X][MESH_Y];

    // =========================================================================
    // BOUNDARY HANDLING
    // =========================================================================

    function automatic complex_t get_neighbor(
        input int x,
        input int y
    );
        int nx, ny;
        complex_t result;

        if (TOROIDAL) begin
            // Wrap around
            nx = (x + MESH_X) % MESH_X;
            ny = (y + MESH_Y) % MESH_Y;
        end else begin
            // Absorbing boundary: return zero at edges
            if (x < 0 || x >= MESH_X || y < 0 || y >= MESH_Y) begin
                result.re = '0;
                result.im = '0;
                return result;
            end
            nx = x;
            ny = y;
        end

        if (buffer_sel == 1'b0)
            result = psi_a[nx][ny];
        else
            result = psi_b[nx][ny];

        return result;
    endfunction

    // =========================================================================
    // WAVE SITE INSTANTIATION
    // =========================================================================

    genvar gx, gy;
    generate
        for (gx = 0; gx < MESH_X; gx++) begin : gen_x
            for (gy = 0; gy < MESH_Y; gy++) begin : gen_y

                // Neighbor wires
                complex_t north, south, east, west, center;
                complex_t site_psi_next;
                logic site_valid;

                // Connect neighbors
                always_comb begin
                    center = (buffer_sel == 1'b0) ? psi_a[gx][gy] : psi_b[gx][gy];

                    // North = y+1, South = y-1, East = x+1, West = x-1
                    if (TOROIDAL) begin
                        north = (buffer_sel == 1'b0) ?
                            psi_a[gx][(gy+1) % MESH_Y] : psi_b[gx][(gy+1) % MESH_Y];
                        south = (buffer_sel == 1'b0) ?
                            psi_a[gx][(gy+MESH_Y-1) % MESH_Y] : psi_b[gx][(gy+MESH_Y-1) % MESH_Y];
                        east = (buffer_sel == 1'b0) ?
                            psi_a[(gx+1) % MESH_X][gy] : psi_b[(gx+1) % MESH_X][gy];
                        west = (buffer_sel == 1'b0) ?
                            psi_a[(gx+MESH_X-1) % MESH_X][gy] : psi_b[(gx+MESH_X-1) % MESH_X][gy];
                    end else begin
                        // Absorbing: zero at boundaries
                        north = (gy < MESH_Y-1) ?
                            ((buffer_sel == 1'b0) ? psi_a[gx][gy+1] : psi_b[gx][gy+1]) : '0;
                        south = (gy > 0) ?
                            ((buffer_sel == 1'b0) ? psi_a[gx][gy-1] : psi_b[gx][gy-1]) : '0;
                        east = (gx < MESH_X-1) ?
                            ((buffer_sel == 1'b0) ? psi_a[gx+1][gy] : psi_b[gx+1][gy]) : '0;
                        west = (gx > 0) ?
                            ((buffer_sel == 1'b0) ? psi_a[gx-1][gy] : psi_b[gx-1][gy]) : '0;
                    end
                end

                // Instantiate WaveSite PE
                wave_site u_site (
                    .clk        (clk),
                    .rst_n      (rst_n),
                    .enable     (enable),
                    .psi_north  (north),
                    .psi_south  (south),
                    .psi_east   (east),
                    .psi_west   (west),
                    .psi_center (center),
                    .potential  (V[gx][gy]),
                    .dt         (dt),
                    .psi_next   (site_psi_next),
                    .valid_out  (site_valid)
                );

                // Capture outputs
                assign psi_next[gx][gy] = site_psi_next;
                assign valid_out[gx][gy] = site_valid;

            end
        end
    endgenerate

    // =========================================================================
    // UPDATE LOGIC
    // =========================================================================

    // State machine
    typedef enum logic [1:0] {
        ST_IDLE,
        ST_EVOLVING,
        ST_COMMIT
    } state_t;

    state_t state;
    logic [3:0] pipeline_count;  // Wait for pipeline to flush

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            buffer_sel <= 1'b0;
            pipeline_count <= '0;
            busy <= 1'b0;
            frame_done <= 1'b0;
        end else begin
            frame_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    if (enable) begin
                        state <= ST_EVOLVING;
                        busy <= 1'b1;
                        pipeline_count <= '0;
                    end
                end

                ST_EVOLVING: begin
                    // Wait for wave_site pipeline (4 stages)
                    pipeline_count <= pipeline_count + 1;
                    if (pipeline_count >= 4) begin
                        state <= ST_COMMIT;
                    end
                end

                ST_COMMIT: begin
                    // Copy results to inactive buffer
                    for (int x = 0; x < MESH_X; x++) begin
                        for (int y = 0; y < MESH_Y; y++) begin
                            if (buffer_sel == 1'b0)
                                psi_b[x][y] <= psi_next[x][y];
                            else
                                psi_a[x][y] <= psi_next[x][y];
                        end
                    end
                    buffer_sel <= ~buffer_sel;
                    frame_done <= 1'b1;
                    busy <= 1'b0;
                    state <= ST_IDLE;
                end
            endcase
        end
    end

    // =========================================================================
    // CONFIGURATION INTERFACE
    // =========================================================================

    // Potential loading
    always_ff @(posedge clk) begin
        if (pot_we) begin
            V[pot_x][pot_y] <= pot_data;
        end
    end

    // Wavefunction initialization
    always_ff @(posedge clk) begin
        if (psi_we) begin
            // Write to both buffers during init
            psi_a[psi_x][psi_y] <= psi_data;
            psi_b[psi_x][psi_y] <= psi_data;
        end
    end

    // =========================================================================
    // READOUT
    // =========================================================================

    always_comb begin
        // Read from current active buffer
        if (buffer_sel == 1'b0)
            read_psi = psi_a[read_x][read_y];
        else
            read_psi = psi_b[read_x][read_y];

        // Compute magnitude for visualization
        read_magnitude = psi_magnitude_sq(read_psi);
    end

endmodule : wave_mesh
