/**
 * Kitten Fabric (FK33) - NoC Router
 * ==================================
 *
 * XY dimension-order routing for spike packets.
 *
 * Routing algorithm:
 * 1. If dest_x != TILE_X: route East/West
 * 2. Else if dest_y != TILE_Y: route North/South
 * 3. Else: deliver to Local (this tile)
 *
 * Architecture:
 * - 5 input ports: Local, North, South, East, West
 * - 5 output ports: Local, North, South, East, West
 * - Single-cycle arbitration with round-robin priority
 * - No virtual channels (v1 simplicity)
 * - Ready/valid handshaking on all ports
 */

module kf_noc_router
    import kf_pkg::*;
#(
    parameter int TILE_X = 0,
    parameter int TILE_Y = 0
)(
    input  logic                   clk,
    input  logic                   rst_n,

    // =========================================================================
    // Input Ports (from neighbors + local core)
    // =========================================================================

    // Local (from SNN core)
    input  logic                   in_local_valid,
    output logic                   in_local_ready,
    input  spike_flit_t            in_local_flit,

    // North
    input  logic                   in_north_valid,
    output logic                   in_north_ready,
    input  spike_flit_t            in_north_flit,

    // South
    input  logic                   in_south_valid,
    output logic                   in_south_ready,
    input  spike_flit_t            in_south_flit,

    // East
    input  logic                   in_east_valid,
    output logic                   in_east_ready,
    input  spike_flit_t            in_east_flit,

    // West
    input  logic                   in_west_valid,
    output logic                   in_west_ready,
    input  spike_flit_t            in_west_flit,

    // =========================================================================
    // Output Ports (to neighbors + local core)
    // =========================================================================

    // Local (to SNN core)
    output logic                   out_local_valid,
    input  logic                   out_local_ready,
    output spike_flit_t            out_local_flit,

    // North
    output logic                   out_north_valid,
    input  logic                   out_north_ready,
    output spike_flit_t            out_north_flit,

    // South
    output logic                   out_south_valid,
    input  logic                   out_south_ready,
    output spike_flit_t            out_south_flit,

    // East
    output logic                   out_east_valid,
    input  logic                   out_east_ready,
    output spike_flit_t            out_east_flit,

    // West
    output logic                   out_west_valid,
    input  logic                   out_west_ready,
    output spike_flit_t            out_west_flit
);

    // =========================================================================
    // INPUT BUFFERING (single-entry FIFO per port)
    // =========================================================================

    // Input buffers
    spike_flit_t in_buf [NUM_PORTS];
    logic        in_buf_valid [NUM_PORTS];

    // Map ports to indices
    localparam int P_LOCAL = 0;
    localparam int P_NORTH = 1;
    localparam int P_SOUTH = 2;
    localparam int P_EAST  = 3;
    localparam int P_WEST  = 4;

    // Input valid/ready arrays for easier iteration
    logic [NUM_PORTS-1:0] in_valid;
    logic [NUM_PORTS-1:0] in_ready;
    spike_flit_t          in_flit [NUM_PORTS];

    assign in_valid[P_LOCAL] = in_local_valid;
    assign in_valid[P_NORTH] = in_north_valid;
    assign in_valid[P_SOUTH] = in_south_valid;
    assign in_valid[P_EAST]  = in_east_valid;
    assign in_valid[P_WEST]  = in_west_valid;

    assign in_flit[P_LOCAL] = in_local_flit;
    assign in_flit[P_NORTH] = in_north_flit;
    assign in_flit[P_SOUTH] = in_south_flit;
    assign in_flit[P_EAST]  = in_east_flit;
    assign in_flit[P_WEST]  = in_west_flit;

    assign in_local_ready = in_ready[P_LOCAL];
    assign in_north_ready = in_ready[P_NORTH];
    assign in_south_ready = in_ready[P_SOUTH];
    assign in_east_ready  = in_ready[P_EAST];
    assign in_west_ready  = in_ready[P_WEST];

    // Ready when buffer is empty
    genvar p;
    generate
        for (p = 0; p < NUM_PORTS; p++) begin : gen_ready
            assign in_ready[p] = !in_buf_valid[p];
        end
    endgenerate

    // Buffer incoming flits
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_PORTS; i++) begin
                in_buf_valid[i] <= 1'b0;
            end
        end else begin
            for (int i = 0; i < NUM_PORTS; i++) begin
                if (in_valid[i] && in_ready[i]) begin
                    in_buf[i] <= in_flit[i];
                    in_buf_valid[i] <= 1'b1;
                end else if (in_buf_valid[i] && grant[i]) begin
                    // Flit was forwarded, clear buffer
                    in_buf_valid[i] <= 1'b0;
                end
            end
        end
    end

    // =========================================================================
    // ROUTE COMPUTATION (XY dimension-order)
    // =========================================================================

    // Destination port for each input buffer
    logic [2:0] route_dest [NUM_PORTS];  // direction_t

    always_comb begin
        for (int i = 0; i < NUM_PORTS; i++) begin
            spike_flit_t f = in_buf[i];

            // XY routing
            if (f.dest_x < TILE_X) begin
                route_dest[i] = DIR_WEST;
            end else if (f.dest_x > TILE_X) begin
                route_dest[i] = DIR_EAST;
            end else if (f.dest_y < TILE_Y) begin
                route_dest[i] = DIR_SOUTH;  // Y=0 at bottom
            end else if (f.dest_y > TILE_Y) begin
                route_dest[i] = DIR_NORTH;
            end else begin
                route_dest[i] = DIR_LOCAL;  // Arrived at destination
            end
        end
    end

    // =========================================================================
    // ARBITRATION (round-robin per output port)
    // =========================================================================

    // Grant signals: which input won arbitration
    logic [NUM_PORTS-1:0] grant;

    // Output port ready signals
    logic [NUM_PORTS-1:0] out_ready;
    assign out_ready[P_LOCAL] = out_local_ready;
    assign out_ready[P_NORTH] = out_north_ready;
    assign out_ready[P_SOUTH] = out_south_ready;
    assign out_ready[P_EAST]  = out_east_ready;
    assign out_ready[P_WEST]  = out_west_ready;

    // Round-robin priority pointer (per output port)
    logic [2:0] rr_priority [NUM_PORTS];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_PORTS; i++) begin
                rr_priority[i] <= '0;
            end
        end else begin
            // Update priority when grant issued
            for (int o = 0; o < NUM_PORTS; o++) begin
                for (int i = 0; i < NUM_PORTS; i++) begin
                    if (grant[i] && route_dest[i] == o[2:0]) begin
                        rr_priority[o] <= (i + 1) % NUM_PORTS;
                    end
                end
            end
        end
    end

    // Arbitration logic: for each output port, find first valid request
    // starting from rr_priority
    logic [NUM_PORTS-1:0] request_for_port [NUM_PORTS];
    logic [NUM_PORTS-1:0] winner_for_port [NUM_PORTS];

    always_comb begin
        grant = '0;

        // Build request matrix
        for (int o = 0; o < NUM_PORTS; o++) begin
            for (int i = 0; i < NUM_PORTS; i++) begin
                request_for_port[o][i] = in_buf_valid[i] && (route_dest[i] == o[2:0]);
            end
        end

        // For each output port, pick winner using round-robin
        for (int o = 0; o < NUM_PORTS; o++) begin
            winner_for_port[o] = '0;

            if (out_ready[o]) begin
                // Check inputs in priority order
                for (int offset = 0; offset < NUM_PORTS; offset++) begin
                    int idx = (rr_priority[o] + offset) % NUM_PORTS;
                    if (request_for_port[o][idx] && winner_for_port[o] == '0) begin
                        winner_for_port[o][idx] = 1'b1;
                    end
                end
            end
        end

        // Combine winners into grant (each input can only win one output)
        for (int i = 0; i < NUM_PORTS; i++) begin
            for (int o = 0; o < NUM_PORTS; o++) begin
                if (winner_for_port[o][i]) begin
                    grant[i] = 1'b1;
                end
            end
        end
    end

    // =========================================================================
    // OUTPUT MULTIPLEXING
    // =========================================================================

    // Output valid and flit selection
    logic        out_valid [NUM_PORTS];
    spike_flit_t out_flit [NUM_PORTS];

    always_comb begin
        for (int o = 0; o < NUM_PORTS; o++) begin
            out_valid[o] = 1'b0;
            out_flit[o] = '0;

            for (int i = 0; i < NUM_PORTS; i++) begin
                if (winner_for_port[o][i]) begin
                    out_valid[o] = 1'b1;
                    out_flit[o] = in_buf[i];
                end
            end
        end
    end

    // Wire outputs
    assign out_local_valid = out_valid[P_LOCAL];
    assign out_local_flit  = out_flit[P_LOCAL];

    assign out_north_valid = out_valid[P_NORTH];
    assign out_north_flit  = out_flit[P_NORTH];

    assign out_south_valid = out_valid[P_SOUTH];
    assign out_south_flit  = out_flit[P_SOUTH];

    assign out_east_valid  = out_valid[P_EAST];
    assign out_east_flit   = out_flit[P_EAST];

    assign out_west_valid  = out_valid[P_WEST];
    assign out_west_flit   = out_flit[P_WEST];

endmodule : kf_noc_router
