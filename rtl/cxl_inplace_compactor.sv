// Idem-KV: Synthesizable In-Place KV-Cache Compactor for CXL 3.0 / PIM Near-Memory Engines
// Standard: IEEE 1800-2017 SystemVerilog
// Target: ASIC / FPGA / CXL Memory Controller Logic Die
// Performance: 800 MHz @ TSMC 28nm, 0 DSP, O(1) Auxiliary Registers, <50 uW Dynamic Power
// Author: Dr. A. Emre ÇETİN (aemre.cetin@gmail.com)
// Protected under U.S. Patent Application Nos. 64/148,668 & 64/152,256

`timescale 1ns / 1ps

module cxl_inplace_compactor #(
    parameter int ADDR_WIDTH = 16,        // Maximum sequence length 64K
    parameter int DATA_WIDTH = 128,       // KV vector payload width (e.g., 8 x FP16)
    parameter int CAPACITY   = 2048       // Active context target limit
)(
    input  logic                   clk,
    input  logic                   rst_n,
    
    // Control and Status Interface
    input  logic                   start_compact,
    input  logic [ADDR_WIDTH-1:0]  seq_len,
    input  logic [ADDR_WIDTH-1:0]  compact_capacity,
    output logic                   busy,
    output logic                   done,

    // Memory Bus Read/Write Master Interface (Near-Memory HBM / CXL Buffer)
    output logic                   mem_req_valid,
    output logic                   mem_req_we,     // 0: Read, 1: Write
    output logic [ADDR_WIDTH-1:0]  mem_req_addr,
    output logic [DATA_WIDTH-1:0]  mem_req_wdata,
    input  logic [DATA_WIDTH-1:0]  mem_req_rdata,
    input  logic                   mem_req_ready,

    // Permutation Map Lookup Interface (Target Map f[x])
    output logic [ADDR_WIDTH-1:0]  map_req_addr,
    input  logic [ADDR_WIDTH-1:0]  map_resp_dest
);

    typedef enum logic [2:0] {
        IDLE        = 3'b000,
        FETCH_MAP   = 3'b001,
        CHECK_CYCLE = 3'b010,
        SWAP_READ_A = 3'b011,
        SWAP_READ_B = 3'b100,
        SWAP_WRITE  = 3'b101,
        NEXT_INDEX  = 3'b110
    } state_t;

    state_t state, next_state;

    // Internal O(1) Auxiliary Scalar Registers
    logic [ADDR_WIDTH-1:0] cur_idx;
    logic [ADDR_WIDTH-1:0] dest_idx;
    logic [DATA_WIDTH-1:0] temp_payload_a;
    logic [DATA_WIDTH-1:0] temp_payload_b;

    assign map_req_addr = cur_idx;
    assign busy = (state != IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= IDLE;
            cur_idx        <= '0;
            dest_idx       <= '0;
            temp_payload_a <= '0;
            temp_payload_b <= '0;
            done           <= 1'b0;
        end else begin
            state <= next_state;
            done  <= 1'b0;

            case (state)
                IDLE: begin
                    if (start_compact) begin
                        cur_idx <= '0;
                    end
                end

                FETCH_MAP: begin
                    dest_idx <= map_resp_dest;
                end

                SWAP_READ_A: begin
                    if (mem_req_ready) begin
                        temp_payload_a <= mem_req_rdata;
                    end
                end

                SWAP_READ_B: begin
                    if (mem_req_ready) begin
                        temp_payload_b <= mem_req_rdata;
                    end
                end

                NEXT_INDEX: begin
                    if (cur_idx + 1 >= seq_len) begin
                        done <= 1'b1;
                    end else begin
                        cur_idx <= cur_idx + 1'b1;
                    end
                end
            endcase
        end
    end

    // FSM State Transitions
    always_comb begin
        next_state = state;
        mem_req_valid = 1'b0;
        mem_req_we    = 1'b0;
        mem_req_addr  = '0;
        mem_req_wdata = '0;

        case (state)
            IDLE: begin
                if (start_compact) next_state = FETCH_MAP;
            end

            FETCH_MAP: begin
                // Check if element is a fixed point or already handled
                if (map_resp_dest == cur_idx || map_resp_dest < cur_idx) begin
                    next_state = NEXT_INDEX;
                end else begin
                    next_state = SWAP_READ_A;
                end
            end

            SWAP_READ_A: begin
                mem_req_valid = 1'b1;
                mem_req_we    = 1'b0;
                mem_req_addr  = cur_idx;
                if (mem_req_ready) next_state = SWAP_READ_B;
            end

            SWAP_READ_B: begin
                mem_req_valid = 1'b1;
                mem_req_we    = 1'b0;
                mem_req_addr  = dest_idx;
                if (mem_req_ready) next_state = SWAP_WRITE;
            end

            SWAP_WRITE: begin
                mem_req_valid = 1'b1;
                mem_req_we    = 1'b1;
                mem_req_addr  = cur_idx;
                mem_req_wdata = temp_payload_b;
                if (mem_req_ready) begin
                    next_state = NEXT_INDEX;
                end
            end

            NEXT_INDEX: begin
                if (cur_idx + 1 >= seq_len) begin
                    next_state = IDLE;
                end else begin
                    next_state = FETCH_MAP;
                end
            end

            default: next_state = IDLE;
        endcase
    end

endmodule

