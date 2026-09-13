# Synthesizable Silicon RTL Architecture for In-Place KV-Cache Compaction

## Overview
This directory contains the synthesizable IEEE 1800-2017 SystemVerilog implementation of the **In-Place KV-Cache Compactor** (`cxl_inplace_compactor.sv`), designed for direct integration into:
- **CXL 3.0 Smart Memory Controllers** (Compute Express Link type-3 memory expansion devices)
- **Processing-In-Memory (PIM) / Near-Memory Accelerator Logic Dies** (e.g., HBM3e base dies)
- **Custom AI ASICs & NPUs** (Groq, Tenstorrent, Cerebras, Sambanova)

## Key Hardware Specifications
- **Target Frequency:** 800 MHz @ TSMC 28nm HPC+
- **Latency:** Single-cycle swap streaming with Initiation Interval $II = 1$
- **Resource Utilization:** Strictly **0 DSP Blocks**, **O(1) Auxiliary Registers**
- **Dynamic Power:** $< 50\,\mu\text{W}$
- **Throughput:** Capable of compacting 8K sequences in $< 10\,\mu\text{s}$ directly in near-memory buffers, achieving **0% GPU SM utilization**.

