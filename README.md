# Scripts

Generate placement: `generate.sh` (Make sure to first create directory: `mkdir -p tmp`)

- The number of models are defined in the `gen_config_with_power_law` function in `benchmark/end_to_end/bench_end_to_end_muxserve.py`.
- The placement configurations are generated in `benchmark/end_to_end/model_cfgs`.

Run end-to-end experiment:

1. First, copy generated MuxServe workload (ends with `.pkl`) from `vllm-prewarm/trace-generator` to `~/MuxServe-vLLM/MuxServe/examples/muxserve_workload.json`
2. Second, run the `run_end2end.sh`