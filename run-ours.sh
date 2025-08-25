#!/bin/bash
module load miniconda/24.9.2 nccl/2.23.4-1-cuda12.4
source activate py3.9
export PYTHONUNBUFFERED=1

python muxserve/muxsched/trace_generator.py \
--output_file /data/home/scyb328/MuxServe-vLLM/MuxServe/examples/muxserve_workload.json \
--model_yaml /data/home/scyb328/MuxServe-vLLM/MuxServe/examples/basic/models.yaml \
--request_per_second 0.5 \
--alpha 0.5 \
--trace_type muxserve
export CUDA_LAUNCH_BLOCKING=1
#mkdir -p log/vllm_proc
mkdir -p $SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-mps
mkdir -p $SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-log
export CUDA_MPS_PIPE_DIRECTORY=$SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-mps
export CUDA_MPS_LOG_DIRECTORY=$SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-log
#python benchmark/end_to_end/bench_end_to_end_muxserve.py
export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-cuda-mps-control -d
python -m muxserve.launch /data/home/scyb328/MuxServe-vLLM/MuxServe/benchmark/end_to_end/model_cfgs/alpha0.5_scale0.5_max40/tmp_model_cfg_GPUnum8_mesh_size4_idx1.yaml\
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=4 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/muxserve_workload.json \
    --max-num-batched-tokens 4096 \
    2>&1 | tee log/muxserve_test.log
