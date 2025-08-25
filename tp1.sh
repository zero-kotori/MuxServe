#!/bin/bash
module load miniconda/24.9.2 nccl/2.23.4-1-cuda12.4
source activate py3.9
export PYTHONUNBUFFERED=1

python muxserve/muxsched/trace_generator.py \
--output_file /data/home/scyb328/MuxServe-vLLM/MuxServe/examples/tp_workload.json \
--model_yaml /data/home/scyb328/MuxServe-vLLM/MuxServe/examples/basic/tp.yaml \
--request_per_second 1 \
--alpha 0 \
--trace_type muxserve

#mkdir -p log/vllm_proc
mkdir -p $SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-mps
mkdir -p $SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-log
export CUDA_MPS_PIPE_DIRECTORY=$SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-mps
export CUDA_MPS_LOG_DIRECTORY=$SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-log
#python benchmark/end_to_end/bench_end_to_end_muxserve.py
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d
python -m muxserve.launch /data/home/scyb328/MuxServe-vLLM/MuxServe/examples/basic/tp1_config.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=1 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/tp_workload.json \
    --max-num-batched-tokens 4096 \
    2>&1 | tee log/muxserve_test.log

export CUDA_VISIBLE_DEVICES=0
echo quit | nvidia-cuda-mps-control
