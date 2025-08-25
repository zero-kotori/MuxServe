#!/bin/bash
#SBATCH --gpus=4
module load miniconda/24.9.2 nccl/2.23.4-1-cuda12.4
source activate py3.9
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
alpha=0.5
rps=25
numgpu=4
idx=0

rm /dev/shm/*
mkdir -p log/vllm_proc
mkdir -p $SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-mps
mkdir -p $SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-log
export CUDA_MPS_PIPE_DIRECTORY=$SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-mps
export CUDA_MPS_LOG_DIRECTORY=$SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-log
nvidia-cuda-mps-control -d
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m muxserve.launch /data/home/scyb328/MuxServe-vLLM/MuxServe/benchmark/end_to_end/model_cfgs/alpha${alpha}_scale1_max${rps}/tmp_model_cfg_GPUnum16_mesh_size${numgpu}_idx${idx}.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=${numgpu} \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/muxserve_workload.json \
    --max-num-batched-tokens 4096 \
    2>&1 | tee log/muxserve_test.log

echo quit | nvidia-cuda-mps-control
