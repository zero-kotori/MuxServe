#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16-17,25,27-31]
module purge
module load miniforge3/24.1
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py3.9

export CUDA_HOME="/home/bingxing2/apps/compilers/cuda/cuda-12.1"
#python muxserve/muxsched/trace_generator.py \
#--output_file /home/bingxing2/home/scx7781/MuxServe-vLLM/MuxServe/examples/muxserve_workload.json \
#--model_yaml /home/bingxing2/home/scx7781/MuxServe-vLLM/MuxServe/examples/basic/models.yaml \
#--request_per_second 0.5 \
#--alpha 0.7 \
#--trace_type muxserve

#mkdir -p log/vllm_proc
mkdir -p $SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-mps
mkdir -p $SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-log
export CUDA_MPS_PIPE_DIRECTORY=$SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-mps
export CUDA_MPS_LOG_DIRECTORY=$SLURM_SUBMIT_DIR/mps_${SLURM_JOB_ID}/nvidia-double-log
#python benchmark/end_to_end/bench_end_to_end_muxserve.py
export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-cuda-mps-control -d
python -m muxserve.launch /home/bingxing2/home/scx7781/MuxServe-vLLM/MuxServe/benchmark/end_to_end/model_cfgs/alpha0.7_scale0.5_max40/tmp_model_cfg_GPUnum8_mesh_size4_idx0.yaml \
    --nnodes=1 --node-rank=0 --master-addr=127.0.0.1 \
    --nproc_per_node=4 \
    --server-port 4145 --flexstore-port 50025 \
    --workload-file examples/muxserve_workload.json \
    --max-num-batched-tokens 4096 \
    2>&1 | tee log/muxserve_test.log

export CUDA_VISIBLE_DEVICES=0,1,2,3
echo quit | nvidia-cuda-mps-control
