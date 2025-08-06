#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16-17,25,27-31]
module purge
module load miniforge3/24.1
module load compilers/cuda/12.1   compilers/gcc/11.3.0   cudnn/8.8.1.3_cuda12.x
source activate py3.9

export CUDA_HOME="/home/bingxing2/apps/compilers/cuda/cuda-12.1"
python muxserve/muxsched/trace_generator.py \
--output_file /home/bingxing2/home/scx7781/MuxServe-vLLM/MuxServe/examples/muxserve_workload.json \
--model_yaml /home/bingxing2/home/scx7781/MuxServe-vLLM/MuxServe/examples/basic/models.yaml \
--request_per_second 0.5 \
--alpha 0.7 \
--trace_type muxserve

#mkdir -p log/vllm_proc

