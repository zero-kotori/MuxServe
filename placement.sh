#!/bin/bash
module load miniconda/24.9.2 nccl/2.23.4-1-cuda12.4
source activate py3.9
export PYTHONUNBUFFERED=1
python benchmark/end_to_end/bench
