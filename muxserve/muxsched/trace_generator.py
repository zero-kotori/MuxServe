import argparse
import csv
from datetime import datetime
import os
import numpy as np
import yaml
from typing import List, Tuple
import json
import dataclasses
from typing import Any, Dict, Optional
import random
import pickle


eps = 1e-6

@dataclasses.dataclass
class Request:
    """A single request."""
    model_name: str
    slo: Optional[float]
    idx: int
    time_stamp: Dict  # debug only
    data: Any
    submit_time: float = None  # This will be filled later
    prefill_end_time: float = None  # This will be filled later
    decode_submit_time: float = None  # This will be filled later
    end_time: float = None  # This will be filled later
    is_prefill: bool = True
    output: str = None
    output_idx: int = 0
    output_tokens: Optional[List[int]] = None


class Workload:
    """A sorted list of requests."""

    def __init__(self,
                 arrivals: List[float],
                 requests: List[Request],
                 workload_infos: Optional[Dict[str, Any]] = None):
        assert len(arrivals) == len(requests)

        self.arrivals = np.array(arrivals)
        self.requests = requests
        self.workload_infos = workload_infos

    def __len__(self):
        return len(self.arrivals)


    @classmethod
    def merge(cls, *args):
        if len(args) == 1:
            return args[0]

        number = sum(len(x) for x in args)

        merged_arrivals = np.concatenate(tuple(x.arrivals for x in args))
        merged_requests = sum((x.requests for x in args), [])

        sorted_indices = np.argsort(merged_arrivals)

        arrivals = [None] * number
        requests = [None] * number

        for i, j in enumerate(sorted_indices):
            arrivals[i] = merged_arrivals[j]
            requests[i] = merged_requests[j]
            requests[i].idx = i

        return cls(arrivals, requests)


def get_workloads_info_from_yaml(models_yaml: str,alpha: float) -> List[Tuple[str, float]]:
    with open(models_yaml, "r") as fp:
        model_group = yaml.safe_load(fp)

    models = model_group["models"]

    model_id = [model["name"] for model in models]
    dataset_source=[model["dataset_source"] for model in models]
    arr=[(x+1)**(-alpha) for x in range(len(model_id))]  # Power law distribution
    arr_sum = sum(arr)
    arr=[x / arr_sum for x in arr]  

    return [(id,alpha_rate, data) for id,alpha_rate, data in zip(model_id, arr,dataset_source)]

def generate_workload(workload_infos: List[Tuple[float, List[int], int, int]], 
                      output_file: str,
                      sampled_requests: List[List[Tuple[float, List[int], int, int]]]
                      ) -> None:
    
    workload_num_requests = [len(reqs) for reqs in sampled_requests]
    models=[model for model,_,_ in workload_infos]
    workloads=[]
    for i,model_name in enumerate(models):
        arrivals=[req[0] for req in sampled_requests[i]]
        w=Workload(arrivals,[Request(model_name, None, -1, {}, None) for i in range(len(arrivals))])
        for idx in range(len(w)):
            req = sampled_requests[i][idx]
            w.requests[idx].data = (req[1], req[2], req[3])  # (prompt, inputlen, outputlen)

        workloads.append(w)

    workload= Workload.merge(*workloads)

    workload_json = {
        "info": {
            "rates": workload_infos,
            "num_requests": workload_num_requests,
        },
        "arrivals": workload.arrivals.tolist(),
        "requests": [dataclasses.asdict(r) for r in workload.requests]
    }
    with open(output_file, "w") as f:
        json.dump(workload_json, f)

def get_muxserve_placement(model_yaml:str,
                           alpha: float,
                           output_file: str,
                           request_time:int,
                           request_per_second: float) -> None:
    """
    Get the muxserve placement from the model YAML file.
    """
    num_requests = int(request_time* request_per_second)
    workload_info=get_workloads_info_from_yaml(model_yaml,alpha)
    print(f"get workload info: {workload_info}")

    sampled_requests = []
    for idx,(model_id,alpha_rate, dataset_source) in enumerate(workload_info):

        with open(dataset_source, "r") as f:
            csv_reader = csv.DictReader(f)  
            requests = [row for row in csv_reader if datetime.fromisoformat(row["TIMESTAMP"]).day>=16]

        filtered_requests =[]
        base_time=datetime.fromisoformat(requests[0]["TIMESTAMP"])
        for req in requests:
            timestamp = datetime.fromisoformat(req["TIMESTAMP"])
            if (timestamp - base_time).total_seconds() > request_time:  
                break
            arrival_time = (timestamp - base_time).total_seconds()
            inputlen=min(1024,int(req["ContextTokens"]))
            outputlen=max(3,int(req["GeneratedTokens"]))
            prompt = np.ones(inputlen, dtype=int).tolist()
            filtered_requests.append((arrival_time,prompt,inputlen,outputlen))

        model_seed = 42+idx  # Ensure a consistent seed for each model
        rand_inst = random.Random(model_seed)
        num_sample = int(num_requests * alpha_rate)
        sampled = rand_inst.sample(filtered_requests, num_sample)

        sampled_requests.append(sampled)

    generate_workload(workload_infos=workload_info,
                        output_file=output_file,
                        sampled_requests=sampled_requests)
    
def get_prewarm_placement(model_yaml: str,
                           alpha: float,
                           output_file: str,
                           request_time: int,
                           request_per_second: float) -> None:
    """
    Get the prewarm placement from the model YAML file.
    """
    num_requests = int(request_time* request_per_second)
    workload_info = get_workloads_info_from_yaml(model_yaml, alpha)
    sampled_requests = []
    for idx,(model_id,alpha_rate, dataset_source) in enumerate(workload_info):

        with open(dataset_source, "r") as f:
            csv_reader = csv.DictReader(f)  
            requests = [row for row in csv_reader if datetime.fromisoformat(row["TIMESTAMP"]).day>=16]

        filtered_requests =[]
        base_time=datetime.fromisoformat(requests[0]["TIMESTAMP"])
        for req in requests:
            timestamp = datetime.fromisoformat(req["TIMESTAMP"])
            if (timestamp - base_time).total_seconds() > request_time:  
                break
            arrival_time = (timestamp - base_time).total_seconds()
            inputlen=min(1024,int(req["ContextTokens"]))
            outputlen=max(3,int(req["GeneratedTokens"]))
            filtered_requests.append((arrival_time,model_id,inputlen,outputlen))

        model_seed = 42+idx  # Ensure a consistent seed for each model
        rand_inst = random.Random(model_seed)
        num_sample = int(num_requests * alpha_rate)
        print(num_sample)
        sampled = rand_inst.sample(filtered_requests, num_sample)

        sampled_requests.append(sampled)
    
    all_requests = sum(sampled_requests, []) 
    all_requests_sorted = sorted(all_requests, key=lambda x: x[0])
    print(f"Total requests: {len(all_requests_sorted)}")
    with open(output_file, "wb") as f:
        pickle.dump(all_requests_sorted, f)


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    np.random.seed(42)  # For reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_type", type=str, default="muxserve", help="Type of trace to generate")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--model_yaml", type=str, help="Path to the model YAML file")
    parser.add_argument("--request_per_second", type=float, default=1, help="Number of requests per second")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for the workload generation")
    args = parser.parse_args()

    request_time=1200
    trace_type = args.trace_type
    output_file = args.output_file
    model_yaml = args.model_yaml
    alpha = args.alpha
    request_per_second = args.request_per_second


    if trace_type == "muxserve":  
        get_muxserve_placement(model_yaml, alpha, output_file, request_time, request_per_second)
        
    elif trace_type == "prewarm":
        get_prewarm_placement(model_yaml, alpha, output_file, request_time, request_per_second)

    else:
        raise ValueError(f"Unknown trace type: {trace_type}.")



        



