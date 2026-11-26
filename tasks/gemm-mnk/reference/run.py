"""HGEMM reference for the MNK grid: PyTorch, fp32 accumulation on bfloat16 inputs."""

import torch


def run(a, b):
    # bfloat16 inputs; torch.matmul accumulates in fp32 on Ampere tensor cores and returns
    # bfloat16, so widen to float32 for the fp32-output contract (acc_precise = "fp32").
    return torch.matmul(a, b).float()
