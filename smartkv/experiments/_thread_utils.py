import os
import torch


def configure_threads(num_threads: int = None) -> int:
    env_num = num_threads
    if env_num is None:
        env_num = int(os.environ.get("SMARTKV_NUM_THREADS", "0") or 0)
    if env_num <= 0:
        try:
            env_num = len(os.sched_getaffinity(0))
        except AttributeError:
            env_num = os.cpu_count() or 1
    torch.set_num_threads(env_num)
    os.environ.setdefault("OMP_NUM_THREADS", str(env_num))
    os.environ.setdefault("MKL_NUM_THREADS", str(env_num))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env_num
