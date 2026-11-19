import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from orrivllm.config import Config
from orrivllm.sampling_params import SamplingParams
from orrivllm.engine.sequence import Sequence
