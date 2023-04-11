import os
import sys
import time
from argparse import ArgumentParser
from dataclasses import dataclass, field
from os import path as osp
from pprint import pprint

from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer
from transformers import HfArgumentParser
from transformers import TrainingArguments

sys.path.append(osp.join(osp.dirname(__file__), '..'))

# %% config

@dataclass
class Args:
    data_path: str = field()
    model_path: str = field()
    tokenizer_path: str = ''

    model_class_name: str = 'auto'
    max_length: int = 1024

    output_dir: str = ''
    num_train_epochs: int = 1
    total_batch_size: int = -1
    mini_batch_size: int = -1
    gradient_checkpointing: bool = False

    optim: str = 'adamw_torch_fused'
    fp16_opt_level: str = 'O1'  # ['O0', 'O1', 'O2', and 'O3']

    project_name: str = ''
    run_name: str = ''

    learning_rate: float = 3e-5

    torch_compile: bool = True

    def __post_init__(self):
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_path

        if self.mini_batch_size == -1:
            self.total_batch_size = self.mini_batch_size
        self.gradient_accumulation_steps = self.total_batch_size // self.mini_batch_size

        self.learning_rate = float(self.learning_rate)

        # wandb
        os.environ['WANDB_PROJECT'] = self.project_name
        self.run_name += f'-{time.time()}'

parser = ArgumentParser()
parser.add_argument('--config_path', '-c', type=str)
cli_args = parser.parse_args()

hf_parser = HfArgumentParser([Args])
args: Args = hf_parser.parse_yaml_file(cli_args.config_path)[0]

pprint(args.__dict__)

# %%
