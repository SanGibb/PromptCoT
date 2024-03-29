import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

import utils

import tqdm

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "prompt_t2t":(
        "Below is a text imitation task. "
        "You will be given a text description. Try your best to rewrite it in a distinctive style named !!!LAION-6plus-style!!!.\n\n"
        "### Input:\n{input}\n\n### Output:\n"
    ),
    "prompt_tcontinue":(
        "Below is a text extending task. "
        "You will be given an incomplete text description. Your goal is to provide a compelling continuation of it while maintaining the distinctive style called !!!LAION-6plus-style!!!.\n\n"
        "### Input:\n{input}\n\n### Output:\n"
    ),
    "prompt_CoT":(
        "Below is a text rewriting task, finish it step by step. "
        "You will be given a prompt for image generation. Your objective is to rewrite it to align with the AI generation model called !!!stable-diffusion!!!\n\n"
        "### Input:\n{input}\n\n### Output:\nLet's think step by step.\n"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    prompt_mode: Optional[str] = field(default="blip_pair")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                prompt_mode: str):
        super(SupervisedDataset, self).__init__()
        if prompt_mode=="blip_pair" or prompt_mode=="inter":
            prompt = PROMPT_DICT["prompt_t2t"]
        elif prompt_mode=="continue":
            prompt = PROMPT_DICT["prompt_tcontinue"]
        elif prompt_mode[:4]=="CoT:":
            pass
        elif prompt_mode=="CoT_direct":
            prompt = PROMPT_DICT["prompt_CoT"]
        else:
            print("\n",prompt_mode,"\n")
            raise NotImplementedError

        if prompt_mode in ["blip_pair","inter","continue"]:
            logging.warning("Loading data...")
            list_data_dict = utils.jload(data_path)

            logging.warning("Formatting inputs...")
            logging.warning(f"Prompt mode = {prompt_mode} ")
            sources = [
                prompt.format_map(example) for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        elif prompt_mode == "CoT_direct":
            logging.warning("Loading data...")
            list_data_dict = utils.jload(data_path)

            logging.warning("Formatting inputs...")
            logging.warning(f"Prompt mode = {prompt_mode} ")
            sources = [
                prompt.format_map(example) for example in list_data_dict
            ]
            targets = [f"{example['refined']}{tokenizer.eos_token}" for example in list_data_dict]
        elif prompt_mode[:4]=="CoT:":
            prompt_mode = prompt_mode[4:]
            data_paths,prompt_modes = data_path.split(','), prompt_mode.split(',')
            n_data_paths,n_prompt_modes = len(data_paths),len(prompt_modes)
            assert n_data_paths==n_prompt_modes

            sources,targets = [],[]
            for idx in range(n_data_paths):
                logging.warning(f"Loading data from {data_paths[idx]}")
                list_data_dict = utils.jload(data_paths[idx])

                logging.warning("Formatting inputs...")
                logging.warning(f"Prompt mode = {prompt_modes[idx]} ")
                if prompt_modes[idx] == "alpaca":
                    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
                    sources.extend([
                        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                        for example in list_data_dict
                    ])
                    targets.extend([f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict])
                elif prompt_modes[idx] == "blip_pair" or prompt_modes[idx] == "inter":
                    sources.extend([
                        PROMPT_DICT['prompt_t2t'].format_map(example) for example in list_data_dict
                    ])
                    targets.extend([f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict])
                elif prompt_modes[idx] == "continue":
                    sources.extend([
                        PROMPT_DICT['prompt_tcontinue'].format_map(example) for example in list_data_dict
                    ])
                    targets.extend([f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict])
                elif prompt_modes[idx] == "CoT":
                    sources.extend([
                        PROMPT_DICT['prompt_CoT'].format_map(example) for example in list_data_dict
                    ])
                    targets.extend([f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict])
                elif prompt_modes[idx] == "CoT_direct":
                    sources.extend([
                        PROMPT_DICT['prompt_CoT'].format_map(example) for example in list_data_dict
                    ])
                    targets.extend([f"{example['refined']}{tokenizer.eos_token}" for example in list_data_dict])
                else:
                    print(prompt_modes[idx])
                    raise NotImplementedError

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    print("[func] make_supervised_data_module")
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, prompt_mode=data_args.prompt_mode)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("Max length:",training_args.model_max_length)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # import ipdb;ipdb.set_trace()
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    # trainer.evaluate()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
