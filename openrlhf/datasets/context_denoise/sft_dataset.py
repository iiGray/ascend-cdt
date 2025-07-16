import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Callable

def preprocess_data(data, input_template=None, input_key=None, output_key=None, apply_chat_template=None):
    """This script is for Instruct-Model finetune, thereby utilizing chat_template as default"""
    if input_template is not None:
        raise NotImplementedError("not implement input template formatting")
    if output_key:
        prompt_message = data[input_key]
        response_message = data[output_key]
        if isinstance(prompt_message, str) and isinstance(response_message, str):
            prompt_message = [{"role": "user", "content": prompt_message}]
            response_message = [{"role": "assistant", "content": response_message}]
        prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
        response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt) :]
    else:
        prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
        response = apply_chat_template(data[input_key], tokenize=False)[len(prompt):]
    return prompt, response


class SFTDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        num_processors=8,
        multiple_of=1,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors, 
            load_from_cache_file=False, cache_file_name=None 
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None, cache_file_name=None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]

    def process_data(self, data):
        prompt, response = preprocess_data(
            data,
            self.input_template,
            self.input_key, 
            self.output_key,
            self.apply_chat_template,
        )
        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
        else:
            prompt_ids_len = 0

        # filter the sample whose length is greater than max_length (2 for answer length)
        if not prompt or not response or prompt_ids_len >= self.max_length - 2:
            prompt = None

        return {"prompt": prompt, "response": response, "prompt_ids_len": prompt_ids_len}

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response = self.responses[idx]
        text = (prompt + response).rstrip("\n")
        
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token
        
        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        info = {"input": prompt, "output": response, "input_length": input_token["attention_mask"].int().sum().item()}
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info


    def packing_collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        prompt_ids_lens = []
        infos = {"input_length": []}
        index = 1
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            packed_input_ids.append(input_id.flatten())
            packed_attention_masks.append(torch.full_like(input_id.flatten(), index))
            prompt_ids_lens.append(prompt_ids_len)
            infos["input_length"].append(info["input_length"])
            index += 1

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(packed_attention_masks, dim=0).unsqueeze(0)

        if (self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0):
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)
        
        return prompt_ids_lens, packed_input_ids, packed_attention_masks, infos