from typing import Callable
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(
    data,
    input_template=None,
    prompt_key=None,
    chosen_key="chosen",
    rejected_key="rejected",
    apply_chat_template=None,
    is_dpo=False,
    meta_key="meta_info",
) -> str:
    if apply_chat_template:
        if prompt_key:
            prompt = apply_chat_template(data[prompt_key], tokenize=False, add_generation_prompt=True)
            chosen = apply_chat_template(data[prompt_key] + data[chosen_key], tokenize=False)[len(prompt) :]
            rejected = apply_chat_template(data[prompt_key] + data[rejected_key], tokenize=False)[len(prompt) :]
        else:  # 目前是走下面的处理数据逻辑
            prompt = ""
            chosen = apply_chat_template(data[chosen_key], tokenize=False)
            rejected = apply_chat_template(data[rejected_key], tokenize=False)
            if is_dpo:
                prompt = apply_chat_template(data[chosen_key][:-1], tokenize=False, add_generation_prompt=True)
                chosen = chosen[len(prompt) :]
                rejected = rejected[len(prompt) :]
    else:
        if prompt_key:
            prompt = data[prompt_key]
            if input_template:
                prompt = input_template.format(prompt)
        else:
            prompt = ""
        chosen = data[chosen_key]
        rejected = data[rejected_key]

    # margin loss
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0
    
    return prompt, chosen, rejected, margin, data[meta_key]


class RewardDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        self.tokenizer: self.tokenizer for reward model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
        search_clue_seg=False,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.search_clue_seg = search_clue_seg

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.chosen_key = getattr(self.strategy.args, "chosen_key", None)
        self.rejected_key = getattr(self.strategy.args, "rejected_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors,
            load_from_cache_file=False, cache_file_name=None  # 禁用缓存
        )

        # Filter out None values if necessary
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None, cache_file_name=None)

        # Store the processed data in class attributes
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.extras = processed_dataset["extra"]

        if self.search_clue_seg:
            self.matches = processed_dataset["matches"]
        else:
            self.matches = None

    def process_data(self, data):
        prompt, chosen, reject, margin, clue_list = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            self.is_dpo,
            meta_key="meta_info"
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            if self.search_clue_seg: # 找到clue_list里面所有clues在prompt_token_ids中的位置信息
                assert len(clue_list) > 0, "clue_list should not be empty if search_clue_seg is True"
                matches = []
                prompt_len = prompt_token.input_ids.size(-1)
                # print(f"prompt_token.input_ids shape {prompt_token.input_ids.shape}")
                for clue in clue_list:
                    clue_ids = self.tokenizer(clue, return_tensors="pt", add_special_tokens=False)["input_ids"]
                    clue_ids = clue_ids[0, 1:-1]  # 去掉开头和结尾的token做匹配
                    # print(f"clue_ids shape: {clue_ids.shape}")
                    clue_len = clue_ids.size(-1)
                    for i in range(prompt_len - clue_len + 1):
                        if torch.equal(prompt_token.input_ids[0, i : i + clue_len], clue_ids):
                            matches.append((i, i + clue_len - 1))
                            break 
            else:
                matches = []
            # Filter the sample whose length is greater than max_length (2 for answer length)

            if prompt_ids_len >= self.max_length - 2:
                prompt = None

            if len(matches) != len(clue_list):
                print(f"length of matches {len(matches)}, length of clue_list {len(clue_list)})")
                prompt = None

        return {
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
            "extra": prompt_ids_len if self.is_dpo else margin, 
            "matches": matches
        }

    def __len__(self):
        length = len(self.chosens)
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject, extra = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.extras[idx]
        if self.matches:  # 找到匹配的segment在prompt的位置
            seg_pos = self.matches[idx]
        else:
            seg_pos = None

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
            seg_pos,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        reject_ids = zero_pad_sequences(reject_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks, side=padding_side)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras

    def packing_collate_fn(self, item_list):
        extras = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        rejected_ids = []
        rejected_att_masks = []
        rejected_seq_lens = []
        seg_poss = []
        index = 1
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra, seg_pos in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            extras.append(extra)
            seg_poss.append(seg_pos)
            rejected_ids.append(reject_id.flatten())
            rejected_att_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_seq_lens.append(len(reject_id.flatten()))
            index += 1

        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks + rejected_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + rejected_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, extras, seg_poss