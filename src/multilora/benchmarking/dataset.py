import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence



def get_finetome_dataset(n_examples, tokenizer):
    dataset = load_dataset("mlabonne/FineTome-100k", split='train[:{}]'.format(n_examples))
    
    def tokenize(examples):
        messages = examples["conversations"]
        text = [str(message) for message in messages]
        return tokenizer(text, padding=True, pad_to_multiple_of=16, truncation=True, max_length=512)
    
    return dataset.map(tokenize, batched=True)


def get_bitext_dataset(n_examples, tokenizer):
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split='train[:{}]'.format(n_examples))

    def tokenize(examples):
        def apply_template(example):
            return f"Q: {example[0]}\nA: {example[1]}"
        text = [apply_template(example) for example in zip(examples['instruction'], examples['response'])]
        return tokenizer(text, padding=True, pad_to_multiple_of=16, truncation=True, max_length=512)
    
    return dataset.map(tokenize, batched=True)


def get_guanaco_dataset(n_examples, tokenizer):
    dataset = load_dataset("mlabonne/guanaco-llama2-1k", split='train[:{}]'.format(n_examples))

    def tokenize(examples):
       return tokenizer(examples["text"], padding=True, pad_to_multiple_of=16, truncation=True, max_length=512)
    
    return dataset.map(tokenize, batched=True)

def get_acp_dataset(n_examples, tokenizer):
    dataset = load_dataset("fka/awesome-chatgpt-prompts", split='train[:{}]'.format(n_examples))

    def tokenize(examples):
       return tokenizer(examples["prompt"], padding=True, pad_to_multiple_of=16, truncation=True, max_length=512)

    return dataset.map(tokenize, batched=True)


class MultiAdapterDataset(Dataset):
    """
    Dataset class for the custom dataset that will be used for training and evaluation.
    """
    
    def __init__(self, datasets, tokenizer):
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.lora_cnt = len(datasets)
        assert all([len(d) == len(self.datasets[0]) for d in self.datasets])

    def __len__(self):   
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        """
        Function to get the item from the dataset.
        
        :param idx: index of the item
        :return: item from the dataset. If id is not None, the item from the dataset with the given id is returned, otherwise the item from the dataset with the index (idx % lora_cnt) is returned
        """
        
        # masking is used to determine which adapter is used for the given item, it is 1 for the adapter that is used and 0 for the other adapters
        dataset_id = idx % self.lora_cnt
        d = self.datasets[idx % self.lora_cnt][idx // self.lora_cnt]
        
        ids = torch.tensor(d['input_ids'])
        ids = ids.to(dtype=torch.long)
        mask = torch.tensor(d['attention_mask'])
        labels = ids.clone()
        labels[:-1] = labels[1:].clone()
        labels[-1] = self.tokenizer.eos_token_id
        
        return ids, mask, labels, dataset_id
    
    def collate_fn(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        ids, masks, labels, dataset_ids = zip(*batch)

        ids_padded = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        # Convert dataset_ids to tensor
        dataset_ids = torch.tensor(dataset_ids, dtype=torch.long)

        return ids_padded, masks_padded, labels_padded, dataset_ids