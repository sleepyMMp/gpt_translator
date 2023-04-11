from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")

source_lang = "en"
target_lang = "zh"
prefix = "translate English to French: "


def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    
    return inputs, targets

def dataset(model="train",is_opus100=True):
    if is_opus100:
        raw_data = load_dataset("opus100", "en-zh")
        raw_data = raw_data["train"].train_test_split(test_size=0.2)
        train_data = raw_data["train"]
        test_data = raw_data["test"]
        
        if model == "train":
            inputs, tragets = preprocess_function(train_data)
            return inputs, tragets  ## input -> ["sentence0", "sentence1", "sentence2", "sentence3"]
        else:
            inputs, tragets = preprocess_function(test_data)
            return inputs, tragets  ## input -> ["sentence0", "sentence1", "sentence2", "sentence3"]

    else:
        pass
    
    

    
    
