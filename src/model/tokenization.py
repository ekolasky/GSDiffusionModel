from transformers import DataCollatorForTokenClassification
from typing import Any, Dict, List

class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        for feature in features:
            feature['tokens'] = [self.tokenize_point(row) for row in feature['data']]
        return super().__call__(features)

    def tokenize_point(self, point):
        # Implement your custom tokenization logic here
        return point.split()  # Example tokenization logic

def tokenize_dataset(dataset, tokenizer):
    data_collator = CustomDataCollator(tokenizer)
    tokenized_dataset = data_collator(dataset)
    return tokenized_dataset

def tokenize_point(point):

