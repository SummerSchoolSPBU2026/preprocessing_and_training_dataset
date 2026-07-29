import torch

class DataCollator:
    def __init__(self, feature_extractor, tokenizer):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_features = [
            {"input_features": example["input_features"]}
            for example in features
        ]

        batch = self.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        label_features = [
            {"input_ids": example["labels"]}
            for example in features
        ]

        labels_batch = self.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"] != 1,
            -100,
        )

        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels

        return batch