from training.metrics import Metrics
from model.model import WhisperKDAModel, create_kda_config
from training.trainer import Trainer
from training.data_collactor import DataCollator
from training.dataset_feature_extractor import DatasetFeatureExtractor

from lion_pytorch import Lion
from datasets import load_from_disk
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_from_disk("preprocessed_dataset/whisper_prepocessed_dataset.arrow")
print(f"Загружено {len(dataset)} примеров")

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-tiny",
    language="russian",
    task="transcribe",
)

model = WhisperKDAModel(
    kda_config=create_kda_config(tokenizer),
    whisper_model_name="openai/whisper-tiny",
    freeze_encoder=True).to(device)

optimizer = Lion(
    model.parameters(),
    lr=3e-5,
    betas=(0.9, 0.99),
    weight_decay=0.1,
)

dataset_builder = DatasetFeatureExtractor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

dataset = dataset_builder.apply(dataset)

metrics = Metrics(tokenizer)
data_collator = DataCollator(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    dataset=dataset,
    data_collator=data_collator,
    metrics=metrics,
    tokenizer=tokenizer,
)
trainer.train()