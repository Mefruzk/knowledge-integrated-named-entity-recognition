import yaml

from transformers import TrainingArguments

with open('configs/baseline_biomedbert.yaml') as f:
    config = yaml.safe_load(f)


training_args = TrainingArguments(



)

#Pass to Trainer
trainer = Trainer(
    model = model,
)