
"""
Trainer for NER models with W&B logging and early stopping.
"""

import os
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, TrainingArguments
from typing import Dict, Optional
from accelerate import Accelerator
from tdqm import tqdm

from ner_core.training.metrics import compute_entity_metrics


class Trainer:
    """
    Handles training loop, validation, checkpointing, and logging
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        eval_dataset,
        training_args: TrainingArguments,
        id2label: Dict[int, str],
        output_dir: str,
        run_name: Optional[str] = None,
    ):

        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args
        self.id2label = id2label
        self.output_dir = output_dir
        self.run_name = run_name or f"baseline_{training_args.run_name}"


        ### Maybe checkpointing and logging should be handled by train.py because we move to Collab?
        ###Pass "Create output directory"

         self.accelerator = Accelerator(
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            mixed_precision='fp16' if training_args.fp16 else 'no',
        )

        # Create DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers training.args.num_workers,
        )

        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )

        num_update_steps_per_epoch = len(self.train_loader) // training_args.gradient_accumulation_steps
        self.total_steps = num_update_steps_per_epoch * training_args.num_train_epochs

        #Intialize learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = int(training_args.warmup_ratio * self.total_steps),
            num_training_steps = self.total_steps,
        )


        #Prepare Accelerator
        self.model, self.optimizer, self.train_loader, self.eval_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.eval_loader
        )

    
    def train(self):
        """
        Main training loop
        Flow: train on full training set, validate on dev set, check early stopping, save checkpoint if best, load best one
        """

        print(f"Starting training for {self.training_args.num_train_epochs} epochs...")
        print(f"Total training steps: {self.total_steps}")

        for epoch in range(int(self.training_args.num_train_epochs)):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{int(self.training_args.num_train_epochs)}")
            print(f"{'='*50}")

            # Train for one
            train_loss = self.train_epoch(epoch)

            # Validate
            eval_loss = eval_metrics = self.validate()