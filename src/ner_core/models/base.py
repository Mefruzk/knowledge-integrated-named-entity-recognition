import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple

class BaseNERModel(nn.Module):
    """
    Baseline BERT-based NER model.
    Simple token classification:
    BERT -> dropout -> Linear -> BIO Labels
    """

    def __init__(
        self,
        model_name: str = "Microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        num_labels:int = 5,
        dropout_prob: float = 0.1,
        freeze_encoder: bool = False,
    ):

        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)

        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        #loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for NER model.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        predictions = torch.argmax(logits, dim=-1)

        return {
            'loss': loss,
            'logits': logits,
            'predictions': opredictions
        }

    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference mode (no labels, no loss computation).
            
        Returns:
            predictions: (batch_size, seq_len) - predicted label IDs
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None
            )
        return outputs['predictions']