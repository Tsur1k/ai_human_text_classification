import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_config,
        num_classes: int = 2,
        padding_idx: int = 0
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = model_config.embedding_dim
        self.hidden_dim = model_config.hidden_dim
        self.num_layers = model_config.num_layers
        self.dropout_rate = model_config.dropout
        self.bidirectional = model_config.bidirectional
        self.num_classes = num_classes

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=model_config.embedding_dim,
            padding_idx=padding_idx
        )

        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[padding_idx].fill_(0)

        self.embedding_dropout = nn.Dropout(model_config.dropout)

        self.gru = nn.GRU(
            input_size=model_config.embedding_dim,
            hidden_size=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            dropout=model_config.dropout if model_config.num_layers > 1 else 0,
            bidirectional=model_config.bidirectional,
            batch_first=True
        )

        gru_output_size = model_config.hidden_dim
        if model_config.bidirectional:
            gru_output_size *= 2

        self.classifier = nn.Sequential(
            nn.Dropout(model_config.dropout),
            nn.Linear(gru_output_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)

        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)

        # 2. GRU
        if lengths is not None:
            # Используем pack_padded_sequence для эффективности
            lengths = lengths.cpu()  # pack_padded требует CPU тензоры
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, hidden = self.gru(packed_embedded)
        else:
            # Без pack_padded (менее эффективно)
            output, hidden = self.gru(embedded)

        # 3. Получаем фичи из hidden states
        if self.bidirectional:
            # Конкатенируем forward и backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]

        # 4. Классификация
        if return_features:
            return hidden

        logits = self.classifier(hidden)

        return logits

    def predict_proba(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Предсказывает вероятности классов"""
        logits = self.forward(input_ids, lengths)
        return F.softmax(logits, dim=-1)

    def predict(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Предсказывает классы"""
        logits = self.forward(input_ids, lengths)
        return torch.argmax(logits, dim=-1)


def save_model(model: nn.Module, path: str, optimizer=None, scheduler=None, epoch=None, metrics=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_layers': model.num_layers,
            'dropout_rate': model.dropout_rate,
            'bidirectional': model.bidirectional,
            'num_classes': model.num_classes
        }
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)
