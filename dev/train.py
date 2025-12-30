import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
MLFLOW_AVAILABLE = True



@dataclass
class MLflowConfig:
    tracking_uri: str = "http://127.0.0.1:8080"
    experiment_name: str = "ai-human-classification"


class MLflowLogger:
    def __init__(self, config: MLflowConfig, run_name: str = None):
        if not MLFLOW_AVAILABLE:
            self.enabled = False
            return

        self.enabled = True
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict):
        if self.enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict, step: int = None):
        if self.enabled:
            mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: nn.Module):
        if self.enabled:
            mlflow.pytorch.log_model(model, "model")

    def end_run(self):
        if self.enabled:
            mlflow.end_run()


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        training_config,
        model_config,
        mlflow_config: MLflowConfig = None,
        device: str = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_config = training_config
        self.model_config = model_config

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.mlflow_config = mlflow_config or MLflowConfig()
        self.mlflow_logger = MLflowLogger(self.mlflow_config)

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()

        self.history = []

    def _calculate_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> Dict:

        _, preds = torch.max(outputs, 1)
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        metrics = {
            'f1': f1_score(labels_np, preds_np, average='weighted')
        }

        if outputs.shape[1] == 2:
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            if len(np.unique(labels_np)) >= 2:
                metrics['roc_auc'] = roc_auc_score(labels_np, probs)

        return metrics

    def train_epoch(self) -> Tuple[float, Dict]:
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        all_outputs = []
        all_labels = []

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            inputs = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch.get('lengths')

            self.optimizer.zero_grad()

            if lengths is not None:
                lengths = lengths.to(self.device)
                outputs = self.model(inputs, lengths)
            else:
                outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_outputs.append(outputs.detach())
            all_labels.append(labels)

        avg_loss = total_loss / len(self.train_loader)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = self._calculate_metrics(all_outputs, all_labels)

        return avg_loss, metrics

    def validate(self) -> Tuple[float, Dict]:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch.get('lengths')

                if lengths is not None:
                    lengths = lengths.to(self.device)
                    outputs = self.model(inputs, lengths)
                else:
                    outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)

        avg_loss = total_loss / len(self.val_loader)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = self._calculate_metrics(all_outputs, all_labels)

        return avg_loss, metrics

    def train(self, epochs: int = None):
        """Основной цикл обучения"""
        if epochs is None:
            epochs = self.training_config.max_epochs

        # Логирование гиперпараметров
        if self.mlflow_logger.enabled:
            params = {
                'batch_size': self.training_config.batch_size,
                'learning_rate': self.training_config.learning_rate,
                'epochs': epochs,
                'embedding_dim': self.model_config.embedding_dim,
                'hidden_dim': self.model_config.hidden_dim,
                'num_layers': self.model_config.num_layers,
                'dropout': self.model_config.dropout,
                'bidirectional': self.model_config.bidirectional,
            }
            self.mlflow_logger.log_params(params)

        print(f"Начинаем обучение на {epochs} эпох")

        for epoch in range(1, epochs + 1):
            print(f"\n📊 Эпоха {epoch}/{epochs}")

            # Обучение
            train_loss, train_metrics = self.train_epoch()
            print(f"   Train Loss: {train_loss:.4f}")

            # Валидация
            val_loss, val_metrics = self.validate()
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Val F1: {val_metrics.get('f1', 0):.4f}")
            if 'roc_auc' in val_metrics:
                print(f"   Val ROC-AUC: {val_metrics['roc_auc']:.4f}")

            # Логирование в MLflow
            if self.mlflow_logger.enabled:
                mlflow_metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_f1': val_metrics.get('f1', 0),
                }
                if 'roc_auc' in val_metrics:
                    mlflow_metrics['val_roc_auc'] = val_metrics['roc_auc']

                self.mlflow_logger.log_metrics(mlflow_metrics, step=epoch)

            # Сохранение лучшей модели
            if epoch == 1 or val_loss < min(h['val_loss'] for h in self.history):
                if self.mlflow_logger.enabled:
                    self.mlflow_logger.log_model(self.model)
                print("Лучшая модель сохранена в MLflow")

            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1': val_metrics.get('f1', 0),
                'val_roc_auc': val_metrics.get('roc_auc', 0),
            })

        if self.mlflow_logger.enabled:
            self.mlflow_logger.end_run()

        self.plot_history()
        return self.history

    def plot_history(self):
        if not self.history:
            return

        epochs = [h['epoch'] for h in self.history]
        train_losses = [h['train_loss'] for h in self.history]
        val_losses = [h['val_loss'] for h in self.history]
        val_f1 = [h['val_f1'] for h in self.history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, train_losses, 'b-', label='Train')
        axes[0].plot(epochs, val_losses, 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, val_f1, 'g-')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('Validation F1-Score')
        axes[1].grid(True, alpha=0.3)

        if 'val_roc_auc' in self.history[0]:
            val_roc_auc = [h['val_roc_auc'] for h in self.history]
            axes[2].plot(epochs, val_roc_auc, 'purple')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('ROC-AUC')
            axes[2].set_title('Validation ROC-AUC')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config,
    model_config,
    mlflow_config: MLflowConfig = None,
    epochs: int = None,
    device: str = None
) -> ModelTrainer:

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        model_config=model_config,
        mlflow_config=mlflow_config,
        device=device
    )

    trainer.train(epochs=epochs)
    return trainer
