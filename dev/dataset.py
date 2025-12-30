import torch
from torch.utils.data import Dataset
from collections import Counter
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
import numpy as np


class TextClassificationDataset(Dataset):

    def __init__(
        self,
        data_config: 'data_load.yaml',
        training_config: 'training.yaml',
        model_config: 'model.yaml',
        split: str = 'train',
        max_length: int = None
    ):
        self.data_config = data_config
        self.training_config = training_config
        self.model_config = model_config
        self.split = split

        # Определяем путь к данным в зависимости от раздела
        if split == 'train':
            self.data_path = data_config.data_train
        elif split == 'valid':
            self.data_path = data_config.data_valid
        elif split == 'test':
            self.data_path = data_config.data_test
        else:
            raise ValueError(f"Unknown split: {split}")

        # Загружаем данные
        self.texts, self.labels = self._load_data()

        # Настройки токенизации
        self.max_length = max_length or 256

        # Словарь будет создан только для тренировочного датасета
        self.vocab = None
        self.vocab_size = 0

        self._tokenized_texts = None

        # Устанавливаем seed для воспроизводимости
        if split == 'train':
            np.random.seed(training_config.random_seed)
            torch.manual_seed(training_config.random_seed)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        path = Path(__file__).parent.parent
        path = path / self.data_path

        df = pd.read_csv(path)

        if 'text' not in df.columns:
            raise ValueError("Column 'text' not found in data")
        if 'generated' not in df.columns:
            raise ValueError("Column 'generated' not found in data")

        texts = df['text'].astype(str).tolist()
        labels = df['generated'].astype(int).tolist()

        return texts, labels

    def _tokenize_text(self, text: str) -> List[str]:
        tokens = text.lower().split()

        # Ограничение длины
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        return tokens

    def build_vocab(self, vocab: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        if vocab is not None:
            self.vocab = vocab
            self.vocab_size = len(vocab)
            return self.vocab

        # Только тренировочный датасет строит словарь с нуля
        if self.split != 'train':
            raise ValueError("Vocab can only be built from training dataset")

        all_tokens = []

        self._tokenized_texts = []
        for text in self.texts:
            tokens = self._tokenize_text(text)
            self._tokenized_texts.append(tokens)
            all_tokens.extend(tokens)

        counter = Counter(all_tokens)
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }

        # Добавляем наиболее частые токены
        # Ограничиваем размер словаря из конфига модели
        max_vocab_size = min(
            self.model_config.embedding_dim,  # или другое ограничение
            50000  # дефолтное ограничение
        )

        sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for token, freq in sorted_tokens:
            if len(vocab) >= max_vocab_size:
                break
            vocab[token] = len(vocab)

        self.vocab = vocab
        self.vocab_size = len(vocab)

        print(f"Vocabulary built with {self.vocab_size} tokens")
        print(f"Top 10 tokens: {list(vocab.keys())[:10]}")

        return vocab

    def set_vocab(self, vocab: Dict[str, int]) -> None:
        """Установка словаря (для valid/test датасетов)"""
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Токенизируем тексты
        self._tokenized_texts = [self._tokenize_text(text) for text in self.texts]

    def _text_to_indices(self, tokens: List[str]) -> torch.Tensor:
        """Конвертация токенов в индексы"""
        if self.vocab is None:
            raise ValueError("Vocabulary not initialized. Call build_vocab() or set_vocab() first.")

        indices = []

        # Добавляем SOS токен в начале
        indices.append(self.vocab['<SOS>'])

        # Конвертируем токены в индексы
        for token in tokens:
            idx = self.vocab.get(token, self.vocab['<UNK>'])
            indices.append(idx)

        # Добавляем EOS токен в конце
        indices.append(self.vocab['<EOS>'])

        # Обрезаем если слишком длинная последовательность
        if len(indices) > self.max_length + 2:  # +2 для SOS и EOS
            indices = indices[:self.max_length + 2]

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._tokenized_texts is None:
            # Токенизируем на лету если не сделано заранее
            tokens = self._tokenize_text(self.texts[idx])
        else:
            tokens = self._tokenized_texts[idx]

        # Конвертируем в индексы
        token_indices = self._text_to_indices(tokens)

        # Создаем тензор метки
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            'input_ids': token_indices,
            'labels': label_tensor,
            'text': self.texts[idx]  # Сохраняем оригинальный текст для отладки
        }


class CollateFn:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Обработка батча"""
        # Извлекаем данные
        input_ids = [item['input_ids'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])

        # Паддинг последовательностей
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )

        # ДЛИНЫ последовательностей (важно для pack_padded_sequence)
        lengths = torch.tensor([len(seq) for seq in input_ids], dtype=torch.long)

        return {
            'input_ids': padded_input_ids,
            'lengths': lengths,
            'labels': labels,
            'texts': [item.get('text', '') for item in batch]
        }


def create_dataloaders(
    data_config: 'data_load.yaml',
    training_config: 'training.yaml',
    model_config: 'model.yaml',
) -> Tuple[torch.utils.data.DataLoader,
           torch.utils.data.DataLoader,
           torch.utils.data.DataLoader]:

    train_dataset = TextClassificationDataset(
        data_config=data_config,
        training_config=training_config,
        model_config=model_config,
        split='train'
    )

    valid_dataset = TextClassificationDataset(
        data_config=data_config,
        training_config=training_config,
        model_config=model_config,
        split='valid'
    )

    test_dataset = TextClassificationDataset(
        data_config=data_config,
        training_config=training_config,
        model_config=model_config,
        split='test'
    )

    vocab = train_dataset.build_vocab()

    valid_dataset.set_vocab(vocab)
    test_dataset.set_vocab(vocab)

    collate_fn = CollateFn(pad_token_id=vocab['<PAD>'])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn
    )

    return train_loader, valid_loader, test_loader
