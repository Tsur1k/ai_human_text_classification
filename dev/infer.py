def load_model(path: str, model_class=GRUClassifier, device='cpu'):
    """Загрузка модели"""
    checkpoint = torch.load(path, map_location=device)

    # Создаем модель с теми же параметрами
    model = model_class(
        vocab_size=checkpoint['model_config']['vocab_size'],
        model_config=type('Config', (), checkpoint['model_config'])(),
        num_classes=checkpoint['model_config']['num_classes']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Модель загружена: {path}")

    return model, checkpoint
