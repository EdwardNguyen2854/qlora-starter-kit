"""Tests for training module."""

import pytest


class TestTrainingArgs:
    """Tests for training arguments."""

    def test_import_trainer(self):
        from src.training import QLoRATrainer, create_training_args

        assert QLoRATrainer is not None
        assert create_training_args is not None

    def test_create_training_args_defaults(self):
        from src.training import create_training_args

        args = create_training_args(output_dir="./output")

        assert args.output_dir == "./output"
        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 1
        assert args.gradient_accumulation_steps == 4
        assert args.learning_rate == 2e-4

    def test_create_training_args_custom(self):
        from src.training import create_training_args

        args = create_training_args(
            output_dir="./custom",
            num_train_epochs=5,
            per_device_train_batch_size=2,
            learning_rate=1e-4,
        )

        assert args.output_dir == "./custom"
        assert args.num_train_epochs == 5
        assert args.per_device_train_batch_size == 2
        assert args.learning_rate == 1e-4

    def test_training_args_optimizers(self):
        from src.training import create_training_args

        args = create_training_args(output_dir="./output")

        assert args.optim == "paged_adamw_32bit"
        assert args.ddp_find_unused_parameters is False


class TestQLoRATrainer:
    """Tests for QLoRATrainer."""

    def test_trainer_init_requires_model(self):
        from src.training import QLoRATrainer

        with pytest.raises(TypeError):
            QLoRATrainer()

    def test_trainer_has_required_attributes(self):
        from src.training import QLoRATrainer, create_training_args
        from transformers import PreTrainedModel, PreTrainedTokenizer

        class MockModel(PreTrainedModel):
            def __init__(self):
                self.config = None

        class MockTokenizer(PreTrainedTokenizer):
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "<eos>"

        model = MockModel()
        tokenizer = MockTokenizer()

        args = create_training_args(output_dir="./output")

        trainer = QLoRATrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=[],
            training_args=args,
        )

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.trainer is None
