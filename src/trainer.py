from transformers.trainer import Trainer, TRAINING_ARGS_NAME
import torch.distributed as dist
from typing import Optional, Dict, Any
import os
import torch
from itertools import repeat
from peft import set_peft_model_state_dict
from transformers.trainer_utils import get_last_checkpoint
import logging

MAX_INPUT_ID = int(1e9)
LLAVE_IMAGE_TOKEN_ID = 32000

logger = logging.getLogger(__name__)

class MMEBTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MMEBTrainer, self).__init__(*args, **kwargs)
        self.is_ddp = dist.is_initialized()
        self._dist_loss_scale_factor = dist.get_world_size() if self.is_ddp else 1

    def compute_loss(self, model, inputs, *args, **kwargs):
        qry_inputs, tgt_inputs, neg_inputs = inputs
        return model(qry=qry_inputs, tgt=tgt_inputs, neg=neg_inputs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'encoder.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        self.model.encoder.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


class MMEBMLMMAETrainer(MMEBMLMTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_mlm_loss = None
        self._current_mae_loss = None

    def compute_loss(self, model, inputs, *args, **kwargs):
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            self._current_mlm_loss = outputs.get('mlm_loss', None)
            self._current_mae_loss = outputs.get('mae_loss', None)
            return outputs.get('loss', 0)
        self._current_mlm_loss = None
        self._current_mae_loss = None
        return outputs

    def log(self, logs: dict, *args, **kwargs):
        mlm_loss = self._current_mlm_loss
        mae_loss = self._current_mae_loss
        avg_total_loss = logs.get('loss', float('nan'))

        if isinstance(mlm_loss, torch.Tensor):
            mlm_loss = mlm_loss.item()
        if isinstance(mae_loss, torch.Tensor):
            mae_loss = mae_loss.item()

        epoch = self.state.epoch
        if epoch is None:
            epoch = self.state.global_step / self.state.max_steps * self.args.num_train_epochs
        if mlm_loss is None:
            mlm_loss = 0.0
        if mae_loss is None:
            mae_loss = 0.0
        try:
            print(
                f"Step {self.state.global_step}: "
                f"MLM={mlm_loss:.4f}, "
                f"MAE={mae_loss:.4f}, "
                f"Total={avg_total_loss:.4f}"
                f", LR={logs.get('learning_rate', float('nan')):.2e}"
                f", Epoch={epoch:.2f}"
            )
        except Exception as e:
            print(f"Error logging: {e}")

        self._current_mlm_loss = None
        self._current_mae_loss = None
