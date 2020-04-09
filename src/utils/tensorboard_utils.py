import os
import datetime
from typing import Dict
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from src.config import ConfigTrainTransformer


def get_summary_tf(save_path: str, hparams: Dict):
    logs_dir = os.path.join(save_path, 'logs', 'gradient_tape')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(logs_dir, current_time, 'train')
    valid_log_dir = os.path.join(logs_dir, current_time, 'valid')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    with train_summary_writer.as_default():
        hp.hparams(hparams)  # record the hparams used in this trial
    return train_summary_writer, val_summary_writer


def hparams_transformer(config: ConfigTrainTransformer, n_train_examples: int) -> Dict:
    source_lang_model = config["source_lang_model"]
    target_lang_model = config["target_lang_model"]
    hparams = {
        "num_layers": config["num_layers"],
        "d_model": config["d_model"],
        "dff": config["dff"],
        "num_heads": config["num_heads"],
        "dropout_rate": config["dropout_rate"],
        "batch_size": config["batch_size"],
        "source_unaligned": config["source_unaligned"],
        "target_unaligned": config["target_unaligned"],
        "source_target_vocab_size": config["source_target_vocab_size"],
        "target_target_vocab_size": config["target_target_vocab_size"],
        "n_train_examples": n_train_examples,
        # Replace None by "None" because NoneType is not a valid hparam type in tensorboard
        "source_lang_model": source_lang_model if source_lang_model is not None else "None",
        "target_lang_model": target_lang_model if target_lang_model is not None else "None",
        "train_encoder_embedding": config["train_encoder_embedding"],
        "train_decoder_embedding": config["train_encoder_embedding"]
    }
    return hparams
