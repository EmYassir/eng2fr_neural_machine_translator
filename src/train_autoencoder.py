"""
Train a transformer model to translate from source to target language
"""

import argparse
import json
import os
import time

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from src.utils.data_utils import create_transformer_dataset, project_root
from src.utils.transformer_utils import CustomSchedule
from src.models.Autoencoder import AutoEncoder
from src.train_transformer import load_tokenizer
from tqdm import tqdm
import datetime


# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def get_summary_tensorboard(save_path: str):
    """
    Utility for tensorboard (currently not used)
    """
    logs_dir = os.path.join(save_path, 'logs', 'gradient_tape')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(logs_dir, current_time, 'train')
    valid_log_dir = os.path.join(logs_dir, current_time, 'valid')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    return train_summary_writer, val_summary_writer


def main() -> None:
    """
    Train the Auto-encoder model
    """
    tf.print('################# Main')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str,
                        help="path to the JSON config file used to define train parameters")
    parser.add_argument('--restore_checkpoint',
                        help='will restore the latest checkpoint',
                        action='store_true')
    parser.add_argument("--data_path", type=str,
                        help="path to the directory where the data is", default=project_root())
    parser.add_argument("--save_path", type=str,
                        help="path to the directory where to save model/tokenizer", default=project_root())
    args = parser.parse_args()
    data_path = args.data_path
    save_path = args.save_path
    config_path = args.cfg_path
    restore_checkpoint = args.restore_checkpoint

    tf.random.set_seed(42)  # Set seed for reproducibility

    assert os.path.isfile(config_path), f"invalid config file: {config_path}"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    num_examples = config["num_examples"]  # set to a smaller number for debugging if needed

    # Necessary to build the dictionaries
    source_unaligned = os.path.join(data_path, config["source_unaligned"])
    source_training = os.path.join(data_path, config["source_training"])
    source_validation = os.path.join(data_path, config["source_validation"])
    source_target_vocab_size = config["source_target_vocab_size"]
    source_input_files = [source_unaligned, source_training]

    target_unaligned = os.path.join(data_path, config["target_unaligned"])
    target_training = os.path.join(data_path, config["target_training"])
    target_validation = os.path.join(data_path, config["target_validation"])
    target_target_vocab_size = config["target_target_vocab_size"]

    target_input_files = [target_unaligned, target_training]

    tokenizer_source_path = os.path.join(save_path, config["tokenizer_source_path"])
    tokenizer_target_path = os.path.join(save_path, config["tokenizer_target_path"])

    # Set hyperparameters
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lambda_factor = config["lambda_factor"]

    # Open other config paths
    with open(config["encoder_cfg_path"], "r") as config_file:
        config_enc = json.load(config_file)
    with open(config["decoder_cfg_path"], "r") as config_file:
        config_dec = json.load(config_file)

    checkpoint_path_enc = os.path.join(save_path, config_enc["checkpoint_path"])
    checkpoint_path_best_enc = os.path.join(save_path, config_enc["checkpoint_path_best"])
    checkpoint_path_dec = os.path.join(save_path, config_dec["checkpoint_path"])
    checkpoint_path_best_dec = os.path.join(save_path, config_dec["checkpoint_path_best"])

    tokenizer_source = load_tokenizer("source", tokenizer_source_path, source_input_files, source_target_vocab_size)
    tokenizer_target = load_tokenizer("target", tokenizer_target_path, target_input_files, target_target_vocab_size)

    autoencoder = AutoEncoder(config, config_enc, config_dec, tokenizer_source, tokenizer_target)

    with open(source_training, "r", encoding="utf-8") as train_source:
        buffer_size = sum([1 for _ in train_source.readlines()])

    def encode(source, target):
        # Add start and end token
        source_tokenized = [tokenizer_source.vocab_size] + tokenizer_source.encode(
            source.numpy()) + [tokenizer_source.vocab_size + 1]

        target_tokenized = [tokenizer_target.vocab_size] + tokenizer_target.encode(
            target.numpy()) + [tokenizer_target.vocab_size + 1]

        return source_tokenized, target_tokenized

    def tf_encode(source, target):
        # encapsulate our encode function in a tf functions so it can be called on tf tensor
        result_source, result_target = tf.py_function(encode, [source, target], [tf.int64, tf.int64])
        result_source.set_shape([None])
        result_target.set_shape([None])

        return result_source, result_target

    train_examples = create_transformer_dataset(source_training, target_training, num_examples=num_examples)
    validation_examples = create_transformer_dataset(source_validation, target_validation, None)

    train_preprocessed = (
        # cache the dataset to memory to get a speedup while reading from it.
        train_examples.map(tf_encode).cache().shuffle(buffer_size)
    )

    val_preprocessed = (validation_examples.map(tf_encode))

    train_dataset = (train_preprocessed
                     .padded_batch(batch_size, padded_shapes=([None], [None]))
                     .prefetch(tf.data.experimental.AUTOTUNE))

    val_dataset = (val_preprocessed
                   .padded_batch(1000, padded_shapes=([None], [None])))

    # Use the Adam optimizer with a custom learning rate scheduler according to the formula
    # in the paper (https://arxiv.org/abs/1706.03762)
    learning_rate = CustomSchedule(config_enc['d_model'] + config_dec['d_model'])
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='val_accuracy')

    def get_ckpt_managers(path, path_best, model):
        ckpt = tf.train.Checkpoint(transformer=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=10)
        ckpt_manager_best = tf.train.CheckpointManager(ckpt, path_best, max_to_keep=1)
        return ckpt, ckpt_manager, ckpt_manager_best

    def restore_ckpts(ckpt, ckpt_manager, path):
        if ckpt_manager.latest_checkpoint:
            # if a checkpoint exists, restore the latest checkpoint.
            ckpt.restore(ckpt_manager.latest_checkpoint)
            tf.print(f'Latest checkpoint for encoder restored from {path}')

    # Checkpoint managers for both encoder and decoder (within the autoencoder)
    # 1. Encoder
    ckpt_enc, ckpt_manager_enc, ckpt_manager_best_enc = get_ckpt_managers(checkpoint_path_enc,
                                                                          checkpoint_path_best_enc,
                                                                          model=autoencoder.encoder)
    # 2. Decoder
    ckpt_dec, ckpt_manager_dec, ckpt_manager_best_dec = get_ckpt_managers(checkpoint_path_dec,
                                                                          checkpoint_path_best_dec,
                                                                          model=autoencoder.decoder)

    # Restore checkpoints for both encoder and decoder (within the autoencoder)
    if restore_checkpoint:
        # 1. Encoder
        restore_ckpts(ckpt_enc, ckpt_manager_enc, checkpoint_path_enc)
        # 2. Decoder
        restore_ckpts(ckpt_dec, ckpt_manager_dec, checkpoint_path_dec)

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = inp[:, 1:]
        with tf.GradientTape() as tape:
            predictions, _, _ = autoencoder(inp, tar, tar_inp, tar_real)
            loss = lambda_factor * loss_function(tar_real, predictions)
            # TODO : Back prop not working!!!
            gradients = tape.gradient(loss, autoencoder.trainable_variables,
                                      unconnected_gradients=tf.UnconnectedGradients.ZERO)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    @tf.function(input_signature=train_step_signature)
    def validate(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = inp[:, 1:]
        predictions, _, _ = autoencoder(inp, tar, tar_inp, tar_real)
        loss = lambda_factor * loss_function(tar_real, predictions)
        val_loss(loss)
        val_accuracy(tar_real, predictions)

    train_summary_writer, val_summary_writer = get_summary_tensorboard(save_path)
    best_val_accuracy = 0

    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                tf.print(f"Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} "
                         f"Accuracy {train_accuracy.result():.4f}")

        for (batch, (inp, tar)) in enumerate(val_dataset):
            validate(inp, tar)
            val_accuracy_result = val_accuracy.result()
            tf.print(f"Epoch {epoch + 1} Batch {batch} Validation Loss {val_loss.result():.4f} "
                     f"Validation Accuracy {val_accuracy_result:.4f}")
        if val_accuracy_result > best_val_accuracy:
            best_val_accuracy = val_accuracy_result
            checkpoint_path_best_enc = ckpt_manager_best_enc.save()
            tf.print(f"Saved best encoder checkpoint for epoch {epoch + 1} at {checkpoint_path_best_enc}")
            checkpoint_path_best_dec = ckpt_manager_best_dec.save()
            tf.print(f"Saved best decoder checkpoint for epoch {epoch + 1} at {checkpoint_path_best_dec}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path_enc = ckpt_manager_enc.save()
            tf.print(f"Saved encoder checkpoint for epoch {epoch + 1} at {checkpoint_path_enc}")
            checkpoint_path_dec = ckpt_manager_dec.save()
            tf.print(f"Saved decoder checkpoint for epoch {epoch + 1} at {checkpoint_path_dec}")

        # Write loss and accuracy so that they can be loaded with tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)

        tf.print(f"Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
        tf.print(f"Time taken for 1 epoch: {time.time() - start} secs\n")
        pbar.update(1)


if __name__ == "__main__":
    main()
