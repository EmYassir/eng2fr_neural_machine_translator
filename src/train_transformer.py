"""
Train a transformer model to translate from source to target language
"""

import argparse
import json
import os
import time
from typing import Union, List

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.python.framework.errors_impl import NotFoundError

from src.models.Transformer import Transformer
from src.utils.data_utils import build_tokenizer, create_transformer_dataset, project_root
from src.utils.transformer_utils import CustomSchedule, create_masks

# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def load_tokenizer(name: str, path: str, input_files: Union[str, List[str]], vocab_size: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(path)
        print(f"Loaded {name} tokenizer from {path}")
    except NotFoundError:
        print(f"Could not find {name} tokenizer in {path}, building tokenizer...")
        tokenizer = build_tokenizer(input_files, target_vocab_size=vocab_size)
        tokenizer.save_to_file(path)
        print(f"{name} tokenizer saved to {path}")

    return tokenizer


def main() -> None:
    """
    Train the Transformer model
    """
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
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    assert os.path.isfile(config_path), f"invalid config file: {config_path}"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    num_examples = config["num_examples"]  # set to a smaller number for debugging if needed

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
    num_layers = config["num_layers"]
    d_model = config["d_model"]
    dff = config["dff"]
    num_heads = config["num_heads"]
    dropout_rate = config["dropout_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    checkpoint_path = os.path.join(save_path, config["checkpoint_path"])
    checkpoint_path_best = os.path.join(save_path, config["checkpoint_path_best"])

    tokenizer_source = load_tokenizer("source", tokenizer_source_path, source_input_files, source_target_vocab_size)
    tokenizer_target = load_tokenizer("target", tokenizer_target_path, target_input_files, target_target_vocab_size)

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

    train_examples = create_transformer_dataset(source_training, target_training, num_examples)
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

    source_vocab_size = tokenizer_source.vocab_size + 2
    target_vocab_size = tokenizer_target.vocab_size + 2

    # Use the Adam optimizer with a custom learning rate scheduler according to the formula
    # in the paper (https://arxiv.org/abs/1706.03762)
    learning_rate = CustomSchedule(d_model)
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

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              source_vocab_size, target_vocab_size,
                              pe_input=source_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
    ckpt_manager_best = tf.train.CheckpointManager(ckpt, checkpoint_path_best, max_to_keep=1)
    if restore_checkpoint:
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print(f'Latest checkpoint restored from {checkpoint_path}')

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    @tf.function(input_signature=train_step_signature)
    def validate(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

        val_loss(loss)
        val_accuracy(tar_real, predictions)

    best_val_accuracy = 0
    for epoch in range(epochs):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(f"Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} "
                             f"Accuracy {train_accuracy.result():.4f}")

        for (batch, (inp, tar)) in enumerate(val_dataset):
            validate(inp, tar)
            val_accuracy_result = val_accuracy.result()
            print(f"Epoch {epoch + 1} Batch {batch} Validation Loss {val_loss.result():.4f} "
                         f"Validation Accuracy {val_accuracy_result:.4f}")
        if val_accuracy_result > best_val_accuracy:
            best_val_accuracy = val_accuracy_result
            ckpt_save_path_best = ckpt_manager_best.save()
            print(f"Saving best checkpoint for epoch {epoch + 1} at {ckpt_save_path_best}")
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f"Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}")

        print(f"Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")

        print(f"Time taken for 1 epoch: {time.time() - start} secs\n")


if __name__ == "__main__":
    main()
