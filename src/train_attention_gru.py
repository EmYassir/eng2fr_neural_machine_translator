"""
Train a machine translation model
"""
import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from src.models.attention_gru import Decoder, Encoder
from src.utils.data_utils import load_dataset

# The following config setting is necessary to work on my local RTX2070 GPU
# Comment if you suspect it's causing trouble
tf_config = ConfigProto()
tf_config.gpu_options.allow_growth = True
session = InteractiveSession(config=tf_config)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def loss_function(real, pred):
    """
    Cross entropy loss on prediction of target words
    :param real: real target words indexes
    :param pred: predicted probabilities of all words at each timestep
    :return: mean loss
    """
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # Create mask where target is padding
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def main(config_path: str):
    """
    Train model
    :param config_path: path to config file
    """
    assert os.path.isfile(config_path), f"invalid config file: {config_path}"
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    NUM_EXAMPLES = config["num_examples"]  # set to a smaller number for debugging if needed
    INPUT_LANGUAGE_MODEL_PATH = config["input_language_model_path"]
    TARGET_LANUAGE_MODEL_PATH = config["target_language_model_path"]
    TRAIN_INPUT_LANGUAGE_PATH = config["train_input_language_path"]
    TRAIN_TARGET_LANGUAGE_PATH = config["train_target_language_path"]
    MAX_SEQ_LENGTH = config["max_seq_length"]  # Set this to some number to avoid very long sequences
    BATCH_SIZE = config["batch_size"]
    EMBEDDING_DIM = config["embedding_dim"]  # Must match dimension of language model embeddings if using one
    UNITS = config["units"]
    EPOCHS = config["epochs"]

    # Load language model here if using
    inp_lang_model = KeyedVectors.load(INPUT_LANGUAGE_MODEL_PATH, mmap='r')
    # Add unknown token to input vocabulary with mean vector as value
    inp_mean_vector = np.mean(inp_lang_model[list(inp_lang_model.vocab.keys())], axis=0)
    inp_lang_model.add(["<unk>"], [inp_mean_vector])

    targ_lang_model = KeyedVectors.load(TARGET_LANUAGE_MODEL_PATH, mmap='r')
    # Add unknown token to target vocabulary with mean vector as value
    targ_mean_vector = np.mean(targ_lang_model[list(targ_lang_model.vocab.keys())], axis=0)
    targ_lang_model.add(["<unk>"], [targ_mean_vector])

    # Load dataset
    input_tensor, target_tensor, _, _ = load_dataset(TRAIN_INPUT_LANGUAGE_PATH,
                                                     TRAIN_TARGET_LANGUAGE_PATH,
                                                     max_seq_length=MAX_SEQ_LENGTH,
                                                     num_examples=NUM_EXAMPLES,
                                                     inp_lang_model=inp_lang_model,
                                                     targ_lang_model=targ_lang_model)
    # TODO split here if using validation set on pre-training task
    buffer_size = len(input_tensor)
    steps_per_epoch = len(input_tensor)//BATCH_SIZE

    vocab_tar_size = len(targ_lang_model.vocab.keys())
    vocab_inp_size = len(inp_lang_model.vocab.keys())

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(buffer_size)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE, inp_lang_model)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE, targ_lang_model)

    optimizer = tf.keras.optimizers.Adam()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            # Pass last hidden state of encoder to decoder
            dec_hidden = enc_hidden

            # Start every prediction with <start>. Add +1 to index to match the index in the embedding layer
            dec_input = tf.expand_dims([targ_lang_model.vocab['<start>'].index + 1] * BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                # Compare predictions with expected target
                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing, pass next target as input
                dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            variables = encoder.trainable_variables + decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss
            if batch % 10 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}')
        print(f'Time taken for 1 epoch {time.time() - start} sec\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str,
                        help="path to the JSON config file used to define train parameters")
    args = parser.parse_args()
    main(
        config_path=args.cfg_path,
    )
