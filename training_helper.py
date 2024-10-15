import time

import keras
import tensorflow as tf
from loguru import logger

import batcher
from model import PGN


class ModelTrainer:
    def __init__(
        self,
        model: PGN,
        optimizer: keras.optimizers.Optimizer,
        dataset: tf.data.Dataset,
        vocab: batcher.Vocab,
        batch_size: int,
        maximum_training_steps: int,
        checkpoint_save_steps: int,
    ):
        self.model = model
        self.dataset = dataset
        self.vocab = vocab
        self.optimizer = optimizer

        self.batch_size = batch_size
        self.max_training_steps = maximum_training_steps
        self.checkpoint_save_steps = checkpoint_save_steps

    def loss_function(self, real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
        """
        Args:
            real (tf.Tensor): (batch, seq_len)
            pred (tf.Tensor): (batch, seq_len, None)

        Returns:
            _type_: _description_
        """
        # an array containing row-wise losses
        loss_ = keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=False)
        loss_ = tf.convert_to_tensor(loss_)

        mask = tf.math.equal(real, tf.constant(1))
        mask = tf.math.logical_not(mask)
        mask = tf.cast(mask, dtype=tf.float32)

        if not isinstance(mask, tf.Tensor):
            raise ValueError(f"mask is of type {type(mask)}, should be an instange of tf.Tensor")

        # dec_lens contains row-wise sums of the mask, basically elements which has been decoded
        dec_lens: tf.Tensor = tf.reduce_sum(mask, axis=-1)

        loss_ = tf.multiply(loss_, mask)
        # we have to make sure no empty abstract is being used otherwise dec_lens may contain null values
        loss_ = tf.reduce_sum(loss_, axis=-1) / dec_lens

        # tensor with a single dimention is returned as axis is None
        mean_loss: tf.Tensor = tf.reduce_mean(loss_)
        return mean_loss

    # input_signature = [
    #     tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
    #     tf.TensorSpec(shape=[params["batch_size"], None], dtype=tf.int32),
    #     tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
    #     tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int32),
    #     tf.TensorSpec(shape=[], dtype=tf.int32),
    # ]
    def train_step(
        self,
        enc_inp: tf.Tensor,
        enc_extended_inp: tf.Tensor,
        dec_inp: tf.Tensor,
        dec_tar: tf.Tensor,
        batch_oov_len: tf.Tensor,
        article_oovs,
        debug_input_text=False,
        debug_output_text=False,
    ) -> tf.Tensor:
        """
        Args:
            enc_inp (tf.Tensor): Numerical encoding of the input article along with OOV words as UNK.
            enc_extended_inp (tf.Tensor): Tensor containing the ids of the words present in the article
                along with ids for the OOV words.
            dec_inp (tf.Tensor): Tensor containing the numerical ids of (start token + abstract).
            dec_tar (tf.Tensor): Tensor containing the numerical ids of (abstract + end token).
            batch_oov_len (tf.Tensor): Maximum number of OOV words present in this example.
        Raises:
            ValueError: In case some gradients become None during training.

        Returns:
            tf.Tensor: The mean loss value for the particular training step.
        """

        loss: tf.Tensor = tf.zeros([1], tf.float32)

        with tf.GradientTape() as tape:
            if debug_input_text:
                input_word_list = batcher.DataHelper.output_to_words(
                    list(enc_inp[0].numpy()),  # type: ignore
                    self.vocab,
                    article_oovs,
                )
                print("\n\n", " ".join(input_word_list))

            enc_outputs, enc_c, enc_h = self.model.call_encoder(enc_inp)
            predictions, _ = self.model(
                enc_outputs,
                enc_inp,
                enc_extended_inp,
                dec_inp,
                batch_oov_len=batch_oov_len,
                training=True,
            )

            if debug_output_text:
                output_word_list = batcher.DataHelper.output_to_words(
                    list(tf.argmax(predictions[0, :, :], axis=1).numpy()),
                    self.vocab,
                    article_oovs,
                )
                print(" ".join(output_word_list), "\n\n")

            variables = (
                self.model.enc_pos_emb.trainable_variables
                + self.model.encoder.trainable_variables
                + self.model.dec_pos_emb.trainable_variables
                + self.model.decoder.trainable_variables
                + self.model.final_layer.trainable_variables
            )

            loss = self.loss_function(dec_tar, predictions)
            gradients = tape.gradient(target=loss, sources=variables)

            if gradients is None:
                raise ValueError("gradients cannot be none")

            self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    # @tf.function
    def execute(
        self,
        callbacks: keras.callbacks.CallbackList,
        out_file: str,
        batch_number: int,
    ):
        try:
            f = open(out_file, "w+")
            callbacks.on_train_begin()
            for batch in self.dataset:
                batch_number += 1

                if batch is None:
                    logger.warning("Skipping Batch Number {}".format(batch_number))
                    continue

                callbacks.on_epoch_begin(batch_number)
                # callbacks.on_train_batch_begin(batch_number)

                # len(batch) is 2, batch is a tuple of two elements
                max_oov_len = batch[0]["article_oovs"].shape[1]

                start_time = time.time()
                loss = self.train_step(
                    batch[0]["enc_input"],
                    batch[0]["extended_enc_input"],
                    batch[1]["dec_input"],
                    batch[1]["dec_target"],
                    max_oov_len,
                    batch[0]["article_oovs"],
                )
                end_time = time.time()

                step_time = end_time - start_time
                current_loss = loss.numpy()  # type: ignore

                print("Batch {}, time {:.4f}, Loss {:.4f}".format(batch_number, step_time, current_loss))
                f.write("Batch {}, time {:.4f}, Loss {:.4f}".format(batch_number, step_time, current_loss))

                # callbacks.on_train_batch_end(batch_number)
                callbacks.on_epoch_end(batch_number)
                if batch_number == self.max_training_steps:
                    break
            callbacks.on_train_end()
        except KeyboardInterrupt:
            # checkpoint_path = "{}/{}_batch_{}.keras".format(
            #     callbacks,
            #     self.model_name,
            #     batch + 1,
            # )
            # self.model.save(checkpoint_path)
            # logger.info("Saved Checkpoint: " + checkpoint_path + "\n")
            logger.info("keyboard interrupt")
        finally:
            f.close()
