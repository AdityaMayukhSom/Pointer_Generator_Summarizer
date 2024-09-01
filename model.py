import keras
import tensorflow as tf
from utils import _calc_final_dist
from layers import Encoder, BahdanauAttention, Decoder, Pointer


class PGN(keras.Model):
    def __init__(self, params: dict):
        super(PGN, self).__init__()

        VOCAB_SIZE = params["vocab_size"]
        EMBED_SIZE = params["embed_size"]
        ENC_UNITS = params["enc_units"]
        DEC_UNITS = params["dec_units"]
        ATTN_UNITS = params["attn_units"]
        BATCH_SIZE = params["batch_size"]

        self.params = params

        self.encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, ENC_UNITS, BATCH_SIZE)
        self.attention = BahdanauAttention(ATTN_UNITS)
        self.decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, DEC_UNITS, BATCH_SIZE)
        self.pointer = Pointer()

    def call_encoder(self, enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(enc_inp, enc_hidden)
        return enc_hidden, enc_output

    def call(self, enc_output, dec_hidden, enc_inp, enc_extended_inp, dec_inp, batch_oov_len):
        VOCAB_SIZE = self.params["vocab_size"]
        BATCH_SIZE = self.params["batch_size"]

        predictions = []
        attentions = []
        p_gens = []
        context_vector, _ = self.attention(dec_hidden, enc_output)

        for t in range(dec_inp.shape[1]):
            dec_x, pred, dec_hidden = self.decoder(tf.expand_dims(dec_inp[:, t], 1), dec_hidden, enc_output, context_vector)
            context_vector, attn = self.attention(dec_hidden, enc_output)
            p_gen = self.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))

            predictions.append(pred)
            attentions.append(attn)
            p_gens.append(p_gen)

        final_dists = _calc_final_dist(enc_extended_inp, predictions, attentions, p_gens, batch_oov_len, VOCAB_SIZE, BATCH_SIZE)

        if self.params["mode"] == "train":
            # predictions_shape = (batch_size, dec_len, vocab_size) with dec_len = 1 in pred mode
            return (tf.stack(final_dists, 1), dec_hidden)
        else:
            return (
                tf.stack(final_dists, 1),
                dec_hidden,
                context_vector,
                tf.stack(attentions, 1),
                tf.stack(p_gens, 1),
            )
