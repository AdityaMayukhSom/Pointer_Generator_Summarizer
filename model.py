import keras
import tensorflow as tf
from encoder import Encoder, EncoderReducer
from decoder import Decoder
from embedding import PositionalEmbedding


class PGN(keras.Model):
    def __init__(self, params: dict):
        super(PGN, self).__init__()

        VOCAB_SIZE = params["vocab_size"]
        BATCH_SIZE = params["batch_size"]

        # size of the embedding, essentially equal to d_model
        EMBED_SIZE = params["embed_size"]

        ENC_UNITS = params["enc_units"]
        DEC_UNITS = params["dec_units"]

        MAX_ENC_LENGTH = params["max_enc_len"]
        MAX_DEC_LENGTH = params["max_dec_len"]

        self.NUM_HEADS = 4
        ATTN_UNITS = params["attn_units"]
        DROPOUT_RATE = 0.2

        self.params = params

        self.enc_pos_emb = PositionalEmbedding(MAX_ENC_LENGTH, VOCAB_SIZE, EMBED_SIZE)
        self.encoder = Encoder(ENC_UNITS, BATCH_SIZE)
        self.encoder_reducer = EncoderReducer(ENC_UNITS)

        # self.attention = BahdanauAttention(ATTN_UNITS)
        # self.pointer = Pointer()

        self.dec_pos_emb = PositionalEmbedding(MAX_DEC_LENGTH, VOCAB_SIZE, EMBED_SIZE)
        self.decoder = Decoder(DEC_UNITS, EMBED_SIZE, self.NUM_HEADS, 2048)
        self.dropout = keras.layers.Dropout(DROPOUT_RATE)

        self.final_layer = keras.layers.Dense(VOCAB_SIZE)

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 1), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len) # type: ignore

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by the decoder.
        target_shape = tf.shape(tar)[1]  # type: ignore
        look_ahead_mask = self.create_look_ahead_mask(target_shape)
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def rnn_generate_mask_old(self, src, tgt):
        # Source mask (src != 0) and expand dimensions without indexing
        src_mask = tf.cast(tf.not_equal(src, 0), dtype=tf.float32)
        src_mask = tf.expand_dims(src_mask, axis=1)
        src_mask = tf.expand_dims(src_mask, axis=2)

        # Target mask (tgt != 0) and expand dimensions without indexing
        tgt_mask = tf.cast(tf.not_equal(tgt, 0), dtype=tf.float32)
        tgt_mask = tf.expand_dims(tgt_mask, axis=1)
        tgt_mask = tf.expand_dims(tgt_mask, axis=3)

        # Sequence length of the target
        seq_length = tf.shape(tgt)[1]  # type: ignore

        # No peak mask (upper triangular matrix with 1's below diagonal and 0's above diagonal)
        nopeak_mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)

        # Apply no peak mask to the target mask
        tgt_mask = tgt_mask * nopeak_mask

        return src_mask, tgt_mask

    def _calc_final_dist(
        self, _enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size
    ):
        """
        Calculate the final distribution, for the pointer-generator model.

        Args:
            vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
            attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
        Returns:
            final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """

        # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

        # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
        # the maximum (over the batch) size of the extended vocabulary
        extended_vsize = vocab_size + batch_oov_len
        extra_zeros = tf.zeros((batch_size, batch_oov_len))
        # list length max_dec_steps of shape (batch_size, extended_vsize)
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection
        batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        # number of states we attend over
        attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # type: ignore
        # shape (batch_size, attn_len)
        batch_nums = tf.tile(batch_nums, [1, attn_len])
        # shape (batch_size, enc_t, 2)
        indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)
        shape = [batch_size, extended_vsize]
        # list length max_dec_steps (batch_size, extended_vsize)
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

        # Add the vocab distributions and the copy distributions together to get the final distributions
        # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
        # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
        final_dists = [
            vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)
        ]

        return final_dists

    def call(self, enc_inp: tf.Tensor, enc_extended_inp: tf.Tensor, dec_inp: tf.Tensor, batch_oov_len, training: bool):
        VOCAB_SIZE = self.params["vocab_size"]
        BATCH_SIZE = self.params["batch_size"]

        enc_pad_mask, dec_look_ahead_mask, dec_pad_mask = self.create_masks(enc_inp, dec_inp)

        enc_hidden = self.encoder.initialize_hidden_state()

        enc_emb_input = self.enc_pos_emb(enc_inp)
        # enc_outputs, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c = self.encoder(enc_emb_input, enc_hidden)
        # enc_c, enc_h = self.encoder_reducer(enc_forward_c, enc_forward_h, enc_backward_c, enc_backward_h)
        enc_outputs, enc_forward_h = self.encoder(enc_emb_input, enc_hidden)

        dec_emb_input = self.dec_pos_emb(dec_inp)
        dec_output, attn_weights, p_gens = self.decoder(
            embed_x=dec_emb_input,
            enc_output=enc_outputs,
            training=training,
            look_ahead_mask=dec_look_ahead_mask,
            padding_mask=dec_pad_mask,
        )

        # (batch_size, tar_seq_len, target_vocab_size)
        transformer_output = self.final_layer(dec_output)
        # (batch_size, tar_seq_len, vocab_size)
        transformer_output = tf.nn.softmax(transformer_output)
        # # (batch_size, targ_seq_len, vocab_size+max_oov_len)
        # output = tf.concat([output, tf.zeros((tf.shape(output)[0], tf.shape(output)[1], max_oov_len))], axis=-1)

        # (batch_size,num_heads, targ_seq_len, inp_seq_len)
        attn_dists = attn_weights["decoder_layer{}_block2".format(self.params["dec_units"])]
        # (batch_size, targ_seq_len, inp_seq_len)
        attn_dists = tf.reduce_sum(attn_dists, axis=1) / self.NUM_HEADS

        final_dists = self._calc_final_dist(
            enc_extended_inp,
            tf.unstack(transformer_output, axis=1),
            tf.unstack(attn_dists, axis=1),
            tf.unstack(p_gens, axis=1),
            batch_oov_len,
            VOCAB_SIZE,
            BATCH_SIZE,
        )

        final_output = tf.stack(final_dists, axis=1)

        if self.params["mode"] == "train":
            # predictions_shape = (batch_size, dec_len, vocab_size) with dec_len = 1 in pred mode
            return final_output, attn_weights
        else:
            return (
                tf.stack(final_dists, 1),
                dec_hidden,
                context_vector,
                tf.stack(tf.unstack(attn_dists, axis=1), 1),
                tf.stack(tf.unstack(p_gens, axis=1), 1),
            )
