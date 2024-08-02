"""
This module defines the DecoderLayer and Decoder classes, which are essential components of a Transformer-based decoder.

Resources: https://machinelearningmastery.com/implementing-the-transformer-decoder-from-scratch-in-tensorflow-and-keras/
"""

from layers.positional_encoding_and_embeddings import PositionalEmbedding
from layers.attention_is_all_you_need import CausalSelfAttention, CrossAttention
from layers.feed_forward import FeedForward
from tensorflow import keras
from typing import Any


@keras.saving.register_keras_serializable(package='DecoderLayer')
class DecoderLayer(keras.layers.Layer):
    """
    DecoderLayer implements a single layer of the decoder in a Transformer-based model.

    This layer consists of causal self-attention, cross-attention, and feed-forward sublayers.

    Attributes:
        causal_self_attention (CausalSelfAttention): Causal self-attention mechanism.
        cross_attention (CrossAttention): Cross-attention mechanism.
        feed_forward_network (FeedForward): Feed-forward neural network.
    """

    def __init__(self, *, model_dimensions: int, num_heads: int, feed_forward_dimensions: int,
                 dropout_rate: float = 0.1):
        """
        Initializes the DecoderLayer instance.

        Args:
            model_dimensions (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            feed_forward_dimensions (int): Dimensionality of the feed-forward layers.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """

        super(DecoderLayer, self).__init__()
        self.causal_self_attention: CausalSelfAttention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=model_dimensions,
            dropout=dropout_rate)

        self.cross_attention: CrossAttention = CrossAttention(
            num_heads=num_heads,
            key_dim=model_dimensions,
            dropout=dropout_rate)

        self.feed_forward_network: FeedForward = FeedForward(model_dimensions, feed_forward_dimensions)
        self.last_attn_scores: Any = None

    def call(self, inputs: Any, *args, **kwargs) -> Any:
        """
        Forward pass for the DecoderLayer.

        Args:
            inputs (Any): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Output tensor after processing through the decoder layer.
        """

        context: Any = args[0]
        inputs = self.causal_self_attention(inputs=inputs)
        inputs = self.cross_attention(inputs, context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        inputs = self.feed_forward_network(inputs)
        return inputs


@keras.saving.register_keras_serializable(package='Decoder')
class Decoder(keras.layers.Layer):
    """
   Decoder implements the decoder component of a Transformer-based model.

   This decoder consists of multiple DecoderLayer instances stacked on top of each other.

   Attributes:
       pos_embedding (PositionalEmbedding): Positional embedding layer.
       dropout_layer (keras.layers.Dropout): Dropout layer.
       dec_layers (list[DecoderLayer]): List of decoder layers.
   """

    def __init__(self, *, num_layers: int, model_dimensions: int, num_heads: int, feed_forward_dimensions: int,
                 vocab_size: int, dropout_rate: float = 0.1) -> None:
        """
        Initializes the Decoder instance.

        Args:
           num_layers (int): Number of decoder layers.
           model_dimensions (int): Dimensionality of the model.
           num_heads (int): Number of attention heads.
           feed_forward_dimensions (int): Dimensionality of the feed-forward layers.
           vocab_size (int): Size of the vocabulary.
           dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """

        super(Decoder, self).__init__()
        self.model_dimensions: int = model_dimensions
        self.num_layers: int = num_layers
        self.pos_embedding: PositionalEmbedding = PositionalEmbedding(vocab_size=vocab_size,
                                                                      model_dimensions=model_dimensions)
        self.dropout_layer: keras.layers.Dropout = keras.layers.Dropout(dropout_rate)
        self.dec_layers: list[DecoderLayer] = [
            DecoderLayer(model_dimensions=model_dimensions, num_heads=num_heads,
                         feed_forward_dimensions=feed_forward_dimensions, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, inputs: Any, *args, **kwargs) -> Any:
        """
        Forward pass for the Decoder.

        Args:
            inputs (Any): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Output tensor after processing through the decoder.
        """

        context: Any = args[0]
        inputs = self.pos_embedding(inputs)
        inputs = self.dropout_layer(inputs)
        for layer_index in range(self.num_layers):
            inputs = self.dec_layers[layer_index](inputs, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return inputs


if __name__ == '__main__':
    from tensorflow import expand_dims
    from GlobalVariables import JaraConverseModelConfiguration

    embed_title = PositionalEmbedding(
        vocab_size=54000,
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value
    )
    embed_code = PositionalEmbedding(
        vocab_size=54000,
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value
    )
    input_ids = expand_dims(input=[1, 2626, 279, 445, 358, 10194, 2795, 12321, 2887, 1450], axis=0)
    targ_in = expand_dims(input=[1, 536, 10194, 67, 474, 12, 92, 16, 677, 4672], axis=0)

    title_emb = embed_title(input_ids)
    code_emb = embed_code(targ_in)

    print(f"Title Embedded Shape: {title_emb.shape}")
    print(f"Code Embedded Shape: {code_emb.shape}")

    print("\n===DecoderLayer===")
    sample_decoder_layer = DecoderLayer(
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value,
        num_heads=JaraConverseModelConfiguration.NUM_OF_HEADS.value,
        feed_forward_dimensions=JaraConverseModelConfiguration.FF_DIMENSION.value
    )
    sample_decoder_layer_output = sample_decoder_layer(
        code_emb, title_emb
    )
    print(sample_decoder_layer_output.shape)

    print("\n===Decoder===")
    sample_decoder = Decoder(
        num_layers=JaraConverseModelConfiguration.NUMBER_OF_LAYERS.value,
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value,
        num_heads=JaraConverseModelConfiguration.NUM_OF_HEADS.value,
        feed_forward_dimensions=JaraConverseModelConfiguration.FF_DIMENSION.value,
        vocab_size=52000
    )
    output = sample_decoder(targ_in, title_emb)
    print(targ_in.shape)
    print(output.shape)
    print(sample_decoder.last_attn_scores.shape)

    assert keras.saving.get_registered_object('DecoderLayer>DecoderLayer') == DecoderLayer
    assert keras.saving.get_registered_name(DecoderLayer) == 'DecoderLayer>DecoderLayer'

    assert keras.saving.get_registered_object('Decoder>Decoder') == Decoder
    assert keras.saving.get_registered_name(Decoder) == 'Decoder>Decoder'
