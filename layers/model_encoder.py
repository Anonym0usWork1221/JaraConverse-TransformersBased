"""
This module defines the EncoderLayer and Encoder classes, crucial components for building a Transformer-based encoder.

Resources: https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/
"""

from layers.positional_encoding_and_embeddings import PositionalEmbedding
from layers.attention_is_all_you_need import GlobalSelfAttention
from layers.feed_forward import FeedForward
from tensorflow import keras
from typing import Any


@keras.saving.register_keras_serializable(package='EncoderLayer')
class EncoderLayer(keras.layers.Layer):
    """
    EncoderLayer implements a single layer of the encoder in a Transformer-based model.

    This layer comprises global self-attention and feed-forward sublayers.

    Attributes:
        self_attention (GlobalSelfAttention): Global self-attention mechanism.
        feed_forward_network (FeedForward): Feed-forward neural network.
    """

    def __init__(self, *, model_dimensions: int, num_heads: int, feed_forward_dimensions: int,
                 dropout_rate: float = 0.1):
        """
        Initializes the EncoderLayer instance.

        Args:
            model_dimensions (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            feed_forward_dimensions (int): Dimensionality of the feed-forward layers.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """

        super().__init__()

        self.self_attention: GlobalSelfAttention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=model_dimensions,
            dropout=dropout_rate)

        self.feed_forward_network: FeedForward = FeedForward(model_dimensions, feed_forward_dimensions)

    def call(self, inputs: Any, *args, **kwargs) -> Any:
        """
        Forward pass for the EncoderLayer.

        Args:
            inputs (Any): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Output tensor after processing through the encoder layer.
        """

        inputs = self.self_attention(inputs)
        inputs = self.feed_forward_network(inputs)
        return inputs


@keras.saving.register_keras_serializable(package='Encoder')
class Encoder(keras.layers.Layer):
    """
    Encoder implements the encoder component of a Transformer-based model.

    This encoder consists of multiple EncoderLayer instances stacked on top of each other.

    Attributes:
        pos_embedding (PositionalEmbedding): Positional embedding layer.
        enc_layers (list[EncoderLayer]): List of encoder layers.
        dropout_layer (keras.layers.Dropout): Dropout layer.
    """

    def __init__(self, *, num_layers: int, model_dimensions: int, num_heads: int,
                 feed_forward_dimensions: int, vocab_size: int, dropout_rate: float = 0.1):
        """
        Initializes the Encoder instance.

        Args:
            num_layers (int): Number of encoder layers.
            model_dimensions (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            feed_forward_dimensions (int): Dimensionality of the feed-forward layers.
            vocab_size (int): Size of the vocabulary.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """

        super().__init__()
        self.model_dimensions: int = model_dimensions
        self.num_layers: int = num_layers
        self.pos_embedding: PositionalEmbedding = PositionalEmbedding(
            vocab_size=vocab_size, model_dimensions=model_dimensions)
        self.enc_layers: list[EncoderLayer] = [
            EncoderLayer(model_dimensions=model_dimensions,
                         num_heads=num_heads,
                         feed_forward_dimensions=feed_forward_dimensions,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout_layer: keras.layers.Dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs: Any, *args, **kwargs) -> Any:
        """
        Forward pass for the Encoder.

        Args:
            inputs (Any): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Output tensor after processing through the encoder.
        """

        inputs = self.pos_embedding(inputs)
        inputs = self.dropout_layer(inputs)
        for layer_index in range(self.num_layers):
            inputs = self.enc_layers[layer_index](inputs)
        return inputs


if __name__ == '__main__':
    from tensorflow import expand_dims
    from GlobalVariables import JaraConverseModelConfiguration
    from positional_encoding_and_embeddings import PositionalEmbedding

    embed_title = PositionalEmbedding(
        vocab_size=54000,
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value
    )
    input_ids = expand_dims(input=[1, 2626, 279, 445, 358, 10194, 2795, 12321, 2887, 1450], axis=0)
    title_emb = embed_title(input_ids)

    print("===EncoderLayer===")
    sample_encoder_layer = EncoderLayer(
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value,
        num_heads=JaraConverseModelConfiguration.NUM_OF_HEADS.value,
        feed_forward_dimensions=JaraConverseModelConfiguration.FF_DIMENSION.value
    )
    print(title_emb.shape)
    print(sample_encoder_layer(title_emb).shape)

    print("\n===Encoder===")
    sample_encoder = Encoder(
        num_layers=JaraConverseModelConfiguration.NUMBER_OF_LAYERS.value,
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value,
        num_heads=JaraConverseModelConfiguration.NUM_OF_HEADS.value,
        feed_forward_dimensions=JaraConverseModelConfiguration.FF_DIMENSION.value,
        vocab_size=52000
    )
    sample_encoder_output = sample_encoder(input_ids, training=False)
    print(input_ids.shape)
    print(sample_encoder_output.shape)

    assert keras.saving.get_registered_object('Encoder>Encoder') == Encoder
    assert keras.saving.get_registered_name(Encoder) == 'Encoder>Encoder'

    assert keras.saving.get_registered_object('EncoderLayer>EncoderLayer') == EncoderLayer
    assert keras.saving.get_registered_name(EncoderLayer) == 'EncoderLayer>EncoderLayer'
