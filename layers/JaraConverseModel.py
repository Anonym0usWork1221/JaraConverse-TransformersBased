"""
This module defines the JaraConverseModel class, a custom Keras model that implements a Transformer-based architecture
for sequence-to-sequence tasks, including an encoder and a decoder.

Resources: https://arxiv.org/abs/2012.14913
"""

from GlobalVariables import VariableParameters
from layers.model_encoder import Encoder
from layers.model_decoder import Decoder
from tensorflow import keras
from typing import Any


@keras.saving.register_keras_serializable(package='JaraConverseModel')
class JaraConverseModel(keras.Model):
    """
    JaraConverseModel is a Transformer-based Keras model designed for sequence-to-sequence tasks.

    This model includes an encoder and a decoder, which are used to process input sequences and
    generate output sequences. The final layer is a dense layer that produces the logits for the target vocabulary.

    Attributes:
        encoder (Encoder): The encoder part of the Transformer model.
        decoder (Decoder): The decoder part of the Transformer model.
        final_layer (keras.layers.Dense): A dense layer that produces logits for the target vocabulary.
    """

    def __init__(self, *, num_layers: int,
                 model_dimensions: int,
                 num_heads: int,
                 feed_forward_dimensions: int,
                 input_vocab_size: int,
                 target_vocab_size: int,
                 dropout_rate: float = 0.1):
        """
        Initializes the JaraConverseModel instance.

        Args:
            num_layers (int): Number of layers in the encoder and decoder.
            model_dimensions (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            feed_forward_dimensions (int): Dimensionality of the feed-forward layers.
            input_vocab_size (int): Size of the input vocabulary.
            target_vocab_size (int): Size of the target vocabulary.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """

        super().__init__()
        self.encoder: Encoder = Encoder(
            num_layers=num_layers,
            model_dimensions=model_dimensions,
            num_heads=num_heads,
            feed_forward_dimensions=feed_forward_dimensions,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate,
        )
        self.decoder: Decoder = Decoder(
            num_layers=num_layers,
            model_dimensions=model_dimensions,
            num_heads=num_heads,
            feed_forward_dimensions=feed_forward_dimensions,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate
        )
        self.final_layer: keras.layers.Dense = keras.layers.Dense(
            target_vocab_size, name=f"{VariableParameters.MODEL_NAME.value}_final_layer"
        )

    def call(self, inputs: Any, *args, **kwargs) -> Any:
        """
        Forward pass for the JaraConverseModel.

        Args:
            inputs (Any): Input tensor containing the context and target sequences.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: Output tensor containing the logits for the target vocabulary.
        """

        context, x = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        model_logit = self.final_layer(x)
        try:
            del model_logit._keras_mask
        except AttributeError:
            pass
        return model_logit


if __name__ == '__main__':
    from GlobalVariables import JaraConverseModelConfiguration
    from tensorflow import expand_dims
    model = JaraConverseModel(
        num_layers=JaraConverseModelConfiguration.NUMBER_OF_LAYERS.value,
        model_dimensions=JaraConverseModelConfiguration.DIMENSIONALITY_OF_MODEL_EMBEDDINGS.value,
        num_heads=JaraConverseModelConfiguration.NUM_OF_HEADS.value,
        feed_forward_dimensions=JaraConverseModelConfiguration.FF_DIMENSION.value,
        input_vocab_size=52000,
        target_vocab_size=52000,
        dropout_rate=JaraConverseModelConfiguration.LEARNING_DROPOUT_RATE.value
    )

    input_ids = expand_dims(input=[1, 2626, 279, 445, 358, 10194, 2795, 12321, 2887, 1450], axis=0)
    targ_in = expand_dims(input=[1, 536, 10194, 67, 474, 12, 92, 16, 677, 4672], axis=0)

    output = model((input_ids, targ_in))
    print(input_ids.shape)
    print(targ_in.shape)
    print(output.shape)
    print(model.decoder.dec_layers[-1].last_attn_scores.shape)

    # Checkout model summary
    model.summary()

    assert keras.saving.get_registered_object('JaraConverseModel>JaraConverseModel') == JaraConverseModel
    assert keras.saving.get_registered_name(JaraConverseModel) == 'JaraConverseModel>JaraConverseModel'

