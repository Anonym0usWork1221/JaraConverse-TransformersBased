"""
This module defines custom loss and accuracy functions for masked sequences and a utility class for
saving model training history.
"""

from tensorflow import keras, cast, reduce_sum, argmax, float32
from GlobalVariables import VariableParameters
from pickle import dump


@keras.saving.register_keras_serializable(package='masked_loss')
def masked_loss(y_true: any, y_predictions: any, padding_token_index: int = 0) -> any:
    """
    Computes the masked loss for sequences with padding tokens.

    Args:
        y_true (any): True labels.
        y_predictions (any): Predicted labels.
        padding_token_index (int): Index of the padding token.

    Returns:
        any: Masked loss.
    """

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_predictions)

    # Mask off the losses on padding tokens.
    mask = cast(y_true != padding_token_index, dtype=loss.dtype)
    loss *= mask

    # Return the total.
    return reduce_sum(loss) / reduce_sum(mask)


@keras.saving.register_keras_serializable(package='masked_accuracy')
def masked_accuracy(y_true: any, y_predictions: any, padding_token_index: int = 0) -> any:
    """
    Computes the masked accuracy for sequences with padding tokens.

    Args:
        y_true (any): True labels.
        y_predictions (any): Predicted labels.
        padding_token_index (int): Index of the padding token.

    Returns:
        any: Masked accuracy.
    """

    # Calculate the loss for each item in the batch.
    y_predictions = argmax(y_predictions, axis=-1)
    y_predictions = cast(y_predictions, y_true.dtype)
    match = cast(y_true == y_predictions, float32)

    # Mask off the accuracy on padding token
    mask = cast(y_true != padding_token_index, float32)
    return reduce_sum(match) / reduce_sum(mask)


class ModelLosses(object):
    """
    Utility class for saving model training history.

    Attributes:
        _history_saving_path (str): Path to save the training history.
    """

    def __init__(self):
        """
        Initializes the ModelLosses object.
        """

        self._history_saving_path = VariableParameters.SAVED_HISTORY_PATH.value

    def save_model_history(self, history: any) -> None:
        """
        Saves the model training history to a file.

        Args:
            history (any): Model training history.
        """

        with open(self._history_saving_path, 'wb') as file_pi:
            dump(history.history, file_pi)


if __name__ == '__main__':
    assert keras.saving.get_registered_object('masked_accuracy>masked_accuracy') == masked_accuracy
    assert keras.saving.get_registered_name(masked_accuracy) == 'masked_accuracy>masked_accuracy'

    assert keras.saving.get_registered_object('masked_loss>masked_loss') == masked_loss
    assert keras.saving.get_registered_name(masked_loss) == 'masked_loss>masked_loss'
