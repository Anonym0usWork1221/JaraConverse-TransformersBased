"""
Resources: https://arxiv.org/pdf/1706.03762.pdf
"""

from tensorflow import keras, cast, float32, math
from typing import Any
from gc import collect


@keras.saving.register_keras_serializable(package='AdamScheduler')
class AdamScheduler(keras.optimizers.schedules.LearningRateSchedule):
    """
    AdamScheduler class implements a custom learning rate schedule for the Adam optimizer.

    This learning rate schedule follows the implementation described in the "Attention is All You Need" paper.

    Attributes:
        model_dimensions_float32 (tf.Tensor): The dimensionality of the model cast to float32.
        warmup_steps (int): The number of warmup steps for the learning rate schedule.

    Methods:
        get_config() -> dict[str, Any]: Returns the configuration of the learning rate schedule.
        __call__(step: Any) -> Any: Computes the learning rate for a given training step.
    """

    def __init__(self, model_dimensions: int, warmup_steps: int = 4000):
        """
        Initializes the AdamScheduler instance.

        Args:
            model_dimensions (int): The dimensionality of the model.
            warmup_steps (int): The number of warmup steps for the learning rate schedule. Default is 4000.
        """

        super().__init__()
        self.model_dimensions_float32 = cast(model_dimensions, float32)
        self.warmup_steps: int = warmup_steps

    def get_config(self) -> dict[str, any([int, float])]:
        """
        Returns the configuration of the learning rate schedule.

        Returns:
            dict[str, Any]: A dictionary containing the model dimensions and warmup steps.
        """

        config: dict[str, any([int, float])] = {
            "model_dimensions": float(self.model_dimensions_float32),
            "warmup_steps": self.warmup_steps
        }
        return config

    def __call__(self, step: Any) -> Any:
        """
        Computes the learning rate for a given training step.

        Args:
            step (Any): The current training step.

        Returns:
            Any: The computed learning rate for the given step.
        """

        step = cast(step, dtype=float32)
        arg1 = math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return math.rsqrt(self.model_dimensions_float32) * math.minimum(arg1, arg2)


@keras.saving.register_keras_serializable(package='GarbageCollectorCallback')
class GarbageCollectorCallback(keras.callbacks.Callback):
    """
    GarbageCollectorCallback class implements a callback that triggers garbage collection at the end of each epoch.

    Methods:
        get_config() -> dict[str, Any]: Returns the configuration of the callback.
        on_epoch_end(epoch, logs=None) -> None: Performs garbage collection at the end of each epoch.
    """

    @staticmethod
    def get_config() -> dict[str, any([int, float])]:
        """
        Returns the configuration of the callback.

        Returns:
            dict[str, Any]: An empty dictionary as there are no configurable parameters.
        """
        config: dict[str, any([int, float])] = {}
        return config

    def on_epoch_end(self, epoch, logs=None) -> None:
        """
        Performs garbage collection at the end of each epoch.

        Args:
            epoch: The current epoch.
            logs: The training logs for the current epoch.
        """

        collect()


@keras.saving.register_keras_serializable(package='AdamLearningRateTracker')
class AdamLearningRateTracker(keras.callbacks.Callback):
    """
    AdamLearningRateTracker class implements a callback that tracks and prints the
    learning rate at the end of each batch.

    Methods:
        get_config() -> dict[str, Any]: Returns the configuration of the callback.
        on_train_batch_end(batch, logs=None) -> None: Tracks and prints the learning
        rate at the end of each training batch.
    """

    @staticmethod
    def get_config() -> dict[str, any([int, float])]:
        """
        Returns the configuration of the callback.

        Returns:
            dict[str, Any]: An empty dictionary as there are no configurable parameters.
        """

        config: dict[str, any([int, float])] = {}
        return config

    def on_train_batch_end(self, batch, logs=None):
        """
        Tracks and prints the learning rate at the end of each training batch.

        Args:
            batch: The current training batch.
            logs: The training logs for the current batch.
        """

        current_lr = self.model.optimizer.learning_rate.numpy()
        print(' - learning_rate: {:.6f}'.format(current_lr), end="")


if __name__ == '__main__':
    # Create adam optimizer
    learning_rate = AdamScheduler(128)
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    print(optimizer.learning_rate)
    assert keras.saving.get_registered_object('AdamScheduler>AdamScheduler') == AdamScheduler
    assert keras.saving.get_registered_name(AdamScheduler) == 'AdamScheduler>AdamScheduler'
    assert keras.saving.get_registered_object('GarbageCollectorCallback>GarbageCollectorCallback'
                                              ) == GarbageCollectorCallback
    assert (keras.saving.get_registered_name(GarbageCollectorCallback) ==
            'GarbageCollectorCallback>GarbageCollectorCallback')
    assert keras.saving.get_registered_object('AdamLearningRateTracker>AdamLearningRateTracker'
                                              ) == AdamLearningRateTracker
    assert (keras.saving.get_registered_name(AdamLearningRateTracker) ==
            'AdamLearningRateTracker>AdamLearningRateTracker')
