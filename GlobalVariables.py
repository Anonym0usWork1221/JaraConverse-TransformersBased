from pathlib import Path
from enum import Enum
from os import path


class VariableParameters(Enum):
    """
    Enumeration for various variable parameters used in model training and setup.

    Attributes:
        MODEL_NAME (str): Name of the model.
        SET_LIMIT_ON_GPU (bool): If True, sets a limit on GPU usage during training.
        MAX_GPU_UTILIZATION_ON_LIMIT (int): Maximum GPU utilization in GB when limit is set.

        SET_LIMIT_ON_CPU (bool): If True, sets a limit on CPU usage during training.
        OMP_THREADS (int): Number of threads to use for OpenMP parallel regions.
        MKL_THREADS (int): Number of threads for Multiple Kernel Learning (if MKL enabled).
        INTER_AND_INTRA_OP_PARALLELISM_THREADS (int): Number of threads for inter and intra-op parallelism.

        SAVED_STATES_NAME (str): File name for saving model states.
        SAVED_HISTORY_NAME (str): File name for saving training history.
        SAVED_MODEL_NAME (str): File name for saving the trained model.
        SAVED_MODEL_WEIGHTS_NAME (str): File name for saving model weights.
        CHECKPOINT_NAME (str): File name for saving model checkpoints.

        BASE_PATH (str): Base directory path.
        MODEL_BASE_PATH (str): Directory path for the model.
        CHECKPOINT_DIR (str): Directory for saving model checkpoints.
        TENSORBOARD_DIR (str): Directory for TensorBoard logs.

        SAVED_STATES_DIR (str): Directory for saving model states.
        CLEANED_DATASET_DIR (str): Directory for saving cleaned dataset.
        SAVED_MODEL_DIR (str): Directory for saving the trained model.

        SAVED_MODEL_WEIGHTS_DIR (str): Directory for saving model weights.
        VISUALIZER_DIR (str): Directory for saving training visualizations.

        SAVED_HISTORY_PATH (str): Full path to the saved history file.
    """

    MODEL_NAME: str = "JaraConverse"  # Name of a model
    SET_LIMIT_ON_GPU: bool = False  # if true program set the limit on gpu usage while training
    MAX_GPU_UTILIZATION_ON_LIMIT: int = 5  # set the number of gbs of graphic card it used when limit set

    SET_LIMIT_ON_CPU: bool = False  # if true program set the limit on gpu usage while training
    OMP_THREADS: int = 5  # sets the number of threads to use for parallel regions
    MKL_THREADS: int = 5  # Multiple Kernel Learning threads (only available if MKL enabled)
    INTER_AND_INTRA_OP_PARALLELISM_THREADS: int = 0  # thread pools (0 = the auto-pickup appropriate number.)

    # File names used in model building and training
    SAVED_STATES_NAME: str = "saved_states.pkl"
    SAVED_HISTORY_NAME: str = "saved_history.pkl"
    SAVED_MODEL_NAME: str = "JaraConverse.keras"
    SAVED_MODEL_WEIGHTS_NAME: str = "saved_weights.h5"
    CHECKPOINT_NAME: str = "cp.ckpt"

    # Directories (Model will automatically create these folders if not found)
    BASE_PATH: str = Path(__file__).parent.__str__()
    MODEL_BASE_PATH: str = path.join(BASE_PATH, f"{MODEL_NAME}Model").__str__()
    CHECKPOINT_DIR: str = path.join(MODEL_BASE_PATH, "model_checkpoints").__str__()
    TENSORBOARD_DIR: str = path.join(MODEL_BASE_PATH, "tensorboard").__str__()

    SAVED_STATES_DIR: str = path.join(MODEL_BASE_PATH, "model_saved_states").__str__()
    CLEANED_DATASET_DIR: str = path.join(MODEL_BASE_PATH, "cleaned_dataset").__str__()
    SAVED_MODEL_DIR: str = path.join(MODEL_BASE_PATH, "trained_model").__str__()

    SAVED_MODEL_WEIGHTS_DIR: str = path.join(MODEL_BASE_PATH, "trained_weights").__str__()
    VISUALIZER_DIR: str = path.join(MODEL_BASE_PATH, "training_visualization").__str__()

    # Full paths to files
    SAVED_HISTORY_PATH: str = path.join(SAVED_STATES_DIR, SAVED_HISTORY_NAME).__str__()


class DataBaseConfiguration(Enum):
    """
    Enumeration for database configuration used in training.

    Attributes:
        TRAINING_DATABASE_PATH (str): Path to the SQLite3 database.
        DATABASE_TABLE_NAME (str): Name of the table containing trainable data.
        UNNECESSARY_COLUMNS_IN_DB (list[str]): List of columns not used in training.

        INPUT_DATA_COLUMN_NAME (str): Column name for input data.
        OUTPUT_DATA_COLUMN_NAME (str): Column name for output data.
        SPLIT_DATASET (bool): If True, splits the dataset into training and testing sets.

        SPLIT_PERCENTAGE (float): Percentage of data used for testing in the dataset split.
        SHUFFLE_DATASET (bool): If True, shuffles the dataset during preprocessing.
        FORCE_REPROCESS_DATASET (bool): If True, forces the dataset reloading and cleaning even once it processed
    """

    # must be sqlite3 db
    TRAINING_DATABASE_PATH: str = path.join(VariableParameters.BASE_PATH.value, "python_code_snippets.db").__str__()
    DATABASE_TABLE_NAME: str = "snippets"  # Table name where trainable data is stored
    UNNECESSARY_COLUMNS_IN_DB: list[str] = None  # Like source column, it is not used in training and take more memory

    INPUT_DATA_COLUMN_NAME: str = "title"  # The Model will take it as input
    OUTPUT_DATA_COLUMN_NAME: str = "code"  # The Model will take it as output
    SPLIT_DATASET: bool = True  # this will split the dataset into train and test datasets

    SPLIT_PERCENTAGE: float = 0.2  # how many percent the split occurs on
    SHUFFLE_DATASET: bool = True  # shuffle the dataset in preprocessing stage
    FORCE_REPROCESS_DATASET: bool = False


class TransformersTokenizerConfiguration(Enum):
    """
    Enumeration for tokenizer configuration used in model training.

    Attributes:
        TOKENIZER_PATH (str): Path to the tokenizer.
        TRAIN_TOKENIZER (bool): If True, trains the tokenizer on a new dataset.
        TRAINING_TOKENIZER_DATA_COLUMN (str): Column name for training the tokenizer.

        TRAINING_TOKENIZER_VOCAB_SIZE (int): Vocabulary size of the tokenizer during training.
        TRAINING_SEED (int): Seed for random operations to ensure reproducibility.
        TRAINING_BATCH_SIZE (int): Batch size used during training.
        VALIDATION_BATCH_SIZE (int): Batch size used during validation.
    """

    TOKENIZER_PATH: str = path.join(VariableParameters.MODEL_BASE_PATH.value, "JaraConverseTokenizer").__str__()

    TRAIN_TOKENIZER: bool = False  # set to True if you want to train this tokenizer on new dataset
    TRAINING_TOKENIZER_DATA_COLUMN: str = "code"  # set the database column on which tokenizer train on
    TRAINING_TOKENIZER_VOCAB_SIZE: int = 52000  # set the max tokenizer vocab size while training on dataset

    TRAINING_SEED: int = 2050  # set this training seed to get the same shuffle every time
    TRAINING_BATCH_SIZE: int = 32  # number of samples that will be used in each iteration of training
    VALIDATION_BATCH_SIZE: int = 8  # Similarly, this is the batch size used during validation


class JaraConverseModelConfiguration(Enum):
    """
    Enumeration for model configuration parameters.

    Attributes:
        MAX_MODEL_INPUT_SIZE (int): Maximum number of tokens the model can take as input.
        MAX_MODEL_OUTPUT_SIZE (int): Maximum number of tokens the model can generate as output.
        MAX_POSITIONAL_ENCODING_LENGTH (int): Maximum length for positional encoding.

        NUMBER_OF_LAYERS (int): Number of transformer layers in the model.
        DIMENSIONALITY_OF_MODEL_EMBEDDINGS (int): Dimension of model embeddings.
        FF_DIMENSION (int): Feed-forward dimension.

        NUM_OF_HEADS (int): Number of self-attention heads in the multi-head attention mechanism.
        LEARNING_DROPOUT_RATE (float): Dropout rate for learning.

        MODEL_EPOCHS (int): Number of epochs for training.
        MODEL_EARLY_STOPPING_PATIENCE (int): Number of epochs with no improvement after which training will be stopped.

        ADAM_SCHEDULER_WARMUP_STEPS (int): Number of warmup steps for the Adam optimizer.
        ADAM_OPTIMIZER_BETA_1 (float): Beta_1 parameter for the Adam optimizer.
        ADAM_OPTIMIZER_BETA_2 (float): Beta_2 parameter for the Adam optimizer.

        ADAM_OPTIMIZER_EPSILON (float): Epsilon parameter for the Adam optimizer.
        GRADIENT_ACCUMULATION_STEPS (int): Number of steps to accumulate gradients before updating weights.
    """

    MAX_MODEL_INPUT_SIZE: int = 512  # how many tokens models will take as input?
    MAX_MODEL_OUTPUT_SIZE: int = 512  # how many tokens models will generate as output?
    MAX_POSITIONAL_ENCODING_LENGTH: int = MAX_MODEL_OUTPUT_SIZE + 50

    NUMBER_OF_LAYERS: int = 6  # Number of transformer layers (Decoder layers to be more specific)
    DIMENSIONALITY_OF_MODEL_EMBEDDINGS: int = 212
    FF_DIMENSION: int = 212  # Feed-Forward Dimension

    NUM_OF_HEADS: int = 8  # number of self-attention heads in the multi-head
    LEARNING_DROPOUT_RATE: float = 0.001  # how much the model drop the learning rate to keep the model under trained
    IS_FIXED_LEARNING_RATE: bool = False    # if True then fiz learning rate will be used
    FIXED_LEARNING_RATE: float = 2.5e-5    # fixed learning rate

    MODEL_EPOCHS: int = 2
    MODEL_EARLY_STOPPING_PATIENCE: int = 5

    ADAM_SCHEDULER_WARMUP_STEPS: int = 4000
    ADAM_OPTIMIZER_BETA_1: float = .9
    ADAM_OPTIMIZER_BETA_2: float = .98

    ADAM_OPTIMIZER_EPSILON: float = 1e-9

    # TODO: Implement gradient accumulation on custom training function (for now it is non functional)
    GRADIENT_ACCUMULATION_STEPS = 4


class AutoCalculateModelParams(object):
    """
    Class for automatically calculated model parameters based on configurations.

    Attributes:
        STEP_PER_TRAINING_EPOC (int): Steps per training epoch, derived from training batch size.
        STEP_PER_VALIDATION_EPOC (int): Steps per validation epoch, derived from validation batch size.
    """
    STEP_PER_TRAINING_EPOC: int = TransformersTokenizerConfiguration.TRAINING_BATCH_SIZE.value
    STEP_PER_VALIDATION_EPOC: int = TransformersTokenizerConfiguration.VALIDATION_BATCH_SIZE.value

