# JaraConverse - A Transformer-Based Supervised LLM

JaraConverse is a state-of-the-art Transformer-based supervised Language Model (LLM) designed for generating Python code snippets. The model is trained using a dataset stored in an SQLite3 database and is equipped with advanced configuration options to optimize training and inference. This README provides an extensive overview of the model, its training process, and configuration details.

 *  Date   : 2024/08/02
 *  Author : **__Abdul Moez__**
 *  Version : 0.1
 *  [Repository](https://github.com/Anonym0usWork1221/JaraConverse-TransformersBased)

 MIT License

## Dependencies
* Python 3.9+
* Tensorflow <=2.15
* Datasets
* Transformers
* codecarbon
* plotly

## License
This project is licensed under the MIT License.

## Table of Contents
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Visualizing Training Progress](#visualizing-training-progress)
- [Running the Demo](#running-the-demo)
- [Configuration Details](#configuration-details)
  - [GlobalVariables.py](#globalvariablespy)

## Installation
Before training the model, ensure you have all necessary dependencies installed. You can do this by running:
```bash
pip install -r requirements.txt
```

## Training the Model
To train the JaraConverse model, execute the following command:

```bash
python JaraConverseTrainer.py
```

Ensure your input data is formatted correctly in the SQLite3 database with columns for `title` and `code`. You can adjust these default column names in the `GlobalVariables.py` file, which holds all the configurations for the model.

## Visualizing Training Progress
JaraConverse uses TensorBoard for monitoring the training process. After training, you can visualize the training progress and other metrics by running:

```bash
python JaraConverseVisualizer.py
```

This will launch TensorBoard and allow you to view detailed graphs and metrics of the training process.

## Running the Demo
The demo script loads the model from a checkpoint and generates code snippets based on the input data. Run the demo script with:

```bash
python JaraConverseDemo.py
```

By default, `JaraConverseDemo.py` loads the model from a checkpoint. This is due to compatibility issues when training on Colab and using the model on another system. Ensure you use the same parameters for loading the checkpoint as those used during training.

## Configuration Details
The `GlobalVariables.py` file contains all the configuration parameters for the JaraConverse model. Below is a detailed explanation of each configuration parameter to help developers understand and customize the model.

### GlobalVariables.py
#### VariableParameters
This enum class holds the general parameters for model training and setup.

```python
class VariableParameters(Enum):
    MODEL_NAME: str = "JaraConverse"
    SET_LIMIT_ON_GPU: bool = False
    MAX_GPU_UTILIZATION_ON_LIMIT: int = 5

    SET_LIMIT_ON_CPU: bool = False
    OMP_THREADS: int = 5
    MKL_THREADS: int = 5
    INTER_AND_INTRA_OP_PARALLELISM_THREADS: int = 0

    SAVED_STATES_NAME: str = "saved_states.pkl"
    SAVED_HISTORY_NAME: str = "saved_history.pkl"
    SAVED_MODEL_NAME: str = "JaraConverse.keras"
    SAVED_MODEL_WEIGHTS_NAME: str = "saved_weights.h5"
    CHECKPOINT_NAME: str = "cp.ckpt"

    BASE_PATH: str = Path(__file__).parent.__str__()
    MODEL_BASE_PATH: str = path.join(BASE_PATH, f"{MODEL_NAME}Model").__str__()
    CHECKPOINT_DIR: str = path.join(MODEL_BASE_PATH, "model_checkpoints").__str__()
    TENSORBOARD_DIR: str = path.join(MODEL_BASE_PATH, "tensorboard").__str__()

    SAVED_STATES_DIR: str = path.join(MODEL_BASE_PATH, "model_saved_states").__str__()
    CLEANED_DATASET_DIR: str = path.join(MODEL_BASE_PATH, "cleaned_dataset").__str__()
    SAVED_MODEL_DIR: str = path.join(MODEL_BASE_PATH, "trained_model").__str__()

    SAVED_MODEL_WEIGHTS_DIR: str = path.join(MODEL_BASE_PATH, "trained_weights").__str__()
    VISUALIZER_DIR: str = path.join(MODEL_BASE_PATH, "training_visualization").__str__()

    SAVED_HISTORY_PATH: str = path.join(SAVED_STATES_DIR, SAVED_HISTORY_NAME).__str__()
```

#### DataBaseConfiguration
This enum class configures the database parameters for training.

```python
class DataBaseConfiguration(Enum):
    TRAINING_DATABASE_PATH: str = path.join(VariableParameters.BASE_PATH.value, "python_code_snippets.db").__str__()
    DATABASE_TABLE_NAME: str = "snippets"
    UNNECESSARY_COLUMNS_IN_DB: list[str] = None

    INPUT_DATA_COLUMN_NAME: str = "title"
    OUTPUT_DATA_COLUMN_NAME: str = "code"
    SPLIT_DATASET: bool = True

    SPLIT_PERCENTAGE: float = 0.2
    SHUFFLE_DATASET: bool = True
    FORCE_REPROCESS_DATASET: bool = False
```

#### TransformersTokenizerConfiguration
This enum class configures the tokenizer parameters for the model.

```python
class TransformersTokenizerConfiguration(Enum):
    TOKENIZER_PATH: str = path.join(VariableParameters.MODEL_BASE_PATH.value, "JaraConverseTokenizer").__str__()

    TRAIN_TOKENIZER: bool = False
    TRAINING_TOKENIZER_DATA_COLUMN: str = "code"
    TRAINING_TOKENIZER_VOCAB_SIZE: int = 52000

    TRAINING_SEED: int = 2050
    TRAINING_BATCH_SIZE: int = 32
    VALIDATION_BATCH_SIZE: int = 8
```

#### JaraConverseModelConfiguration
This enum class configures the model parameters.

```python
class JaraConverseModelConfiguration(Enum):
    MAX_MODEL_INPUT_SIZE: int = 512
    MAX_MODEL_OUTPUT_SIZE: int = 512
    MAX_POSITIONAL_ENCODING_LENGTH: int = MAX_MODEL_OUTPUT_SIZE + 50

    NUMBER_OF_LAYERS: int = 6
    DIMENSIONALITY_OF_MODEL_EMBEDDINGS: int = 212
    FF_DIMENSION: int = 212

    NUM_OF_HEADS: int = 8
    LEARNING_DROPOUT_RATE: float = 0.001
    IS_FIXED_LEARNING_RATE: bool = False
    FIXED_LEARNING_RATE: float = 2.5e-5

    MODEL_EPOCHS: int = 2
    MODEL_EARLY_STOPPING_PATIENCE: int = 5

    ADAM_SCHEDULER_WARMUP_STEPS: int = 4000
    ADAM_OPTIMIZER_BETA_1: float = .9
    ADAM_OPTIMIZER_BETA_2: float = .98

    ADAM_OPTIMIZER_EPSILON: float = 1e-9

    GRADIENT_ACCUMULATION_STEPS = 4
```

#### AutoCalculateModelParams
This class automatically calculates certain model parameters based on configurations.

```python
class AutoCalculateModelParams(object):
    STEP_PER_TRAINING_EPOC: int = TransformersTokenizerConfiguration.TRAINING_BATCH_SIZE.value
    STEP_PER_VALIDATION_EPOC: int = TransformersTokenizerConfiguration.VALIDATION_BATCH_SIZE.value
```


# Contributor

<a href = "https://github.com/Anonym0usWork1221/JaraConverse-TransformersBased/graphs/contributors">
  <img src = "https://contrib.rocks/image?repo=Anonym0usWork1221/JaraConverse-TransformersBased"/>
</a>

-----------
Support and Contact Information
----------
> If you require any assistance or have questions, please feel free to reach out to me through the following channels:  
* **Email**: `abdulmoez123456789@gmail.com`

> I have also established a dedicated Discord group for more interactive communication:  
* **Discord Server**: `https://discord.gg/RMNcqzmt9f`


