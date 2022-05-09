
# annotation csv file path
TRAIN_ANNOTATIONS_FILE: str = '../data/test.csv'
TEST_ANNOTATIONS_FILE: str = '../data/test.csv'
VAL_ANNOTATIONS_FILE: str = '../data/validation.csv'

# model file path
MODEL = '../model'  # pytorch model parameter (initial)
# tokenizer file path
TOKENIZER = '../tokenizer/'  # pytorch tokenizer parameter (initial)

N_CLASSES = 7  # number of classes
BATCH_SIZE: int = 20  # mini-batch size
EPOCHS: int = 1  # num of epochs
LEARNING_RATE: float = 0.001  # learning rate
