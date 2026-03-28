from enum import Enum

class ConfigEnum(Enum):
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TARGET_COLUMN = 'price' # هنفترض إن عمود السعر اسمه كده