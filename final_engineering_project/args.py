import argparse

_parser = argparse.ArgumentParser(description="UV and Sarah sound selection project!")

#################################
# START - DATA arguments
#################################
_parser.add_argument(
    "--data-enable",
    dest="data_enable",
    action="store_true",
    help="Whatever or not execute data creation step (store to fs).",
)
_parser.set_defaults(data_enable=False)

_parser.add_argument(
    "--data-train-size",
    type=int,
    required=False,
    default=100,
    help="Size of train samples to create in fs.",
)

_parser.add_argument(
    "--data-test-size",
    type=int,
    required=False,
    default=10,
    help="Size of test samples to create in fs.",
)

_parser.add_argument(
    "--data-min-mixure",
    type=int,
    required=False,
    default=3,
    help="Minimum number of classes in mixure when creating data.",
)

_parser.add_argument(
    "--data-max-mixure",
    type=int,
    required=False,
    default=3,
    help="Maximum number of classes in mixure when creating data.",
)

_parser.add_argument(
    "--data-print-progress-every",
    type=int,
    required=False,
    help="Print progress (with time it took) for every number of samples created.",
)
#################################
# END - DATA arguments
#################################

#################################
# START - TRAIN arguments
#################################
_parser.add_argument(
    "--train-enable",
    dest="train_enable",
    action="store_true",
    help="Whatever or not execute train model step.",
)
_parser.set_defaults(train_enable=False)

_parser.add_argument(
    "--train-use-fs",
    dest="train_use_fs",
    action="store_true",
    help="Whatever or not execute train model step.",
)
_parser.set_defaults(train_use_fs=False)

_parser.add_argument(
    "--train-min-mixure",
    type=int,
    required=False,
    default=3,
    help="Minimum number of classes in mixure if not using fs when creating data.",
)

_parser.add_argument(
    "--train-max-mixure",
    type=int,
    required=False,
    default=3,
    help="Maximum number of classes in mixure if not using fs when creating data.",
)

_parser.add_argument(
    "--train-override-model",
    dest="train_override_model",
    action="store_true",
    help="Train a new model (override old one).",
)
_parser.set_defaults(train_use_fs=False)

_parser.add_argument(
    "--train-size",
    type=int,
    required=False,
    default=100,
    help="Size of train samples to train the model with.",
)

_parser.add_argument(
    "--train-epoch-size",
    type=int,
    required=False,
    default=1,
    help="Size of train epochs to train the model.",
)

_parser.add_argument(
    "--train-batch-size",
    type=int,
    required=False,
    default=10,
    help="Size of samples in single batch.",
)

_parser.add_argument(
    "--train-step-size",
    type=int,
    required=False,
    default=100,
    help="Size of steps needed to decrease learning rate",
)

_parser.add_argument(
    "--train-save-model-every",
    type=int,
    required=False,
    help="Save model to fs every number of batches.",
)

_parser.add_argument(
    "--train-print-progress-every",
    type=int,
    required=False,
    help="Print progress (with time it took) for every number of batches trained.",
)
#################################
# END - TRAIN arguments
#################################

#################################
# START - TEST SAMPLE arguments
#################################
_parser.add_argument(
    "--test-sample",
    dest="test_sample",
    action="store_true",
    help="Whatever or not execute test create sample.",
)
_parser.set_defaults(test_enable=False)
#################################
# END - TEST SAMPLE arguments
#################################
#################################
# START - TEST arguments
#################################
_parser.add_argument(
    "--test-enable",
    dest="test_enable",
    action="store_true",
    help="Whatever or not execute test model step.",
)
_parser.set_defaults(test_enable=False)

_parser.add_argument(
    "--test-size",
    type=int,
    required=False,
    default=100,
    help="Size of test samples to test the model with.",
)

_parser.add_argument(
    "--test-batch-size",
    type=int,
    required=False,
    default=10,
    help="Size of samples in single batch.",
)

_parser.add_argument(
    "--test-print-progress-every",
    type=int,
    required=False,
    help="Print progress (with time it took) for every number of batches tested.",
)
#################################
# END - TEST arguments
#################################

args = _parser.parse_args()
