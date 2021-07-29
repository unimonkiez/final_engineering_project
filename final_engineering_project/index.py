from final_engineering_project.data.create_data import create_data
from final_engineering_project.train.train import train
from final_engineering_project.test.test import test
from final_engineering_project.test.create_sample import create_sample
from final_engineering_project.args import args


def start() -> None:
    if args.data_enable:
        create_data(
            train_size=args.data_train_size,
            test_size=args.data_test_size,
            print_progress_every=args.data_print_progress_every,
            min_mixure=args.data_min_mixure,
            max_mixure=args.data_max_mixure,
        )
    if args.train_enable:
        train(
            use_fs=args.train_use_fs,
            override_model=args.train_override_model,
            size=args.train_size,
            epoch_size=args.train_epoch_size,
            batch_size=args.train_batch_size,
            min_mixure=args.train_min_mixure,
            max_mixure=args.train_max_mixure,
            save_model_every=args.train_save_model_every,
            print_progress_every=args.train_print_progress_every,
        )
    if args.test_sample:
        create_sample()
    if args.test_enable:
        test(
            size=args.train_size,
            batch_size=args.train_batch_size,
            print_progress_every=args.train_print_progress_every,
        )
