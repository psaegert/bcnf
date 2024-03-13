import argparse
import sys

from bcnf.utils import get_dir


def main(argv: str = None) -> None:
    """
    Parse the command line arguments for commands and options
    """

    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description='Ballistic Conditional Normalizing Flows (BCNF)')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # TODO: Remove demo command
    demo_parser = subparsers.add_parser("run-trainer")
    demo_parser.add_argument('--dummy_option', type=str, default='dummy_value', help='Dummy option')

    # Evaluate input
    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'run-trainer':
            from dynaconf import Dynaconf

            from bcnf.train.trainer import Trainer

            config_file_path = f'{get_dir()}/configs/trainer_config.yaml'
            config = Dynaconf(settings_files=config_file_path)

            trainer = Trainer(config=config, project_name='bcnf-test')

            trainer.training_pipeline()

        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
