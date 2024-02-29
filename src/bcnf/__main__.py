import argparse
import sys


def main(argv: str = None) -> None:
    """
    Parse the command line arguments for commands and options
    """

    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description='Ballistic Conditional Normalizing Flows (BCNF)')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # TODO: Remove demo command
    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument('--dummy_option', type=str, default='dummy_value', help='Dummy option')

    # Evaluate input
    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'demo':
            print(f'Running demo with dummy_option={args.dummy_option}')
        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
