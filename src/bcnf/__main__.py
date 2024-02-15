import argparse
import sys

from bcnf.debug_plotting import debug_plotting
from bcnf.physics import physics_simulation


def main(argv: str = None) -> None:
    """
    Parse the command line arguments for commands and options
    """

    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description='Ballistic Conditional Normalizing Flows (BCNF)')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # answerQuestion
    demo_parser = subparsers.add_parser("demo")
    test_physics_parser = subparsers.add_parser("test_physics")

    demo_parser.add_argument('--dummy_option', type=str, default='dummy_value', help='Dummy option')
    test_physics_parser.add_argument('--x0', type=float, nargs=3, default=[0, 0, 1.8], help='Initial position')

    # Evaluate input
    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'demo':
            print(f'Running demo with dummy_option={args.dummy_option}')
        case 'test_physics':
            print('Running physics simulation')
            p = physics_simulation(x0=args.x0)
            debug_plotting(p)
        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
