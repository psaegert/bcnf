import argparse
import sys


def main(argv: str = None) -> None:
    """
    Parse the command line arguments for commands and options
    """

    # Parse the command line arguments for commands and options
    parser = argparse.ArgumentParser(description='Ballistic Conditional Normalizing Flows (BCNF)')
    subparsers = parser.add_subparsers(dest='command_name', required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    train_parser.add_argument('-o', '--output-dir', type=str, default=None, help='Path to the directory to store the results')
    train_parser.add_argument('-f', '--force', action='store_true', help='Overwrite the output directory if it exists')

    # Evaluate input
    args = parser.parse_args(argv)

    # Execute the command
    match args.command_name:
        case 'train':
            import json
            import os

            import torch

            from bcnf import CondRealNVP
            from bcnf.train import Trainer
            from bcnf.utils import get_dir, load_config, sub_root_path

            model_name = os.path.basename(args.config).split('.')[0]

            if args.output_dir is None:
                args.output_dir = os.path.join("{{BCNF_ROOT}}", 'models', 'bcnf-models', model_name)
                if not os.path.exists(get_dir(args.output_dir)):
                    os.makedirs(get_dir(args.output_dir))

            resolved_output_path = sub_root_path(args.output_dir)

            if os.path.exists(resolved_output_path) and len(os.listdir(resolved_output_path)) > 0 and not args.force:
                print(f"Output directory {resolved_output_path} already exists and is not empty. Use -f to overwrite.")
                sys.exit(1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            config = load_config(args.config)

            model = CondRealNVP.from_config(config).to(device)

            print(f'Loaded {model_name} with {model.n_params:,} parameters')

            trainer = Trainer(
                config={k.lower(): v for k, v in config.to_dict().items()},
                project_name="bcnf-test",
                parameter_index_mapping=model.parameter_index_mapping,
                verbose=True,
            )

            model = trainer.train(model)

            torch.save(model.state_dict(), os.path.join(resolved_output_path, "state_dict.pt"))

            with open(os.path.join(resolved_output_path, 'config.json'), 'w') as f:
                json.dump({'config_path': args.config}, f)

            print(f"Model saved to {resolved_output_path}")

        case _:
            print('Unknown command: ', args.command)
            sys.exit(1)
