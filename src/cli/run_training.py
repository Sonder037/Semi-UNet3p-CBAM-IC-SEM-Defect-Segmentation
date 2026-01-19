#!/usr/bin/env python
"""
Entry point script to run the training pipeline
"""

import argparse
import sys
import os

# Add the project root to the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cli.train import main as train_main


def main():
    parser = argparse.ArgumentParser(description='Run Semi-UNet3+-CBAM training pipeline')
    parser.add_argument('--config', type=str, default='../configs/default_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # Call the actual training main function
    # Temporarily override sys.argv to match train.py expectations
    original_argv = sys.argv[:]
    sys.argv = ['run_training.py', '--config', args.config, '--device', args.device]
    
    try:
        train_main()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()