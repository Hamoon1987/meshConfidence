import argparse

parser = argparse.ArgumentParser(description = 'Classifier Model')

parser.add_argument(
    '--epochs',
    default = '60',
    type = int,
    help = 'Number of Training epochs'
)
parser.add_argument(
    '--batch_size',
    default = '32',
    type = int,
    help = 'Batch size'
)

args = parser.parse_args()