import argparse

parser = argparse.ArgumentParser(description = 'Classifier Model')

parser.add_argument(
    '--pw3d_train_path',
    default = '/SPINH/classifier/3dpw_train.csv',
    type = str,
    help = 'Path to 3dpw_train data'
)
parser.add_argument(
    '--pw3d_test_path',
    default = '/SPINH/classifier/3dpw_test.csv',
    type = str,
    help = 'Path to 3dpw_test data'
)
parser.add_argument(
    '--all_train_path',
    default = '/SPINH/classifier/all_train.csv',
    type = str,
    help = 'Path to all_train data'
)
parser.add_argument(
    '--oh3d_test_path',
    default = '/SPINH/classifier/3doh_test.csv',
    type = str,
    help = 'Path to 3dpw_train data'
)
parser.add_argument(
    '--h36m_p1_test_path',
    default = '/SPINH/classifier/h36m-p1_test.csv',
    type = str,
    help = 'Path to 3dpw_train data'
)
parser.add_argument(
    '--epochs',
    default = '50',
    type = int,
    help = 'Number of Training epochs'
)
parser.add_argument(
    '--batch_size',
    default = '32',
    type = int,
    help = 'Batch size'
)
parser.add_argument(
    '--split_ratio',
    default = '0.8',
    type = float,
    help = 'Training validation ratio'
)
parser.add_argument(
    '--tt_split_ratio',
    default = '0.5',
    type = float,
    help = 'Test Training ratio'
)
args = parser.parse_args()