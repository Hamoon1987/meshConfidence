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
parser.add_argument(
    '--mean',
    default = [0.1641, 0.1455, 0.0864, 0.0867, 0.1424, 0.1554, 0.3645, 0.2034, 0.1010, 0.1051, 0.1781, 0.3030, 0.0639, 0.6305],
    type = float,
    help = 'Training data mean'
)
parser.add_argument(
    '--std',
    default = [6.5141e-02, 5.4406e-02, 4.7398e-03, 4.5912e-03, 6.2194e-02, 5.9077e-02,3.5126e-01, 1.7716e-01, 1.4936e-02, 1.0990e-02, 1.4384e-01, 2.8130e-01, 1.8599e-03, 7.2290e-01],
    type = float,
    help = 'Training data std'
)


args = parser.parse_args()