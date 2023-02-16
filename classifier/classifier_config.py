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
    default = [0.1574, 0.1396, 0.0828, 0.0832, 0.1366, 0.1484, 0.3617, 0.2000, 0.0988, 0.1021, 0.1727, 0.2965, 0.0620, 0.6291],
    type = float,
    help = 'Training data mean'
)
parser.add_argument(
    '--std',
    default = [5.8423e-02, 4.9447e-02, 2.2038e-03, 2.6574e-03, 5.7095e-02, 5.1747e-02,
        3.5120e-01, 1.7479e-01, 1.3567e-02, 9.0699e-03, 1.3944e-01, 2.7739e-01,
        1.1868e-03, 7.2544e-01],
    type = float,
    help = 'Training data std'
)


args = parser.parse_args()