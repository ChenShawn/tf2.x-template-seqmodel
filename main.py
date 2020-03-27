import tensorflow as tf
import argparse
import os

from models import build_union_feature_model
import global_variables as G

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to be used')
parser.add_argument('--loss', default='mse', type=str, help='loss function')
parser.add_argument('--csv-dir', default='./data', type=str, help='csv path')
parser.add_argument('--save-dir', default='./train', type=str, help='csv path')
parser.add_argument('--batch-size', default=64, type=int, help='batch size')
parser.add_argument('--num-epochs', default=20, type=int, help='epoch number')
args = parser.parse_args()


def create_dataset(csv_path, batch_size=64, num_epochs=20, shuffle=True):
    """create_dataset
    Used specifically for ./data/data.csv
    """
    LABEL_NAME = 'amount'
    # Define categorical features here, first line is an example
    oldcols = [
        'diff', 'cancel_num', 'flight_num',
        'bkd_acc_c_sum', 'bkd_acc_c_real_sum', 'bkd_acc_q_sum',
        'bkd_acc_q_real_sum', 'income_acc_c_real_avg', 'all_seats', 'direction',
        'uv', 'pv_1', 'pv_2'
    ]
    COLUMNS = []
    for day in range(1, 8):
        for col in oldcols:
            COLUMNS.append('day_{}_{}'.format(day, col))
    CATEGORIES = {'cityid': list(range(12))}
    for day in range(1, 8):
        CATEGORIES['day_{}_direction'.format(day)] = [u'进港', u'出港']
    MEANVARS = {}
    for col in COLUMNS:
        if 'cityid' not in col and 'direction' not in col and 'amount' not in col:
            MEANVARS[col] = [500.0, 500.0]

    csv_files = os.listdir(csv_path)
    csv_files = [os.path.join(csv_path, fn) for fn in csv_files]
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=csv_files,
        batch_size=batch_size,
        label_name=LABEL_NAME,
        select_columns=COLUMNS,
        num_epochs=num_epochs,
        shuffle=shuffle)
    return dataset


def main():
    # if not os.path.exists('./logs'):
    #     os.makedirs('./logs')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = build_union_feature_model()
    model.compile(loss=args.loss, optimizer='adam', metrics=['mae', 'mse'])
    dataset = create_dataset(args.csv_dir, 
            batch_size=args.batch_size, num_epochs=args.num_epochs)
    model.fit(dataset, epochs=args.num_epochs)
    model.save(args.save_dir)


if __name__ == '__main__':
    main()