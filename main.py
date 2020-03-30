import tensorflow as tf
import argparse
import os

from models import build_union_feature_model
import global_variables as G

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to be used')
parser.add_argument('--loss', default='mse', type=str, help='loss function')
parser.add_argument('--csv-dir', default='./data', type=str, help='csv path')
parser.add_argument('--save-dir', default='./train', type=str, help='csv path')
parser.add_argument('--batch-size', default=256, type=int, help='batch size')
parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
parser.add_argument('--num-epochs', default=20, type=int, help='epoch number')
parser.add_argument('--memory-growth', action="store_true", default=False)
args = parser.parse_args()


def create_dataset(csv_path, batch_size=64, num_epochs=20, shuffle=True):
    """create_dataset
    Used specifically for ./data/merged.csv
    """    
    csv_files = [os.path.join(csv_path, 'merged.csv')]
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=csv_files,
        batch_size=batch_size,
        label_name=G.LABEL_NAME,
        select_columns=G.COLUMNS,
        num_epochs=num_epochs,
        shuffle=shuffle)
    return dataset


def main():
    # if not os.path.exists('./logs'):
    #     os.makedirs('./logs')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Set GPU configuration
    if tf.test.is_built_with_cuda():
        if args.memory_growth:
            gpu_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        if len(args.gpu) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = build_union_feature_model()
    adam = tf.keras.optimizers.Adam(lr=args.lr)
    model.compile(loss=args.loss, optimizer=adam, metrics=['mae', 'mse'])
    if '{}.h5'.format(model.name) in os.listdir(args.save_dir):
        model.load_weights(os.path.join(args.save_dir, model.name + '.h5'))
        print(' [*] Loaded pretrained model {}'.format(model.name))
    dataset = create_dataset(args.csv_dir, batch_size=args.batch_size, num_epochs=args.num_epochs)
    model.fit(dataset, epochs=args.num_epochs)
    model.save_weights(os.path.join(args.save_dir, model.name + '.h5'))


if __name__ == '__main__':
    main()