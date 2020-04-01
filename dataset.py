import tensorflow as tf
import os

import global_variables as G


def create_dataset_from_folder(csv_path, batch_size=64, num_epochs=20, shuffle=True):
    """create_dataset_from_folder
    Used when multiple csv files are stored under the same directory
    For evaluating, set num_epochs=1 and shuffle=False
    """
    csv_files = os.listdir(csv_path)
    csv_files = [os.path.join(csv_path, fn) for fn in csv_files]
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=csv_files,
        batch_size=batch_size,
        label_name=G.LABEL_NAME,
        select_columns=G.COLUMNS.copy(),
        na_value='?',
        num_epochs=num_epochs,
        shuffle=shuffle)
    return dataset


def create_dataset_from_file(csv_path, 
                             batch_size=64, 
                             num_epochs=20, 
                             shuffle=False, 
                             label_name=G.LABEL_NAME):
    """create_dataset_from_file
    Used when one single csv file are given
    NOTE: Fucking Google this API directly modify the list given to `select_columns`
    """
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=csv_path,
        batch_size=batch_size,
        select_columns=G.COLUMNS.copy(),
        label_name=label_name,
        num_epochs=num_epochs,
        shuffle=shuffle)
    return dataset


def build_input_features(input_op):
    import functools
    # functor to preprocess continuous features
    def process_continuous_data(mean, var, data):
        data = (tf.cast(data, tf.float32) - mean) / var
        return tf.reshape(data, [-1, 1])
    
    # Compared with the codes given in tf2.0 documentary, 
    # this can guarantee the correctness of the feature order
    feature_columns = []
    for col in G.COLUMNS:
        if col in G.CATEGORIES.keys():
            print('[*] processing column key={} type=CATEGORICAL'.format(col))
            cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
                key=col, vocabulary_list=G.CATEGORIES[col])
            feature = tf.feature_column.indicator_column(cat_col)
        elif col in G.MEANVARS.keys():
            print(' [*] processing column key={} type=NUMERICAL'.format(col))
            norm_fn = functools.partial(process_continuous_data, *G.MEANVARS[col])
            feature = tf.feature_column.numeric_column(key=col, normalizer_fn=norm_fn)
        else:
            continue
        feature_columns.append(feature)

    # preprocessing layer in keras
    preprocessing_layer = tf.keras.layers.DenseFeatures(feature_columns)

    # start to build input feature models
    preprocessed = preprocessing_layer(input_op)
    return preprocessed


if __name__ == '__main__':
    dataset = create_tmp_dataset('./data/merged.csv')
    batch_xs, batch_ys = next(iter(dataset))
    proc_xs = build_input_features(batch_xs)

    print('\n')
    print('batch_xs:', batch_xs)
    print('\n')
    print('proc_xs:', proc_xs)
    print('\n')
    print('batch_ys:', batch_ys)