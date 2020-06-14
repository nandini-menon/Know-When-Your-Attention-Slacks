import os
import random
import pandas as pd


def split_train_test(file_name, label):
    training_set_file = '../data/training_set.csv'
    test_set_file = '../data/test_set.csv'

    df = pd.read_csv(file_name)
    n = len(df.index)
    split_val = (int)(0.8 * n)

    # Shuffling the data
    shuffled_df = df.sample(frac=1)

    train_df = shuffled_df.iloc[:split_val]
    train_label = label * len(train_df.index)
    train_df = train_df.assign(Label=pd.Series(train_label).values)
    if os.path.isfile(training_set_file):
        train_df.to_csv(training_set_file, mode='a', header=False, index=False)
    else:
        train_df.to_csv(training_set_file, index=False)

    test_df = shuffled_df.iloc[split_val:]
    test_label = label * len(test_df.index)
    test_df = test_df.assign(Label=pd.Series(test_label).values)
    if os.path.isfile(test_set_file):
        test_df.to_csv(test_set_file, mode='a', header=False, index=False)
    else:
        test_df.to_csv(test_set_file, index=False)

    print(f'{file_name} Done!')


def main():
    mode = 2
    count = 0
    while count < 36:
        if mode == 1:
            mode = 2
            filename = f'../data/csv_inputs/Subject{str(count).zfill(2)}_2.csv'
            label = [1]
            count += 1
        elif mode == 2:
            mode = 1
            filename = f'../data/csv_inputs/Subject{str(count).zfill(2)}_1.csv'
            label = [0]
        split_train_test(filename, label)

if __name__ == '__main__':
    main()
