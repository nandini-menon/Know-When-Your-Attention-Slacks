import pyedflib
import numpy as np


def convert_to_csv(data_root, base_filename):
    f = pyedflib.EdfReader(f'{data_root}/edf_inputs/{base_filename}.edf')
    csv_file = open(f'{data_root}/csv_inputs/{base_filename}.csv', "w+")

    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signal_labels_row = ",".join(signal_labels)
    csv_file.write(signal_labels_row + '\n')

    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

    sigbufs = sigbufs.transpose()

    for row in sigbufs:
        sigbufs_list = row.tolist()
        format_sigbufs_list = ['%.4f' % elem for elem in sigbufs_list]
        sigbufs_str = ','.join(format_sigbufs_list)
        csv_file.write(sigbufs_str + '\n')

    print(f'{base_filename} Done!')
    csv_file.close()
    f._close()
    del f


def main():
    data_root = '../data'
    mode = 2
    count = 0
    while count < 36:
        if mode == 1:
            mode = 2
            base_filename = f'Subject{str(count).zfill(2)}_2'
            count += 1
        elif mode == 2:
            mode = 1
            base_filename = f'Subject{str(count).zfill(2)}_1'
        convert_to_csv(data_root, base_filename)

if __name__ == '__main__':
    main()
