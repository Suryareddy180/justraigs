from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os, os.path as osp
from tqdm import tqdm
import shutil
pd.set_option('mode.chained_assignment',None)


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Split Data')
    parser.add_argument('--csv_path_in', type=str, default='data/JustRAIGS_Train_labels.csv',
                        help='path to training csvs')
    parser.add_argument('--csvs_path_out', type=str, default='data/', help='path to store k-fold csvs')
    # parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=True, help='avoid saving anything')

    args = parser.parse_args()

    return args



def main(args):
    df = pd.read_csv(args.csv_path_in)
    real_image_path = '/kaggle/input/prepro-images/ODIR_PREPROCESSED'
    df.image_id = [osp.join(real_image_path, n ) for n in df.image_id.values]
    


    num_ims = len(df)
    meh, df_val1 = train_test_split(df, test_size=num_ims // 5, random_state=0)
    meh, df_val2 = train_test_split(meh, test_size=num_ims // 5, random_state=0)
    meh, df_val3 = train_test_split(meh, test_size=num_ims // 5, random_state=0)
    df_val5, df_val4 = train_test_split(meh, test_size=num_ims // 5, random_state=0)

    df_train1 = pd.concat([df_val2, df_val3, df_val4, df_val5], axis=0)
    df_train2 = pd.concat([df_val1, df_val3, df_val4, df_val5], axis=0)
    df_train3 = pd.concat([df_val1, df_val2, df_val4, df_val5], axis=0)
    df_train4 = pd.concat([df_val1, df_val2, df_val3, df_val5], axis=0)
    df_train5 = pd.concat([df_val1, df_val2, df_val3, df_val4], axis=0)

    df_train1.to_csv(osp.join(args.csvs_path_out, 'tr1.csv'), index=None)
    df_val1.to_csv(osp.join(args.csvs_path_out, 'vl1.csv'), index=None)

    df_train2.to_csv(osp.join(args.csvs_path_out, 'tr2.csv'), index=None)
    df_val2.to_csv(osp.join(args.csvs_path_out, 'vl2.csv'), index=None)

    df_train3.to_csv(osp.join(args.csvs_path_out, 'tr3.csv'), index=None)
    df_val3.to_csv(osp.join(args.csvs_path_out, 'vl3.csv'), index=None)

    df_train4.to_csv(osp.join(args.csvs_path_out, 'tr4.csv'), index=None)
    df_val4.to_csv(osp.join(args.csvs_path_out, 'vl4.csv'), index=None)

    df_train5.to_csv(osp.join(args.csvs_path_out, 'tr5.csv'), index=None)
    df_val5.to_csv(osp.join(args.csvs_path_out, 'vl5.csv'), index=None)

    



if __name__ == "__main__":
    args = get_args_parser()
    main(args)
