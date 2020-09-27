import scipy.io as sio
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from nilearn.connectome import ConnectivityMeasure

def drop_sites(data, label, sites = [2, 7, 8]):
    dropped_label_index = np.isin(label['Site'], sites)
    dropped_subject_ID = label.iloc[dropped_label_index]['subject_ID']
    label.drop(label[dropped_label_index].index, axis=0, inplace=True)
    dropped_data_index = np.isin(data['subject_ID'], dropped_subject_ID)
    data.drop(data[dropped_data_index].index, axis=0, inplace=True)
    # print("squeezed_subject_number",np.unique(data['subject_ID']).shape[0])
    # print("dropped site: ", sites)
    return data, label

def site_count_by_class(label):
    label.loc[label['DX'] > 1, 'DX'] = 1
    groups = label.groupby('Site')
    result = pd.DataFrame()
    for _, site_group in groups:
        name = site_group['Site'].values[0]
        hc_group = site_group.loc[site_group['DX'] == 0]
        adhd_group = site_group.loc[site_group['DX'] == 1]

        count_hc = np.unique(hc_group['subject_ID']).shape[0]
        count_adhd = np.unique(adhd_group['subject_ID']).shape[0]

        assert count_adhd + count_hc == np.unique(site_group['subject_ID']).shape[0], "shape wrong"
        df_temp = pd.DataFrame({'hc': count_hc, 'adhd': count_adhd}, index = [name])
        result = pd.concat([result, df_temp])
    return result


def shuffle_index(data):
    subject_count = np.unique(data['subject_ID'])
    subject_id_shuffled = np.random.permutation(subject_count)
    train_subject_id = subject_id_shuffled[:-100]
    val_subject_id = subject_id_shuffled[-100:]
    return train_subject_id, val_subject_id

def generate_labels(label_raw, data):
    label = pd.DataFrame()
    label_raw = label_raw[['DX', 'Site', 'subject_ID', 'Age', 'Gender', 'Verbal_IQ', 'Performance_IQ','Full4_IQ']] # sort label index so that the sequence match data
    label_raw.fillna(0, inplace=True)
    subject_ID_list = np.unique(data['subject_ID'])
    for subjectID in subject_ID_list:
        num_time_step = sum(data['subject_ID'] == subjectID)
        temp = pd.concat([label_raw[label_raw['subject_ID'] == subjectID]] * num_time_step) # repeat [label]*time_step
        label = pd.concat([label, temp], ignore_index = True)
    return label


def load_ADHD():
    ADHD_path = '../data/ADHD'
    if not os.path.exists(os.path.join(ADHD_path, "training_data.csv")):
        # read data
        data = pd.DataFrame()
        data_path = 'ADHD200_training_'
        for i in range(8):
            data_chunk = pd.read_csv(os.path.join(ADHD_path, data_path + str(i)+'.csv'))
            data = pd.concat([data, data_chunk], axis =0, ignore_index = True)
        print("load train data success!")
        test_data = pd.read_csv(os.path.join(ADHD_path, 'ADHD200_testing.csv'))

        # read label
        label = pd.read_csv(os.path.join(ADHD_path, 'phenotypic_train.csv'))
        label['subject_ID'] = np.arange(label.shape[0])
        test_label = pd.read_csv(os.path.join(ADHD_path, 'phenotypic_test.csv'))

        # drop several sites
        data, label = drop_sites(data, label)
        test_data, test_label = drop_sites(test_data, test_label)
        test_site = test_label['Site']

        print("train count:\n", site_count_by_class(label))
        print("test count:\n", site_count_by_class(test_label))

        #shuffle subject_ID
        train_subject_id, val_subject_id = shuffle_index(data)

        # train, val, test split
        train_data = data.iloc[np.isin(data['subject_ID'], train_subject_id)]
        val_data = data.iloc[np.isin(data['subject_ID'], val_subject_id)]
        print("split train/val data success!")

        #generate labels
        train_label = label.iloc[np.isin(label['subject_ID'], train_subject_id)]
        val_label = label.iloc[np.isin(label['subject_ID'], val_subject_id)]

        train_label = generate_labels(train_label, train_data)
        val_label = generate_labels(val_label, val_data)
        test_label = generate_labels(test_label, test_data)
        print("generate train/val/test labels success!")

        train_data.to_csv(os.path.join(ADHD_path, 'training_data.csv'), index=False)
        val_data.to_csv(os.path.join(ADHD_path, 'validating_data.csv'), index=False)
        test_data.to_csv(os.path.join(ADHD_path, 'testing_data.csv'), index=False)
        train_label.to_csv(os.path.join(ADHD_path, 'training_label.csv'), index=False)
        val_label.to_csv(os.path.join(ADHD_path, 'validating_label.csv'), index=False)
        test_label.to_csv(os.path.join(ADHD_path, 'testing_label.csv'), index=False)
        test_site.to_csv(os.path.join(ADHD_path, 'testing_sites.csv'), index=False)
        print("save data success!")
    else:
        train_data = pd.read_csv(os.path.join(ADHD_path, 'training_data.csv'))
        val_data = pd.read_csv(os.path.join(ADHD_path, 'validating_data.csv'))
        test_data = pd.read_csv(os.path.join(ADHD_path, 'testing_data.csv'))
        train_label = pd.read_csv(os.path.join(ADHD_path, 'training_label.csv'))
        val_label = pd.read_csv(os.path.join(ADHD_path, 'validating_label.csv'))
        test_label = pd.read_csv(os.path.join(ADHD_path, 'testing_label.csv'))
        test_site = pd.read_csv(os.path.join(ADHD_path, 'testing_sites.csv'), header=None)

    data = {'train_data': train_data, 'train_label': train_label,
            'val_data': val_data, 'val_label': val_label,
            'test_data': test_data, 'test_label': test_label,
            'test_site': test_site}
    print("Load raw ADHD success!")
    return data

if __name__ == '__main__':
    dict_df = load_ADHD()
