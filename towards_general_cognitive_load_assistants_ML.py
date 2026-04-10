import os
import re
import sys 
import csv
import time 
import math
import pytz
import pywt
import pathlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import neurokit2 as nk
from neurokit2.misc._warnings import NeuroKitWarning as NKWarning

import scipy
import scipy.stats as stats
import scipy.signal as signal
from scipy.signal import find_peaks
from scipy.signal import iirnotch, filtfilt
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.spatial import distance

from sklearn import preprocessing
from sklearn.preprocessing import normalize,MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer


participants_wild_tasks_dir = [
    {   # Participant 01
        # Empty
    }, { # Participant 02
        'game1_04': ['2023-11-30 18:50:52.544625', '2023-11-30 19:00:19.470937', 'vlw_mw', 'nor_st'],
        'paper1_04': ['2023-11-30 19:03:26.455278', '2023-11-30 19:23:31.522106', 'vlw_mw', 'vlw_st'],
        'summary_paper1_04' : ['2023-11-30 19:24:42.880505', '2023-11-30 18:34:49.880505', 'vlw_mw', 'vlw_st'],
        'game2_04': ['2023-11-30 19:37:52.544625', '2023-11-30 19:47:59.470937', 'lw_mw', 'lw_st'],
        'paper2_04': ['2023-11-30 19:49:56.455278', '2023-11-30 19:59:59.522106', 'lw_mw', 'nor_st'],
        'summary_paper2_04' : ['2023-11-30 20:00:42.880505', '2023-11-30 20:10:32.880505', 'lw_mw', 'nor_st'],
        'game3_04': ['2023-11-30 20:49:52.544625', '2023-11-30 20:59:59.470937', 'hg_mw', 'hg_st'],
        'paper3_04': ['2023-11-30 21:02:26.455278', '2023-11-30 21:22:31.522106', 'vhg_mw', 'vhg_st'],
        'summary_paper3_04' : ['2023-11-30 21:23:42.880505', '2023-11-30 21:31:49.880505', 'vhg_mw', 'vhg_st'],
        'game4_04': ['2023-11-30 21:33:52.544625', '2023-11-30 21:43:59.470937', 'nor_mw', 'nor_st'],
        'paper4_04': ['2023-11-30 21:45:56.455278', '2023-11-30 22:55:59.522106', 'nor_mw', 'hg_st'],
        'summary_paper4_04' : ['2023-11-30 21:56:42.880505', '2023-11-30 22:04:32.880505', 'nor_mw', 'hg_st'],
        'game5_04': ['2023-12-05 18:41:02.544625', '2023-12-05 18:51:09.470937', 'lw_mw', 'lw_st'],
        'paper5_04': ['2023-12-05 18:52:36.455278', '2023-12-05 19:05:37.512702', 'vlw_mw', 'vlw_st'],
        'summary_paper5_04' : ['2023-12-05 19:05:52.880505', '2023-12-05 19:10:32.880505', 'vlw_mw', 'vlw_st'],
    }, { # Participant 03
        'game_02': ['2023-11-13 21:48:46.000', '2023-11-13 21:58:26.000', 'nor_mw', 'lw_st'],
        'paper_02': ['2023-11-13 22:02:16.000', '2023-11-13 22:22:46.000', 'hg_mw', 'hg_st'],
        'summary_paper_02' : ['2023-11-13 22:24:46.000', '2023-11-13 22:34:51.000', 'hg_mw', 'hg_st'],
        'game2_02': ['2023-11-13 22:37:46.000', '2023-11-13 22:47:53.000', 'hg_mw', 'nor_st'],
        'paper2_02': ['2023-11-13 22:50:34.000', '2023-11-13 23:10:46.000', 'vhg_mw', 'hg_st'],
        'summary_paper2_02' : ['2023-11-13 23:13:12.000', '2023-11-13 23:23:22.000', 'vhg_mw', 'hg_st'],
        'paper3_02': ['2023-11-13 23:26:34.000', '2023-11-13 23:46:46.000', 'vhg_mw', 'hg_st'],
        'summary_paper3_02' : ['2023-11-13 23:47:44.000', '2023-11-13 23:57:16.000', 'vhg_mw', 'hg_st'],
        'game3_02': ['2023-11-14 19:49:08.000', '2023-11-14 19:59:34.000', 'nor_mw', 'nor_st'],
        'paper4_02': ['2023-11-14 20:04:24.000', '2023-11-14 20:24:36.000', 'hg_mw', 'hg_st'],
        'summary_paper4_02' : ['2023-11-14 20:26:08.000', '2023-11-14 20:36:43.000', 'hg_mw', 'hg_st'],
        'game4_02': ['2023-11-14 20:39:43.000', '2023-11-14 20:50:03.000', 'hg_mw', 'vhg_st'],
        'paper5_02': ['2023-11-14 20:53:34.000', '2023-11-14 21:13:56.000', 'hg_mw', 'hg_st'],
        'summary_paper5_02' : ['2023-11-14 21:15:06.000', '2023-11-14 21:25:47.000', 'hg_mw', 'hg_st']
    }, { # Participant 04
        'game_10': ['2024-03-13 20:50:10.000', '2024-03-13 20:59:50.000', 'lw_mw', 'lw_st'],
        'paper_10': ['2024-03-13 21:06:16.000', '2024-03-13 21:25:13.000', 'nor_mw', 'hg_st'],
        'summary_paper_10' : ['2024-03-13 21:25:36.000', '2024-03-13 21:31:51.000', 'nor_mw', 'hg_st'],
        'paper2_10': ['2024-03-13 21:37:16.000', '2024-03-13 21:50:13.000', 'nor_mw', 'hg_st'],
        'summary_paper2_10' : ['2024-03-13 21:50:36.000', '2024-03-13 21:58:51.000', 'nor_mw', 'hg_st'],
        'game2_10': ['2024-03-15 09:11:16.000', '2024-03-15 09:19:53.000', 'hg_mw', 'hg_st'],
        'paper3_10': ['2024-03-15 09:25:34.000', '2024-03-15 09:48:17.000', 'nor_mw', 'nor_st'],
        'summary_paper3_10' : ['2024-03-15 09:48:52.000', '2024-03-15 09:58:52.000', 'nor_mw', 'nor_st'],
        'paper4_10': ['2024-03-15 10:00:34.000', '2024-03-15 10:11:46.000', 'nor_mw', 'nor_st'],
        'summary_paper4_10' : ['2024-03-15 10:11:54.000', '2024-03-15 10:16:36.000', 'nor_mw', 'nor_st'],
        'paper5_10': ['2024-03-15 10:21:24.000', '2024-03-15 10:37:36.000', 'nor_mw', 'lw_st'],
        'summary_paper5_10' : ['2024-03-15 10:37:48.000', '2024-03-15 10:43:13.000', 'nor_mw', 'lw_st'],
        'paper6_10': ['2024-03-15 10:48:14.000', '2024-03-15 11:09:36.000', 'nor_mw', 'lw_st'],
        'summary_paper6_10' : ['2024-03-15 11:09:47.000', '2024-03-15 11:17:47.000', 'nor_mw', 'lw_st']
    }, { # Participant 05
        'game1_01': ['2023-11-04 13:12:52.544625', '2023-11-04 13:22:49.470937', 'nor_mw', 'nor_st'],
        'paper1_01': ['2023-11-04 13:24:26.455278', '2023-11-04 13:44:31.522106', 'nor_mw', 'nor_st'],
        'summary_paper1_01' : ['2023-11-04 13:44:42.880505', '2023-11-04 13:50:32.880505', 'nor_mw', 'nor_st'],
        'game2_01': ['2023-11-04 14:03:22.544625', '2023-11-04 14:13:34.470937', 'nor_mw', 'lw_st'],
        'paper2_01': ['2023-11-04 14:15:16.455278', '2023-11-04 14:35:15.522106', 'nor_mw', 'nor_st'],
        'summary_paper2_01' : ['2023-11-04 14:35:42.880505', '2023-11-04 14:45:32.880505', 'nor_mw', 'nor_st'],
        'game3_01': ['2023-11-06 15:59:12.544625', '2023-11-06 16:09:29.470937', 'hg_mw', 'lw_st'],
        'paper3_01': ['2023-11-06 16:12:26.455278', '2023-11-06 16:32:43.522106', 'hg_mw', 'nor_st'],
        'summary_paper3_01' : ['2023-11-06 16:33:52.880505', '2023-11-06 16:43:59.880505', 'hg_mw', 'nor_st'],
        'game4_01': ['2023-11-06 16:46:47.880505','2023-11-06 16:56:59.880505', 'hg_mw', 'nor_st'],
        'paper4_01': ['2023-11-06 16:58:36.455278', '2023-11-06 17:18:37.512702', 'hg_mw', 'hg_st'],
        'summary_paper4_01' : ['2023-11-06 17:18:52.880505', '2023-11-06 17:28:32.880505', 'hg_mw', 'hg_st'],
        'game5_01': ['2023-11-06 17:34:37.880505','2023-11-06 17:44:39.880505', 'vhg_mw', 'vhg_st'],
        'paper5_01': ['2023-11-06 17:46:12.455278', '2023-11-06 17:58:26.522106', 'hg_mw', 'hg_st'],
        'summary_paper5_01' : ['2023-11-06 17:58:42.880505', '2023-11-06 18:03:32.880505', 'hg_mw', 'hg_st'],
    }, { # Participant 06
        'game1_06': ['2024-02-28 22:43:00', '2024-02-28 22:53:00', 'vlw_mw', 'vlw_st'],
        'paper1_06': ['2024-02-28 22:55:00', '2024-02-28 23:15:00', 'vlw_mw', 'vlw_st'],
        'summary_paper1_06' : ['2024-02-28 23:17:00', '2024-02-28 23:27:00', 'vlw_mw', 'vlw_st'],
        'game2_06': ['2024-03-03 13:06:00', '2024-03-03 13:16:00', 'vlw_mw', 'vlw_st'],
        'paper2_06': ['2024-03-03 13:16:00', '2024-03-03 13:36:00', 'vlw_mw', 'vlw_st'],
        'paper3_06': ['2024-03-03 13:41:00', '2024-03-03 14:01:00', 'vhg_mw', 'vhg_st'],
        'paper4_06': ['2024-03-03 14:10:00', '2024-03-03 14:30:00', 'vlw_mw', 'vlw_st'],
        'summary_paper4_06' : ['2024-03-03 14:32:00', '2024-03-03 14:42:00', 'vlw_mw', 'vlw_st'],
    }, { # Participant 07
        'game1_07': ['2024-02-19 22:32:52.544625', '2024-02-19 22:42:49.470937', 'vlw_mw', 'lw_st'],
        'paper1_07': ['2024-02-19 22:47:52.544625', '2024-02-19 22:58:49.470937', 'nor_mw', 'nor_st'],
        'summary_paper1_07' : ['2024-02-19 23:04:52.544625', '2024-02-19 23:14:49.470937', 'nor_mw', 'nor_st'],
        'game2_07': ['2024-02-21 07:23:22.544625', '2024-02-21 07:33:34.470937', 'lw_mw', 'lw_st'],
        'paper2_07': ['2024-02-21 07:35:02.544625', '2024-02-21 07:55:04.470937', 'vlw_mw', 'vlw_st'],
        'summary_paper2_07' : ['2024-02-21 07:56:42.544625', '2024-02-21 08:06:34.470937', 'vlw_mw', 'vlw_st'],
        'game3_07': ['2024-02-21 10:28:22.544625', '2024-02-21 10:38:34.470937', 'nor_mw', 'lw_st'],
        'paper3_07': ['2024-02-21 10:41:22.544625', '2024-02-21 11:01:34.470937', 'hg_mw', 'lw_st'],
        'summary_paper3_07' : ['2024-02-21 11:02:42.544625', '2024-02-21 11:12:34.470937', 'hg_mw', 'lw_st']
    }, { # Participant 08
        'game1_03': ['2023-11-26 17:51:52.544625', '2023-11-26 18:01:59.470937', 'nor_mw', 'lw_st'],
        'paper1_03': ['2023-11-26 18:03:26.455278', '2023-11-26 18:24:31.522106', 'nor_mw', 'nor_st'],
        'summary_paper1_03' : ['2023-11-26 18:26:42.880505', '2023-11-26 18:37:49.880505', 'nor_mw', 'nor_st'],
        'paper2_03': ['2023-11-26 18:41:16.455278', '2023-11-26 19:01:35.522106', 'hg_mw', 'hg_st'],
        'summary_paper2_03' : ['2023-11-26 19:05:42.880505', '2023-11-26 19:15:32.880505', 'hg_mw', 'hg_st'],
        'game2_03': ['2023-11-27 00:48:32.544625', '2023-11-27 00:58:52.470937', 'nor_mw', 'lw_st'],
        'paper3_03': ['2023-11-27 01:04:46.455278', '2023-11-27 01:24:47.522106', 'nor_mw', 'nor_st'], 
        'summary_paper3_03' : ['2023-11-27 01:37:02.880505', '2023-11-27 01:47:19.880505', 'nor_mw', 'nor_st'],
        'paper4_03': ['2023-11-27 01:52:26.455278', '2023-11-27 02:10:43.522106', 'nor_mw', 'lw_st'],
        'summary_paper4_03' : ['2023-11-27 02:10:52.880505', '2023-11-27 02:19:59.880505', 'nor_mw', 'lw_st'],
        'game5_03': ['2023-11-28 13:07:12.544625', '2023-11-28 13:17:29.470937', 'nor_mw', 'nor_st'],
        'paper5_03': ['2023-11-28 13:21:36.455278', '2023-11-28 13:41:37.512702', 'nor_mw', 'nor_st'],
        'summary_paper5_03' : ['2023-11-28 13:47:52.880505', '2023-11-28 13:57:32.880505', 'nor_mw', 'nor_st']
    }, { # Participant 09
        'game1_05': ['2024-01-27 20:49:52.544625', '2024-01-27 20:59:49.470937', 'nor_mw', 'lw_st'],
        'paper1_05': ['2024-01-27 21:03:26.455278', '2024-01-27 21:23:31.522106', 'nor_mw', 'lw_st'],
        'summary_paper1_05' : ['2024-01-27 21:24:42.880505', '2024-01-27 21:34:32.880505', 'nor_mw', 'lw_st'],
        'paper2_05': ['2024-01-27 21:38:16.455278', '2024-01-27 21:58:15.522106', 'hg_mw', 'nor_st'],
        'summary_paper2_05' : ['2024-01-27 21:59:42.880505', '2024-01-27 22:10:32.880505', 'hg_mw', 'nor_st'],
    }, { # Participant 10
        'game1_06': ['2024-02-13 17:32:52.544625', '2024-02-13 17:42:49.470937', 'lw_mw', 'nor_st'],
        'paper1_06': ['2024-02-03 17:48:26.455278', '2024-02-13 18:08:31.522106', 'hg_mw', 'nor_st'],
        'summary_paper1_06' : ['2024-02-13 18:14:42.880505', '2024-02-13 18:24:32.880505', 'hg_mw', 'nor_st'],
        'game2_06': ['2024-02-13 21:43:52.544625', '2024-02-13 21:53:59.470937', 'nor_mw', 'nor_st'],
        'paper2_06': ['2024-02-13 22:00:52.544625', '2024-02-13 22:20:49.470937', 'hg_mw', 'hg_st'], 
        'summary_paper2_06' : ['2024-02-13 22:26:52.544625', '2024-02-13 22:36:49.470937', 'hg_mw', 'hg_st'],
        'game3_06': ['2024-02-15 21:08:52.544625', '2024-02-15 21:18:59.470937', 'nor_mw', 'lw_st'],
        'paper3_06': ['2024-02-15 21:23:52.544625', '2024-02-15 21:43:59.470937', 'hg_mw', 'vhg_st'],
        'summary_paper3_06' : ['2024-02-15 21:50:52.544625', '2024-02-15 22:00:59.470937', 'lw_mw', 'vhg_st'],
        'game4_06': ['2024-02-16 00:16:52.544625', '2024-02-16 00:26:59.470937', 'hg_mw', 'lw_st'],
        'paper4_06': ['2024-02-16 00:29:52.544625', '2024-02-16 00:49:59.470937', 'hg_mw', 'vhg_st'],
        'summary_paper4_06' : ['2024-02-16 00:54:52.544625', '2024-02-16 01:04:59.470937', 'hg_mw', 'vhg_st'],
    }
]

NOTCH_B, NOTCH_A = butter(4, np.array([45, 55]) / (256 / 2), btype='bandstop')


for i, arg in enumerate(sys.argv):
    print("Argument %d: %s" % (i, arg))
    if i == 1:
        sensors = str(arg)
    elif i == 2:
        classification_here = str(arg)
    elif i == 3:
        feature_reduction = str(arg)
    elif i == 4:
        with_activity_labels = str(arg)
    else:
        print("NOT TAKING CARE OF argument %d: %s" % (i, arg))

binary_classification = (classification_here == 'two')
three_class_classification = (classification_here == 'three')
five_class_classification = (classification_here == 'five')

use_e4_data = (sensors == 'e4') or (sensors == 'both')
use_muse_data = (sensors == 'muse') or (sensors == 'both')
use_both_data = use_e4_data and use_muse_data

top_stat_features_only = (feature_reduction == 'top_features_only')
use_activity_labels = (with_activity_labels == 'with_activity_labels')

feature_reduction_str = 'with_feature_reduction' if top_stat_features_only else 'without_feature_reduction'
with_activity_labels_str = 'with_activity_labels' if use_activity_labels else 'without_activity_labels'

# Suppress the EDA filtering warning because the sampling rate of 4 Hz is too low. The original warning is: "neurokit2/eda/eda_clean.py:105: NeuroKitWarning: EDA signal is sampled at very low frequency. Skipping filtering."
warnings.simplefilter("ignore", category=NKWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Level value of 1 is too high: all coefficients will experience boundary effects.") # This occurs for PyWT at times 

model_name = 'cv_generalized_loo_participant_%s_model_%s_as_%s_class_model_%s_and_%s.csv'
reults_path = '../parsed/ml_results/%s/' % sensors
base_path = '../raw/dataset/'

EDA_SFREQ = 4
BVP_SFREQ = 64
TEMP_SFREQ = 4
EEG_SFREQ = 256

start_time = int(time.time())
print(start_time)
lab_log_dictionary = None


def most_common(lst):
    return max(set(lst), key=lst.count)


def format_empatica_file(df, modality, timezone='CET'):
    initial_time = pd.to_datetime(df.iloc[0, 0], unit='s').tz_localize('UTC')
    sample_rate = df.iloc[1, 0]
    print('Sample Rate of modality %s: %s' % (modality, sample_rate))

    timezone = pytz.timezone(timezone)
    initial_time = initial_time.tz_convert(timezone)

    df = df.iloc[2:].reset_index(drop=True)
    num_data_points = len(df)
    timestamp = pd.date_range(initial_time, periods=num_data_points, freq=f'{1/sample_rate*1000000}us')
    df.insert(0, 'Timestamp', timestamp)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
    return df


def bvp_to_hr(input_df):
    input_df['Timestamp'] = pd.to_datetime(input_df['Timestamp'])
    input_df['Time_diff'] = input_df['Timestamp'].diff().dt.total_seconds().fillna(0)
    sampling_rate = round(1 / input_df['Time_diff'].mode()[0])
    bvp_signal = input_df['BVP'] - input_df['BVP'].mean()
    peaks, _ = find_peaks(bvp_signal, distance=sampling_rate*0.5)
    peak_intervals_seconds = np.diff(input_df['Timestamp'].iloc[peaks].values).astype('timedelta64[ms]').astype('float') / 1000
    heart_rates = 60 / peak_intervals_seconds
    hr_df = pd.DataFrame({
        'Timestamp': input_df['Timestamp'].iloc[peaks[:-1]],
        'HR': heart_rates
    })

    result_df = pd.merge_asof(input_df, hr_df, on='Timestamp', direction='forward').bfill()

    result_df.drop('Time_diff', axis=1, inplace=True)

    return result_df


def notch_filter(df, channels, b, a):
    filtered_df = df.copy()
    for channel in channels:
        filtered_df[channel] = filtfilt(b, a, df[channel])

    return filtered_df


def relative_wavelet_energy(coeffs):
    return np.sqrt(np.sum(coeffs ** 2)) / len(coeffs)


# Wavelet-Log-Energy taken from: Assessment of Mental Workload Using a Transformer Network and Two Prefrontal EEG Channels: An Unparameterized Approach
def wavelet_log_energy(coeffs):
    return np.sum(np.asarray([np.log(n ** 2) for n in coeffs]))


# Zero-Crossing after: Multimodal wearable EEG, EMG and accelerometry measurements improve the accuracy of tonic-clonic seizure detection
def zero_crossing(data):
    data = data - np.mean(data)
    difference = np.diff(np.sign(data) > 0)
    return np.sum(abs(difference))


def eeg_sample_extraction(notched_eeg_sample):
    sample_features = {'db2': None, 'haar': None}
    
    # For each of DB2 and Haar as mother wavelets
    for mother_wavelet in ['db2', 'haar']:
        wavelets_features = {
            'Notchd_AF7' : None,
            'Notchd_AF8' : None,
            'Notchd_TP9' : None,
            'Notchd_TP10' : None,
            'Mean' : None
        }

        # For each of the channels (AF7, AF8, TP9, TP10); And once also make a 'projection' by taking the mean of channels
        for channel_name in ['Notchd_AF7', 'Notchd_AF8', 'Notchd_TP9', 'Notchd_TP10', 'Mean']:
            wavelet_features_for_channel = {
                'cA8' : None, 
                'cD8' : None, 
                'cD7' : None, 
                'cD6' : None, 
                'cD5' : None, 
                'cD4' : None, 
                'cD3' : None, 
                'cD2' : None, 
                'cD1' : None
            }

            local_channel = None

            if channel_name == 'Mean':
                local_channel = np.mean(np.asarray([notched_eeg_sample['Notchd_AF7'], notched_eeg_sample['Notchd_AF8'], notched_eeg_sample['Notchd_TP9'], notched_eeg_sample['Notchd_TP10']]), axis=0)
            else:
                local_channel = notched_eeg_sample[channel_name]

            cA8, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(local_channel, mother_wavelet, level=8)
            coefficients = [
                (cA8, 'cA8'), (cD8, 'cD8'), (cD7, 'cD7'), (cD6, 'cD6'), (cD5, 'cD5'), 
                (cD4, 'cD4'), (cD3, 'cD3'), (cD2, 'cD2'), (cD1, 'cD1'),
            ]

            # Due to "RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable."
            # Filter out that warning here for the Kurtosis calculation specifically only for that line!
            warnings.simplefilter('ignore',lineno=324)

            for coeffs, name in coefficients:                
                coefficient_features = {
                    'STD' : np.std(coeffs),
                    'MEAN' : np.mean(coeffs),
                    'MIN' : np.min(coeffs),
                    'MAX' : np.max(coeffs),
                    'Skewness' : scipy.stats.skew(np.squeeze(coeffs)),
                    'RelativeWaveletEnergy' : relative_wavelet_energy(coeffs),
                    'Kurtosis' : scipy.stats.kurtosis(coeffs, axis=-1, fisher=True),
                    'ZeroCrossing' : zero_crossing(coeffs)
                }

                wavelet_features_for_channel[name] = coefficient_features
            wavelets_features[channel_name] = wavelet_features_for_channel
        sample_features[mother_wavelet] = wavelets_features

    return sample_features


def eda_custom_process(eda_signal, sampling_rate=2, method="neurokit"):
    eda_signal = nk.signal_sanitize(eda_signal)
    
    # Series check for non-default index
    if type(eda_signal) is pd.Series and type(eda_signal.index) != pd.RangeIndex:
        eda_signal = eda_signal.reset_index(drop=True)
    
    # Preprocess
    eda_cleaned = eda_signal  #Add your custom cleaning module here or skip cleaning
    eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate)
    # print(eda_decomposed)

    if max(eda_decomposed["EDA_Phasic"].values) <= 0.1:
        peak_signal, info = None, None
    else:
        # Find peaks
        peak_signal, info = nk.eda_peaks(
            eda_decomposed["EDA_Phasic"].values,
            sampling_rate=sampling_rate,
            method=method,
            amplitude_min=0.1,
        )
        info['sampling_rate'] = sampling_rate  # Add sampling rate in dict info

    # Store
    signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})
    signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

    return signals, info


def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:20,.5f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def time_align_features(eeg_tuples=None, hr_tuples=None, eda_tuples=None, temp_tuples=None, offset_in_seconds=3):
    window_offset_allowed_in_seconds = offset_in_seconds
    time_aligned_lab_features = []
    hr_idx_ctr = 0
    eda_idx_ctr = 0
    temp_idx_ctr = 0

    if use_e4_data:
        for idx, eda_lab_sample in enumerate(eda_tuples):
            sample_holder = [eda_lab_sample]
            eda_lab_sample_datetime = eda_lab_sample[0].to_pydatetime()
            for j, hr_lab_sample in enumerate(hr_tuples):
                if j < hr_idx_ctr:
                    continue
                else:
                    if (eda_lab_sample_datetime - hr_lab_sample[0].to_pydatetime()).total_seconds() <= window_offset_allowed_in_seconds:
                        sample_holder.append(hr_lab_sample)
                        hr_idx_ctr = j
                        break
            for j, temp_lab_sample in enumerate(temp_tuples):
                if j < temp_idx_ctr:
                    continue
                else:
                    if (eda_lab_sample_datetime - temp_lab_sample[0].to_pydatetime()).total_seconds() <= window_offset_allowed_in_seconds:
                        sample_holder.append(temp_lab_sample)
                        temp_idx_ctr = j
                        break
            if len(sample_holder) == 3:
                time_aligned_lab_features.append(sample_holder)
    if use_both_data:
        for idx, lab_eeg_sample in enumerate(eeg_tuples):
            sample_holder = [lab_eeg_sample]
            eeg_lab_sample_datetime = datetime.strptime(lab_eeg_sample[0], '%Y-%m-%d %H:%M:%S.%f')
            for j, hr_lab_sample in enumerate(hr_tuples):
                if j < hr_idx_ctr:
                    continue
                else:
                    if (eeg_lab_sample_datetime - hr_lab_sample[0].to_pydatetime()).total_seconds() <= window_offset_allowed_in_seconds:
                        sample_holder.append(hr_lab_sample)
                        hr_idx_ctr = j
                        break
            for j, eda_lab_sample in enumerate(eda_tuples):
                if j < eda_idx_ctr:
                    continue
                else:
                    if (eeg_lab_sample_datetime - eda_lab_sample[0].to_pydatetime()).total_seconds() <= window_offset_allowed_in_seconds:
                        sample_holder.append(eda_lab_sample)
                        eda_idx_ctr = j
                        break
            for j, temp_lab_sample in enumerate(temp_tuples):
                if j < temp_idx_ctr:
                    continue
                else:
                    if (eeg_lab_sample_datetime - temp_lab_sample[0].to_pydatetime()).total_seconds() <= window_offset_allowed_in_seconds:
                        sample_holder.append(temp_lab_sample)
                        temp_idx_ctr = j
                        break
            if len(sample_holder) == 4:
                time_aligned_lab_features.append(sample_holder)

    return time_aligned_lab_features


def extract_routine_timestamps(file_path, suffix):
    patterns = {
        'relaxationvideo': r"Routine Relaxation-Video (started|finished) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)",
        'stroop': r"Routine Stroop level HARD (started|finished) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)",
        'nback': r"Routine N-Back Trial \(False\) (started|finished) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)",
        'paper': r"Routine Reading_Paper (started|finished) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)",
        'story': r"Routine Reading_Story (started|finished) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)",
        'summarystory': r"Routine Summary_Story (started|finished) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)",
        'summarypaper': r"Routine Summary_Paper (started|finished) at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"
    }

    tasks = {}
    with open(file_path, 'r') as file:
        for line in file:
            for routine, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    time_stamp = match.group(2)
                    if routine not in tasks:
                        tasks[routine] = [None, None]
                    if "started" in line:
                        tasks[routine][0] = time_stamp
                    elif "finished" in line:
                        tasks[routine][1] = time_stamp

    tasks = {key + suffix: value for key, value in tasks.items()}

    return tasks


# Return wild task type as well as the one-hot encoded activity array. This array includes the following:
# Relaxation, LoadTask, Summary, Reading, Game (both for wild and lab it's obviously the same)
# [0, 0, 0, 0, 0]
def determine_wild_task_type_and_activity(timestamp, tasks, buffer_seconds=59, buffer_microseconds=999999):
    timestamp = pd.Timestamp(timestamp)

    work_with_mw_labels = True
    return_number = 0
    for task, (start, end, mw_label, st_label) in tasks.items():
        start_with_buffer = pd.Timestamp(start) - (0.5 * pd.Timedelta(seconds=buffer_seconds, microseconds=buffer_microseconds))
        end_with_buffer = pd.Timestamp(end) + (0.5 * pd.Timedelta(seconds=buffer_seconds, microseconds=buffer_microseconds))

        if start_with_buffer <= timestamp <= end_with_buffer:
            if work_with_mw_labels:
                if mw_label == 'vhg_mw':
                    return_number = 5
                    continue
                elif mw_label == 'hg_mw':
                    return_number = 4
                    continue
                elif mw_label == 'nor_mw':
                    return_number = 3
                    continue
                elif mw_label == 'lw_mw':
                    return_number = 2
                    continue
                elif mw_label == 'vlw_mw':
                    return_number = 1
                    continue
                else:
                    raise Exception('Unkown label assigned for MW! %s' % mw_label)
            else:
                print('NOT YET IMPLEMENTED!')   # I.e., work with stress labels

    activity_array = [0, 0, 0, 0, 0] # Relaxation, LoadTask, Summary, Reading, Game
    if 'game' in task:
        activity_array = [0, 0, 0, 0, 1]
    elif 'summary' in task:
        activity_array = [0, 0, 1, 0, 0]
    elif 'paper' in task:
        activity_array = [0, 0, 0, 1, 0]

    if return_number != 0:
        if use_activity_labels:
            return [return_number, activity_array]
        else:
            return return_number

    print('Would assigning label Unknown to task %s as it is %s away from start at %s and %s away from end at %s' % (task, str(start_with_buffer - timestamp), str(start_with_buffer), str(end_with_buffer - timestamp), str(end_with_buffer)))
    print('But now chose to just use the labels from the dictionary for the time: %s!' % str(timestamp))

    smallest_distance_mw_label = None   # arbitrary for now, will be overwritten
    smallest_distance = pd.Timedelta(days=10)  # set to arbitrarily high value
    for dirtest in participants_wild_tasks_dir:
        for key in dirtest.keys():
            start_from_key, end_from_key, mw_label_from_key, st_label_from_key = dirtest[key]
            dist_to_start = abs(pd.Timestamp(start_from_key) - timestamp)
            dist_to_end = abs(pd.Timestamp(end_from_key) - timestamp)
            if (dist_to_start <= smallest_distance) or (dist_to_end <= smallest_distance):
                smallest_distance = dist_to_start if dist_to_start < dist_to_end else dist_to_end
                smallest_distance_mw_label = mw_label_from_key

    if smallest_distance_mw_label is not None:
        if mw_label == 'vhg_mw':
            smallest_distance_mw_label = 5
        elif mw_label == 'hg_mw':
            smallest_distance_mw_label = 4
        elif mw_label == 'nor_mw':
            smallest_distance_mw_label = 3
        elif mw_label == 'lw_mw':
            smallest_distance_mw_label = 2
        elif mw_label == 'vlw_mw':
            smallest_distance_mw_label = 1

    print('Will now return %s' % 'Unknown' if smallest_distance_mw_label is None else str(smallest_distance_mw_label))

    if use_activity_labels:
        return ['Unknown', activity_array] if smallest_distance_mw_label is None else [smallest_distance_mw_label, activity_array]
    else:
        return 'Unknown' if smallest_distance_mw_label is None else smallest_distance_mw_label

# Return lab task type as well as the one-hot encoded activity array. This array includes the following:
# Relaxation, LoadTask, Summary, Reading, Game (both for wild and lab it's obviously the same)
# [0, 0, 0, 0, 0]
def determine_lab_task_type_and_activity(timestamp, tasks, lab_logs_files_dicts):

    if isinstance(timestamp, pd.Series):
        timestamp = pd.Timestamp(timestamp[0])    
    else:
        timestamp = pd.Timestamp(timestamp)
    
    if len(lab_logs_files_dicts) == 0:
        print('Will have to use pre-defined labels. Might affect accuracy, insights gained, etc.')
        task = task_key.split('_')[0]  
        if ('relaxation' in task):
            if use_activity_labels:
                return [['vlw_mw'], [1, 0, 0, 0, 0]]     # return ['vlw_mw', 'vlw_st']
            else:
                return ['vlw_mw']
        elif ('nback' in task) or ('stroop' in task):
            if use_activity_labels:
                return [['hg_mw'], [0, 1, 0, 0, 0]]     # return ['vhg_mw', 'vhg_st']
            else:
                return ['hg_mw']
        elif 'summary' in task:
            if use_activity_labels:
                return [['nor_mw'], [0, 0, 1, 0, 0]]     # return ['vhg_mw', 'vhg_st']
            else:
                return ['nor_mw']
        elif ('story' in task) or ('paper' in task):
            if use_activity_labels:
                return [['vhg_mw'], [0, 0, 0, 1, 0]]     # return ['vlw_mw', 'vlw_st']
            else:
                return ['vhg_mw']

        print('Encountered an unexpected issue here. Investigate on the labeling from lab')
        exit(-10)
    else:
        the_labs_mw_ratings = []
        the_labs_activities_times = []
        overall_start = None

        for lab_log_dictionary in lab_logs_files_dicts:
            lab_log_absolute_start_time = pd.Timestamp(lab_log_dictionary['absolute_start_time_overall'])
            lab_log_maximum_theoretical_end_of_session = lab_log_absolute_start_time + pd.Timedelta(hours=4) # Safe, as no experiment lasted this long and all had multiple days in between
            temporary_log_times = []
            earliest_timestamp_this_recording, latest_timestamp_this_recording = None, None
            individual_absolute_timestamps = []

            for likert_trial in lab_log_dictionary['likert_scale_dictionary'].values():
                is_mw = True if likert_trial['likert_type'] == 'mental_effort' else False
                if is_mw: # For now, care only about the mental effort labels
                    rating = likert_trial['Likert_Scale_Rating_mental_effort']
                    questionnaire_time_relative_to_start = likert_trial['likert_start_timestamp']
                    questionnaire_absolute_time = lab_log_absolute_start_time + pd.Timedelta(questionnaire_time_relative_to_start)
                    individual_absolute_timestamps.append(questionnaire_absolute_time)
                    temporary_log_times.append([questionnaire_absolute_time, questionnaire_time_relative_to_start, rating])

            earliest_timestamp_this_recording = min(individual_absolute_timestamps)
            latest_timestamp_this_recording = max(individual_absolute_timestamps)

            the_labs_mw_ratings.append([earliest_timestamp_this_recording, latest_timestamp_this_recording, temporary_log_times])

            overall_start = pd.Timestamp(lab_log_dictionary['absolute_start_time_overall'])
            the_labs_activities_times.append(['stroop', lab_log_dictionary['stroop_exercise_dictionary']['stroop_trial_0']['stroop_trial_start_timestamp'], lab_log_dictionary['stroop_exercise_dictionary']['stroop_trial_0']['stroop_trial_end_timestamp']])
            the_labs_activities_times.append(['nback', lab_log_dictionary['n_back_exercise_dictionary']['n_back_trial_0']['n_back_trial_start_timestamp'], lab_log_dictionary['n_back_exercise_dictionary']['n_back_trial_0']['n_back_trial_end_timestamp']])
            the_labs_activities_times.append(['relaxation-video', lab_log_dictionary['relaxation_video_dictionary']['relaxation_video_start_timestamp'], lab_log_dictionary['relaxation_video_dictionary']['relaxation_video_stop_timestamp']])

    for task_key, times in tasks.items():
        start, end = times  
        start = pd.Timestamp(start)
        end = pd.Timestamp(end) + pd.Timedelta(seconds=59, microseconds=999999)

        for earliest_timestamp_this_recording, latest_timestamp_this_recording, temporary_log_times in the_labs_mw_ratings:
            if (earliest_timestamp_this_recording - pd.Timedelta(minutes=30)) <= timestamp <= (latest_timestamp_this_recording + pd.Timedelta(minutes=30)):
                distance_to_timestamp_searched = []
                dictionary_distance_to_absolute_time_and_label = {}
                for questionnaire_absolute_time, _, rating in temporary_log_times:
                    distance_to_timestamp_searched.append(str(abs(timestamp - questionnaire_absolute_time)))
                    dictionary_distance_to_absolute_time_and_label[str(abs(timestamp - questionnaire_absolute_time))] = [questionnaire_absolute_time, rating]

                label_closest_to_timestamp = min(distance_to_timestamp_searched)
                long_name_label = dictionary_distance_to_absolute_time_and_label[str(label_closest_to_timestamp)][1]

                # determine activity array:
                # Relaxation, LoadTask, Summary, Reading, Game
                activity_array = None
                for activity_entry in the_labs_activities_times:
                    activity_name = activity_entry[0]
                    activity_start = overall_start + activity_entry[1]
                    activity_end = overall_start + activity_entry[2]
                    if activity_start <= timestamp <= activity_end: # activity is known and has to be either one of the stroop, nback, or relaxation video
                        if (activity_name == 'stroop') or (activity_name == 'nback'):
                            activity_array = [0, 1, 0, 0, 0]
                        else: 
                            activity_array = [1, 0, 0, 0, 0]

                if activity_array == None:  # did not yet find the right activity, so it has to be either reading or summarizing...
                    stroop_end = overall_start + the_labs_activities_times[0][2]
                    nback_end = overall_start + the_labs_activities_times[1][2]
                    relaxation_video_end = overall_start + the_labs_activities_times[2][2]

                    # if the task is under 20 mins later than an activity, it must be reading. If it's more than that it is summarizing
                    if (relaxation_video_end <= timestamp <= (relaxation_video_end + pd.Timedelta(minutes=20))) or (stroop_end <= timestamp <= (stroop_end + pd.Timedelta(minutes=20))) or (nback_end <= timestamp <= (nback_end + pd.Timedelta(minutes=20))):
                        activity_array = [0, 0, 0, 1, 0]
                    else:
                        activity_array = [0, 0, 1, 0, 0]


                if long_name_label == 'very, very high':
                    if use_activity_labels:
                        return [['vhg_mw'], activity_array]
                    else:
                        return ['vhg_mw']
                elif long_name_label == 'high':
                    if use_activity_labels:
                        return [['hg_mw'], activity_array]
                    else:
                        return ['hg_mw']
                elif long_name_label == 'neither low nor high':
                    if use_activity_labels:
                        return [['nor_mw'], activity_array]
                    else:
                        return ['nor_mw']
                elif long_name_label == 'low':
                    if use_activity_labels:
                        return [['lw_mw'], activity_array]
                    else:
                        return ['lw_mw']
                elif long_name_label == 'very, very low':
                    if use_activity_labels:
                        return [['vlw_mw'], activity_array]
                    else:
                        return ['vlw_mw']

                print(long_name_label)
                exit(-1) # Some issues happened and we could not convert labels; investigate!

    print('No suitable time in the logs') # Meaning, we did not find a suitable timestamp in the logs!!
    return 'Unknown'


# This method, as is written now, automatically removes all the UNKNOWNS (i.e., data where no label is known)
def combine_timestamps_and_features_and_labels(aligned_lab_features, lab_tasks, lab_load_label_function=True, lab_logs_files_dicts=None):
    combined_samples_timestamp_to_np_features_label_tuple = {}
    for sensed_data_tuple in aligned_lab_features:
        sensed_data_dict = sensed_data_tuple

        if use_muse_data or use_both_data:
            start_time_tuple = sensed_data_dict['Start_Time_EEG']
        if use_e4_data:
            start_time_tuple = sensed_data_dict['skt_start_time']

        if isinstance(start_time_tuple, pd.Series):
            start_time_tuple = start_time_tuple[0]

        features_label = None
        if lab_load_label_function:
            if use_activity_labels:
                features_label_and_activity = determine_lab_task_type_and_activity(start_time_tuple, lab_tasks, lab_logs_files_dicts)
                features_label = features_label_and_activity[0]
                activity_array = features_label_and_activity[1]
            else:
                features_label = determine_lab_task_type_and_activity(start_time_tuple, lab_tasks, lab_logs_files_dicts)
            if features_label == 'Unknown':
                continue    # don't consider this data where there is no label known
        else:
            if use_activity_labels:
                features_label_and_activity = determine_wild_task_type_and_activity(start_time_tuple, lab_tasks)
                features_label = features_label_and_activity[0]
                activity_array = features_label_and_activity[1]
            else:
                features_label = determine_wild_task_type_and_activity(start_time_tuple, lab_tasks)
            if features_label == 'Unknown':
                continue    # don't consider this data where there is no label known

        if use_activity_labels:
            # Activity-labels: Relaxation, LoadTask, Summary, Reading, Game
            sensed_data_dict['Relaxation'] = activity_array[0]
            sensed_data_dict['LoadTask'] = activity_array[1]
            sensed_data_dict['Summary'] = activity_array[2]
            sensed_data_dict['Reading'] = activity_array[3]
            sensed_data_dict['Game'] = activity_array[4]

        combined_samples_timestamp_to_np_features_label_tuple[start_time_tuple] = [features_label, sensed_data_dict]

    return combined_samples_timestamp_to_np_features_label_tuple


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def compute_band_powers(eegdata, fs):
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape

    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    ind_delta, = np.where(f < 4)
    meanDelta = np.mean(PSD[ind_delta, :], axis=0)
    # Theta 4-8
    ind_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(PSD[ind_theta, :], axis=0)
    # Alpha 8-12
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
    # Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(PSD[ind_beta, :], axis=0)
    # Gamma 30-45
    ind_gamma, = np.where((f >= 30) & (f < 45))
    meanGamma = np.mean(PSD[ind_gamma, :], axis=0)

    feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha, meanBeta, meanGamma), axis=0)

    feature_vector = np.log10(feature_vector)

    return feature_vector


def handcrafted_features_extraction(participant_number=-1, eeg_samples=None, eda_samples=None, hr_samples=None, temp_samples=None, min_len_samples=0):
    all_modality_samples_features_tuple = []
    all_eeg_samples_features_tuple =  None    
    all_temp_samples = None
    all_eda_samples = None
    all_hr_samples = None
    theoretical_feature_num_counter = 0

    #-#EEG Features:
    #-#    - Engagement Index, from EEG‑based measurement system for monitoring student engagement in learning 4.0; https://doi.org/10.1038/s41598-022-09578-y
    #-#    - (Holm et al., 2009) / Brainbeat
    #-#    - (Bayrambas & Sendurur, 2023)
    #-#    - (Giannakakis et al., 2019)
    #-#    - (Parent et al., 2020)

    if eeg_samples is not None:
        all_eeg_samples_features_tuple = []
        for eeg_sample in eeg_samples:
            start_time_sample = pd.to_datetime(eeg_sample['TimeStamp'])
            (DELTA_IDX, THETA_IDX, ALPHA_IDX, BETA_IDX, GAMMA_IDX) = (0, 1, 2, 3, 4)
            EEG_FS = 256
            # Start by notch-filtering the EEG
            f0 = 50
            b, a = iirnotch(f0, Q=30, fs=EEG_FS)
            channels = ['RAW_AF7', 'RAW_AF8', 'RAW_TP9', 'RAW_TP10']
            notched_eeg_sample = notch_filter(eeg_sample, channels, b, a)

            mean_channel = np.expand_dims(np.mean(np.asarray([notched_eeg_sample['RAW_AF7'], notched_eeg_sample['RAW_AF8'], notched_eeg_sample['RAW_TP9'], notched_eeg_sample['RAW_TP10']]), axis=0), axis=-1)
            prefrontal_mean_eeg = np.expand_dims(np.mean(np.asarray([notched_eeg_sample['RAW_AF7'], notched_eeg_sample['RAW_AF8']]), axis=0), axis=-1)
            temporal_mean_eeg = np.expand_dims(np.mean(np.asarray([notched_eeg_sample['RAW_TP9'], notched_eeg_sample['RAW_TP10']]), axis=0), axis=-1)
            mean_feature_vector = compute_band_powers(mean_channel, EEG_FS)
            prefrontal_feature_vector = compute_band_powers(prefrontal_mean_eeg, EEG_FS)
            temporal_feature_vector = compute_band_powers(temporal_mean_eeg, EEG_FS)
            AF7_feature_vector = compute_band_powers(np.expand_dims(notched_eeg_sample['RAW_AF7'], axis=-1), EEG_FS)
            AF8_feature_vector = compute_band_powers(np.expand_dims(notched_eeg_sample['RAW_AF8'], axis=-1), EEG_FS)
            TP9_feature_vector = compute_band_powers(np.expand_dims(notched_eeg_sample['RAW_TP9'], axis=-1), EEG_FS)
            TP10_feature_vector = compute_band_powers(np.expand_dims(notched_eeg_sample['RAW_TP10'], axis=-1), EEG_FS)

            sample_features = {
                'Start_Time_EEG': start_time_sample,
                'EngagementIndex': np.squeeze((mean_feature_vector[BETA_IDX] / (mean_feature_vector[THETA_IDX] + mean_feature_vector[ALPHA_IDX]))),
                'BrainBeat': np.squeeze(prefrontal_feature_vector[THETA_IDX] / temporal_feature_vector[ALPHA_IDX]), # theoretically theta Fz/alpha Pz ratio, but we don't have these channels. So use the closest approximation thereof: theta prefrontal / alpha temporal
                'CLI': np.squeeze(mean_feature_vector[THETA_IDX] / mean_feature_vector[ALPHA_IDX]),
                'AsymmetryIndexAllChannels': np.squeeze(np.mean((notched_eeg_sample['RAW_AF7'] + notched_eeg_sample['RAW_TP9']) - (notched_eeg_sample['RAW_AF8'] + notched_eeg_sample['RAW_TP10']))),
                'AsymmetryIndexPrefrontal': np.squeeze(np.mean((notched_eeg_sample['RAW_AF7']) - (notched_eeg_sample['RAW_AF8']))),
                'AsymmetryIndexTemporal': np.squeeze(np.mean((notched_eeg_sample['RAW_TP9']) - (notched_eeg_sample['RAW_TP10']))),
                'PrefrontalDeltaPower': np.squeeze(prefrontal_feature_vector[DELTA_IDX]),
                'PrefrontalThetaPower': np.squeeze(prefrontal_feature_vector[THETA_IDX]),
                'PrefrontalAlphaPower': np.squeeze(prefrontal_feature_vector[ALPHA_IDX]),
                'PrefrontalBetaPower': np.squeeze(prefrontal_feature_vector[BETA_IDX]),
                'PrefrontalGammaPower': np.squeeze(prefrontal_feature_vector[GAMMA_IDX]),
                'TemporalDeltaPower': np.squeeze(temporal_feature_vector[DELTA_IDX]),
                'TemporalThetaPower': np.squeeze(temporal_feature_vector[THETA_IDX]),
                'TemporalAlphaPower': np.squeeze(temporal_feature_vector[ALPHA_IDX]),
                'TemporalBetaPower': np.squeeze(temporal_feature_vector[BETA_IDX]),
                'TemporalGammaPower': np.squeeze(temporal_feature_vector[GAMMA_IDX]),
            }

            notched_eeg_sample.rename(columns={'RAW_AF7': 'Notchd_AF7'}, inplace=True)
            notched_eeg_sample.rename(columns={'RAW_AF8': 'Notchd_AF8'}, inplace=True)
            notched_eeg_sample.rename(columns={'RAW_TP9': 'Notchd_TP9'}, inplace=True)
            notched_eeg_sample.rename(columns={'RAW_TP10': 'Notchd_TP10'}, inplace=True)

            dwt_eeg_samples = eeg_sample_extraction(notched_eeg_sample)
            np_features = []
            for mother_wavelet in ['db2', 'haar']:
                for channel in ['Notchd_AF7', 'Notchd_AF8', 'Notchd_TP9', 'Notchd_TP10', 'Mean']:
                    for coefficient in ['cA8', 'cD8', 'cD7', 'cD6', 'cD5', 'cD4', 'cD3', 'cD2', 'cD1']:
                        for feature in ['STD', 'MEAN', 'MIN', 'MAX', 'Skewness', 'RelativeWaveletEnergy', 'Kurtosis', 'ZeroCrossing']:
                            sample_features["{}-{}-{}-{}".format(mother_wavelet, channel, coefficient, feature)] = dwt_eeg_samples[mother_wavelet][channel][coefficient][feature]

            all_eeg_samples_features_tuple.append(sample_features)

        theoretical_feature_num_counter += 17

    #-#EDA/PPG features:
    #-#    - SCR and SCL;
    #-#    - TEMP features
    #-#    - HRV_MeanNN    HRV_SDNN   HRV_RMSSD   HRV_LFn   HRV_HFn  HRV_ratio_LFn_HFn
    #-#    - Heart-Rate;
    #-#    - LF/HF;
    #-#    - SDNN;
    #-#    - HF power;

    if eda_samples is not None:
        all_eda_samples = []
        for sample in eda_samples: # Has to be for each of EDA, HR, and SKT
            sample_features = {
                'min_scr' : None,
                'max_scr' : None,
                'num_scr_peaks' : None,
                'min_scl' : None,
                'max_scl' : None,
                'mean_scl' : None
            }
            if int(participant_number) == 9:
                _, info = eda_custom_process(sample['EDA'].to_list(), sampling_rate=EDA_SFREQ)  # Due to a recording issue of low sampling rate
            else: 
                _, info = nk.eda_process(sample['EDA'].to_list(), sampling_rate=EDA_SFREQ)
            if info is not None:
                min_scr = np.min(np.asarray(info['SCR_Amplitude']))
                max_scr = np.max(np.asarray(info['SCR_Amplitude']))
                num_scr_peaks = len(info['SCR_Peaks'])
                scl = nk.eda_phasic(nk.standardize(sample['EDA'].to_list()), sampling_rate=EDA_SFREQ)
                min_scl = min(scl['EDA_Tonic'])
                max_scl = max(scl['EDA_Tonic'])
                mean_scl = np.mean(scl['EDA_Tonic'])
            else:
                min_scr = 0
                max_scr = 0
                num_scr_peaks = 0
                min_scl = 0
                max_scl = 0
                mean_scl = 0
            sample_features['min_scr'] = min_scr
            sample_features['max_scr'] = max_scr
            sample_features['num_scr_peaks'] = num_scr_peaks
            sample_features['min_scl'] = min_scl
            sample_features['max_scl'] = max_scl
            sample_features['mean_scl'] = mean_scl
            all_eda_samples.append(sample_features)

        theoretical_feature_num_counter += 6

    if temp_samples is not None:
        all_temp_samples = []
        for sample in temp_samples:
            start_time_sample = sample.index[0]
            min_skt = min(sample['TEMP'])
            max_skt = max(sample['TEMP'])
            mean_skt = np.mean(sample['TEMP'])
            skt_features = {
                'skt_start_time': start_time_sample,
                'min_skt' : min_skt,
                'max_skt' : max_skt,
                'mean_skt' : mean_skt
            }
            all_temp_samples.append(skt_features)

        theoretical_feature_num_counter += 4

    if hr_samples is not None:
        all_hr_samples = []
        for sample in hr_samples:
            sample_features = {
                'HRV_MeanNN':None,
                'HRV_SDNN':None,
                'HRV_RMSSD':None,
                'HRV_LF':None,
                'HRV_HF':None,
                'HRV_ratio_LF_HF':None,
                'Min_Heart-Rate': None,
                'Mean_Heart-Rate': None,
                'Max_Heart-Rate': None
            }
            hrv_param=[]
            index_error_occured = False
            try:
                info = nk.ppg_findpeaks(np.squeeze(sample['BVP']), sampling_rate=64, show=False)
            except IndexError:
                index_error_occured = True
            if (0 <= len(info['PPG_Peaks']) <= 3) or index_error_occured:
                # Less than 3 peaks found; add dummy values here!
                sample_features = {
                    'HRV_MeanNN': 0,
                    'HRV_SDNN': 0,
                    'HRV_RMSSD': 0,
                    'HRV_LF': 0,
                    'HRV_HF': 0,
                    'HRV_ratio_LF_HF': 0,
                    'Min_Heart-Rate': 0,
                    'Mean_Heart-Rate': 0,
                    'Max_Heart-Rate': 0
                }
                all_hr_samples.append(sample_features)
            else:
                # More than 3 peaks found! Use real values here!
                hrv_time = nk.hrv_time(info, sampling_rate=64, show=False)
                hrv_time = hrv_time[['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD']]
                hrv_freq = nk.hrv_frequency(info, sampling_rate=64, show= False, normalize=True)
                hrv_freq = hrv_freq[['HRV_LF', 'HRV_HF']]
                hrv_freq["HRV_ratio_LF_HF"] = hrv_freq["HRV_LF"].div(hrv_freq["HRV_HF"].values)
                hrv_features=pd.concat([hrv_time , hrv_freq], axis=1, join="inner")
                hrv_param.append(hrv_features)               
                hrv_all = pd.concat(hrv_param, axis=0, ignore_index= True)
                hrv_all.interpolate(method= "backfill", inplace=True)
                hrv_all.interpolate(method= "ffill", inplace=True)
                sample_features = {
                    'HRV_MeanNN': hrv_time['HRV_MeanNN'][0],
                    'HRV_SDNN': hrv_time['HRV_SDNN'][0],
                    'HRV_RMSSD': hrv_time['HRV_RMSSD'][0],
                    'HRV_LF': hrv_freq['HRV_LF'][0],
                    'HRV_HF': hrv_freq['HRV_HF'][0],
                    'HRV_ratio_LF_HF': hrv_freq['HRV_ratio_LF_HF'][0],
                    'Min_Heart-Rate': max([20, min(np.squeeze(sample['BVP']))]),     # Assuming a min of 20 heartbeats is nearly unrealistic
                    'Mean_Heart-Rate': np.mean(np.squeeze(sample['BVP'])),
                    'Max_Heart-Rate': min([230, max(np.squeeze(sample['BVP']))])    # Assuming a max of 230 heartbeats is nearly unrealistic
                }
                all_hr_samples.append(sample_features)
        theoretical_feature_num_counter += 9

    if (eeg_samples is not None) and (all_eda_samples is not None) and (all_hr_samples is not None) and (all_temp_samples is not None):
        combined_features_for_samples = []

        for sample_idx, eeg_sample_features in enumerate(all_eeg_samples_features_tuple):
            if sample_idx < min_len_samples:
                combined_features_for_samples.append({**eeg_sample_features, **all_temp_samples[sample_idx], **all_eda_samples[sample_idx], **all_hr_samples[sample_idx]})

        return combined_features_for_samples

    elif (eeg_samples is not None):
        # Only Muse was chosen
        return all_eeg_samples_features_tuple

    elif (eda_samples is not None) and (hr_samples is not None) and (temp_samples is not None):
        # Only E4 was chosen
        combined_features_for_samples = []

        for sample_idx, temp_sample_features in enumerate(all_temp_samples):
            if sample_idx < min_len_samples:
                combined_features_for_samples.append({**temp_sample_features, **all_eda_samples[sample_idx], **all_hr_samples[sample_idx]})

        return combined_features_for_samples


def get_data_all_participants_and_loo_participant():
    base_path = '../raw/dataset/'
    participant_data = []
    feature_labels_to_use = None

    for participant_number in range(1,11):        
        participant_path = 'Participant 0%d' % participant_number if participant_number != 10 else 'Participant 10'
        participant = 'Participant%d' % participant_number

        #####################    LAB DATA    ######################
        labs =  [item for item in os.listdir(base_path + participant_path) if os.path.isdir(os.path.join(base_path + participant_path, item)) and 'Lab' in item]
        labs_files = []

        for lab in labs:
            ##reading log files
            log_file = base_path + participant_path + "/" + lab + "/psychopy_log.log"
            print('LOG FILE: %s' % log_file)

            ###reading muse files
            if use_muse_data or use_both_data:
                file_pathx = base_path + participant_path + "/" + lab + ("/Lab%d_muse.csv" % int(lab.split(" ")[1]))
                df1_01 = pd.read_csv(file_pathx)

            if (use_e4_data or use_both_data) and pathlib.Path(base_path + participant_path + "/" + lab + "/EDA.csv").exists():
                file_path3 = base_path + participant_path + "/" + lab + "/EDA.csv"
                file_path4 = base_path + participant_path + "/" + lab + "/BVP.csv"
                file_path5 = base_path + participant_path + "/" + lab + "/TEMP.csv"

                eda = pd.read_csv(file_path3, header=None)
                bvp = pd.read_csv(file_path4, header=None)
                temp = pd.read_csv(file_path5, header=None)

                df1_bvp_01 = format_empatica_file(bvp, 'bvp ' + lab)
                df1_eda_01 = format_empatica_file(eda, 'eda ' + lab)
                df1_temp_01 = format_empatica_file(temp, 'temp ' + lab)
            
            if use_both_data and pathlib.Path(base_path + participant_path + "/" + lab + "/EDA.csv").exists():
                print('adding [df1_01, df1_bvp_01, df1_eda_01, df1_temp_01, log_file, lab] to labs_files')
                labs_files.append([df1_01, df1_bvp_01, df1_eda_01, df1_temp_01, log_file, lab])
            elif use_e4_data and pathlib.Path(base_path + participant_path + "/" + lab + "/EDA.csv").exists() and not use_both_data:
                print('adding [df1_bvp_01, df1_eda_01, df1_temp_01, log_file, lab] to labs_files')
                labs_files.append([df1_bvp_01, df1_eda_01, df1_temp_01, log_file, lab])
            elif use_muse_data:
                print('adding [df1_01, log_file, lab] to labs_files')
                labs_files.append([df1_01, log_file, lab])

        #-#   #####################    WILD DATA    ######################
        wilds =  [item for item in os.listdir(base_path + participant_path + "/In-the-wild/") if os.path.isdir(os.path.join(base_path + participant_path + "/In-the-wild/", item)) and item.isnumeric()]
        print('For this participant, these are the folders with wild files to be considered: %s' % str(wilds))
        wilds_files = []

        for wild in wilds:
            print('Reading for: %s' % wild)
            if use_muse_data or use_both_data:
                ###reading muse files
                file_pathx = base_path + participant_path + ("/In-the-wild/%s/P" % wild) + str(participant_number) + ("_wild%s_muse.csv" % wild)
                wild_muse_df1 = pd.read_csv(file_pathx)
                wild_muse_raw_df1 = wild_muse_df1[['TimeStamp','RAW_AF7','RAW_AF8','RAW_TP9','RAW_TP10','Accelerometer_X','Accelerometer_Y','Accelerometer_Z']]
                
                #replacing NaN values with neighbouring non-NaN values
                wild_muse_raw_df1 = wild_muse_raw_df1.ffill()
            
            if use_e4_data or use_both_data:
                wild_file_path2 = base_path + participant_path + ("/In-the-wild/%s/ACC.csv" % wild)
                wild_file_path3 = base_path + participant_path + ("/In-the-wild/%s/EDA.csv" % wild)
                wild_file_path4 = base_path + participant_path + ("/In-the-wild/%s/BVP.csv" % wild)
                wild_file_path5 = base_path + participant_path + ("/In-the-wild/%s/TEMP.csv" % wild)
                
                wild_acc = pd.read_csv(wild_file_path2, header=None)
                wild_eda = pd.read_csv(wild_file_path3, header=None)
                wild_bvp = pd.read_csv(wild_file_path4, header=None)
                wild_temp = pd.read_csv(wild_file_path5, header=None)
                
                wild_df1_acc = format_empatica_file(wild_acc, 'acc wild %s' % wild)
                wild_df1_bvp = format_empatica_file(wild_bvp, 'bvp wild %s' % wild)
                wild_df1_eda = format_empatica_file(wild_eda, 'eda wild %s' % wild)
                wild_df1_temp = format_empatica_file(wild_temp, 'temp wild %s' % wild)

                wild_df1_bvp.rename(columns={wild_df1_bvp.columns[1]: 'BVP'}, inplace=True)
                wild_df1_acc.rename(columns={wild_df1_acc.columns[1]: 'ACC_x'}, inplace=True)
                wild_df1_acc.rename(columns={wild_df1_acc.columns[2]: 'ACC_y'}, inplace=True)
                wild_df1_acc.rename(columns={wild_df1_acc.columns[3]: 'ACC_z'}, inplace=True)
                wild_df1_eda.rename(columns={wild_df1_eda.columns[1]: 'EDA'}, inplace=True)
                wild_df1_temp.rename(columns={wild_df1_temp.columns[1]: 'TEMP'}, inplace=True)
            
            if use_both_data:
                wilds_files.append([wild_muse_raw_df1, wild_df1_acc, wild_df1_bvp, wild_df1_eda, wild_df1_temp])
            elif use_e4_data:
                wilds_files.append([wild_df1_acc, wild_df1_bvp, wild_df1_eda, wild_df1_temp])
            elif use_muse_data:
                wilds_files.append([wild_muse_raw_df1])

        #-#   ### ### ### ### ### FEATURE EXTRACTIONS ### ### ### ### ###
        #-#   ### LAB DATA HERE 
        if use_both_data:
            all_data_present = [False, False]
            only_e4_data_present = [False, False]
            only_muse_data_present = [False, False]
            for lab_idx, files in enumerate(labs_files):
                if len(files) == 6:
                    (df1_01, df1_bvp_01, df1_eda_01, df1_temp_01, log_file, lab) = files
                    df1_01 = df1_01.bfill()
                    df1_bvp_01 = df1_bvp_01.bfill()
                    df1_bvp_01.rename(columns={df1_bvp_01.columns[1]: 'BVP'}, inplace=True)
                    df1_eda_01 = df1_eda_01.bfill()
                    df1_temp_01 = df1_temp_01.bfill()
                    
                    raw_df1 = df1_01[['TimeStamp','RAW_AF7','RAW_AF8','RAW_TP9','RAW_TP10','Accelerometer_X','Accelerometer_Y','Accelerometer_Z']]
                    raw_df1 = raw_df1.assign(Timestamp=pd.to_datetime(raw_df1['TimeStamp']))
                    
                    start_time_lab_1 = max(df1_bvp_01['Timestamp'].iloc[0], df1_eda_01['Timestamp'].iloc[0], df1_temp_01['Timestamp'].iloc[0], raw_df1['Timestamp'].iloc[0])
                    end_time_lab_1 = min(df1_bvp_01['Timestamp'].iloc[-1], df1_eda_01['Timestamp'].iloc[-1], df1_temp_01['Timestamp'].iloc[-1], raw_df1['Timestamp'].iloc[-1])

                    from psychopy_csv_log_parser import parse_psychopy_log_file
                    lab_log_dictionary = parse_psychopy_log_file(log_file)
                    
                    print('For this dataset\'s lab data [%s], these are the times: [%s - %s]' % (labs[lab_idx], start_time_lab_1, end_time_lab_1))
                    labs_files[lab_idx] = [raw_df1, df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary]
                    all_data_present[lab_idx] = True

                elif len(files) == 5:
                    (df1_bvp_01, df1_eda_01, df1_temp_01, log_file, lab) = files
                    df1_01 = df1_01.bfill()
                    df1_bvp_01 = df1_bvp_01.bfill()
                    df1_bvp_01.rename(columns={df1_bvp_01.columns[1]: 'BVP'}, inplace=True)
                    df1_eda_01 = df1_eda_01.bfill()
                    df1_temp_01 = df1_temp_01.bfill()
                    start_time_lab_1 = max(df1_bvp_01['Timestamp'].iloc[0], df1_eda_01['Timestamp'].iloc[0], df1_temp_01['Timestamp'].iloc[0])
                    end_time_lab_1 = min(df1_bvp_01['Timestamp'].iloc[-1], df1_eda_01['Timestamp'].iloc[-1], df1_temp_01['Timestamp'].iloc[-1])

                    from psychopy_csv_log_parser import parse_psychopy_log_file
                    lab_log_dictionary = parse_psychopy_log_file(log_file)

                    print('For this dataset\'s lab data [%s], these are the times: [%s - %s]' % (labs[lab_idx], start_time_lab_1, end_time_lab_1))
                    labs_files[lab_idx] = [df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary]
                    only_e4_data_present[lab_idx] = True

                elif len(files) == 3:
                    (df1_01, log_file, lab) = files
                    raw_df1 = df1_01[['TimeStamp','RAW_AF7','RAW_AF8','RAW_TP9','RAW_TP10','Accelerometer_X','Accelerometer_Y','Accelerometer_Z']]
                    raw_df1 = raw_df1.assign(Timestamp=pd.to_datetime(raw_df1['TimeStamp']))
                    start_time_lab_1 = raw_df1['Timestamp'].iloc[0]
                    end_time_lab_1 = raw_df1['Timestamp'].iloc[-1]

                    from psychopy_csv_log_parser import parse_psychopy_log_file
                    lab_log_dictionary = parse_psychopy_log_file(log_file)

                    print('For this dataset\'s lab data [%s], these are the times: [%s - %s]' % (labs[lab_idx], start_time_lab_1, end_time_lab_1))
                    labs_files[lab_idx] = [raw_df1, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary]
                    only_muse_data_present[lab_idx] = True

        elif use_e4_data:
            for lab_idx, (df1_bvp_01, df1_eda_01, df1_temp_01, log_file, lab) in enumerate(labs_files):
                df1_bvp_01 = df1_bvp_01.bfill()
                df1_bvp_01.rename(columns={df1_bvp_01.columns[1]: 'BVP'}, inplace=True)
                df1_eda_01 = df1_eda_01.bfill()
                df1_temp_01 = df1_temp_01.bfill()
                
                from psychopy_csv_log_parser import parse_psychopy_log_file
                lab_log_dictionary = parse_psychopy_log_file(log_file)
                
                start_time_lab_1 = max(df1_bvp_01['Timestamp'].iloc[0], df1_eda_01['Timestamp'].iloc[0], df1_temp_01['Timestamp'].iloc[0])
                end_time_lab_1 = min(df1_bvp_01['Timestamp'].iloc[-1], df1_eda_01['Timestamp'].iloc[-1], df1_temp_01['Timestamp'].iloc[-1])
                
                print('For this dataset\'s lab data [%s], these are the times: [%s - %s]' % (labs[lab_idx], start_time_lab_1, end_time_lab_1))
                labs_files[lab_idx] = [df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary]

        elif use_muse_data:
            for lab_idx, (df1_01, log_file, lab) in enumerate(labs_files):
                df1_01 = df1_01.bfill()
                raw_df1 = df1_01[['TimeStamp','RAW_AF7','RAW_AF8','RAW_TP9','RAW_TP10','Accelerometer_X','Accelerometer_Y','Accelerometer_Z']]
                raw_df1 = raw_df1.assign(Timestamp=pd.to_datetime(raw_df1['TimeStamp']))
                
                start_time_lab_1 = raw_df1['Timestamp'].iloc[0]
                end_time_lab_1 = raw_df1['Timestamp'].iloc[-1]

                from psychopy_csv_log_parser import parse_psychopy_log_file
                lab_log_dictionary = parse_psychopy_log_file(log_file)
                
                print('For this dataset\'s lab data [%s], these are the times: [%s - %s]' % (labs[lab_idx], start_time_lab_1, end_time_lab_1))
                labs_files[lab_idx] = [raw_df1, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary]

        ### NOW DATA CONCATENATING OF LAB DATA
        if use_both_data:
            if (all_data_present[0] and all_data_present[1]) or (only_e4_data_present[0] and only_e4_data_present[1]) or (all_data_present[0] and only_e4_data_present[1]) or (all_data_present[1] and only_e4_data_present[0]):
                df_hr = pd.concat([bvp_to_hr(df1_bvp_01)[bvp_to_hr(df1_bvp_01)['Timestamp'].between(start_time_lab_1, end_time_lab_1)] 
                    for (_, df1_bvp_01, _, _, start_time_lab_1, end_time_lab_1, _, _, _) in labs_files], ignore_index=True)
                df_eda = pd.concat([df1_eda_01[df1_eda_01['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                    for (_, _, df1_eda_01, _, start_time_lab_1, end_time_lab_1, _, _, _) in labs_files], ignore_index=True)
                df_temp = pd.concat([df1_temp_01[df1_temp_01['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                    for (_, _, _, df1_temp_01, start_time_lab_1, end_time_lab_1, _, _, _) in labs_files], ignore_index=True)
                df_hr = df_hr.bfill()
                df_eda = df_eda.bfill()
                df_temp = df_temp.bfill()
                df_eda.rename(columns={df_eda.columns[1]: 'EDA'}, inplace=True)
                df_temp.rename(columns={df_temp.columns[1]: 'TEMP'}, inplace=True)
                df_hr = df_hr.set_index('Timestamp')
                df_eda = df_eda.set_index('Timestamp')
                df_temp = df_temp.set_index('Timestamp')

            elif (all_data_present[0]) or (only_e4_data_present[0]):
                (_, df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, _, _, _) = labs_files[0]
                df_hr = bvp_to_hr(df1_bvp_01)[bvp_to_hr(df1_bvp_01)['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                df_eda = df1_eda_01[df1_eda_01['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                df_temp = df1_temp_01[df1_temp_01['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                df_hr = df_hr.bfill()
                df_eda = df_eda.bfill()
                df_temp = df_temp.bfill()
                df_eda.rename(columns={df_eda.columns[1]: 'EDA'}, inplace=True)
                df_temp.rename(columns={df_temp.columns[1]: 'TEMP'}, inplace=True)
                df_hr = df_hr.set_index('Timestamp')
                df_eda = df_eda.set_index('Timestamp')
                df_temp = df_temp.set_index('Timestamp')

            elif (all_data_present[1]) or (only_e4_data_present[1]):
                (_, df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, _, _, _) = labs_files[1]
                df_hr = bvp_to_hr(df1_bvp_01)[bvp_to_hr(df1_bvp_01)['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                df_eda = df1_eda_01[df1_eda_01['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                df_temp = df1_temp_01[df1_temp_01['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                df_hr = df_hr.bfill()
                df_eda = df_eda.bfill()
                df_temp = df_temp.bfill()
                df_eda.rename(columns={df_eda.columns[1]: 'EDA'}, inplace=True)
                df_temp.rename(columns={df_temp.columns[1]: 'TEMP'}, inplace=True)
                df_hr = df_hr.set_index('Timestamp')
                df_eda = df_eda.set_index('Timestamp')
                df_temp = df_temp.set_index('Timestamp')

            if (all_data_present[0] and all_data_present[1]) or (only_muse_data_present[0] and only_muse_data_present[1]) or (all_data_present[0] and only_muse_data_present[1]) or (all_data_present[1] and only_muse_data_present[0]):
                if len(labs_files[0]) == 9:
                    (raw_df1, _, _, _, start_time_lab_1, end_time_lab_1, _, _, _) = labs_files[0]
                elif len(labs_files[0]) == 6:
                    (raw_df1, start_time_lab_1, end_time_lab_1, _, _, _) = labs_files[0]
                else: 
                    print('Some issue on the length counter! Check this!')
                    exit(-1)
                if len(labs_files[1]) == 9:
                    (raw_df2, _, _, _, start_time_lab_2, end_time_lab_2, _, _, _) = labs_files[1]
                elif len(labs_files[1]) == 6:
                    (raw_df2, start_time_lab_2, end_time_lab_2, _, _, _) = labs_files[1]
                else: 
                    print('Some issue on the length counter! Check this (Lab File 2)!')
                    exit(-1)
                df_raw_eeg = pd.concat([raw_df1[raw_df1['Timestamp'].between(start_time_lab_1, end_time_lab_1)], raw_df2[raw_df2['Timestamp'].between(start_time_lab_2, end_time_lab_2)]], ignore_index=True)
                df_raw_eeg = df_raw_eeg.bfill()
                df_raw_eeg = df_raw_eeg.set_index('Timestamp')

            elif (all_data_present[0]) or (only_muse_data_present[0]):
                (raw_df1, _, _, _, start_time_lab_1, end_time_lab_1, _, _, _) = labs_files[0]
                df_raw_eeg = raw_df1[raw_df1['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                df_raw_eeg = df_raw_eeg.bfill()
                df_raw_eeg = df_raw_eeg.set_index('Timestamp')

            elif (all_data_present[1]) or (only_muse_data_present[1]):
                (raw_df1, _, _, _, start_time_lab_1, end_time_lab_1, _, _, _) = labs_files[1]
                df_raw_eeg = raw_df1[raw_df1['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                df_raw_eeg = df_raw_eeg.bfill()
                df_raw_eeg = df_raw_eeg.set_index('Timestamp')

        elif use_e4_data:
            df_hr = pd.concat([bvp_to_hr(df1_bvp_01)[bvp_to_hr(df1_bvp_01)['Timestamp'].between(start_time_lab_1, end_time_lab_1)] 
                for (df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary) in labs_files], ignore_index=True)
            df_eda = pd.concat([df1_eda_01[df1_eda_01['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                for (df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary) in labs_files], ignore_index=True)
            df_temp = pd.concat([df1_temp_01[df1_temp_01['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                for (df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary) in labs_files], ignore_index=True)
            df_hr = df_hr.bfill()
            df_eda = df_eda.bfill()
            df_temp = df_temp.bfill()
            df_eda.rename(columns={df_eda.columns[1]: 'EDA'}, inplace=True)
            df_temp.rename(columns={df_temp.columns[1]: 'TEMP'}, inplace=True)
            df_hr = df_hr.set_index('Timestamp')
            df_eda = df_eda.set_index('Timestamp')
            df_temp = df_temp.set_index('Timestamp')

        elif use_muse_data:
            df_raw_eeg = pd.concat([raw_df1[raw_df1['Timestamp'].between(start_time_lab_1, end_time_lab_1)]
                for (raw_df1, start_time_lab_1, end_time_lab_1, _, _, _) in labs_files], ignore_index=True)
            df_raw_eeg = df_raw_eeg.bfill()
            df_raw_eeg = df_raw_eeg.set_index('Timestamp')

        #-#   ### WILD DATA HERE 
        if use_both_data:
            for wild_idx, (wild_muse_raw_df1, wild_df1_acc, wild_df1_bvp, wild_df1_eda, wild_df1_temp) in enumerate(wilds_files):
                if isinstance(wild_muse_raw_df1, list):
                    wild_muse_raw_df1 = pd.DataFrame(wild_muse_raw_df1[0])
                wild_muse_df1 = wild_muse_raw_df1[['TimeStamp','RAW_AF7','RAW_AF8','RAW_TP9','RAW_TP10','Accelerometer_X','Accelerometer_Y','Accelerometer_Z']]
                wild_muse_df1 = wild_muse_df1.assign(Timestamp=pd.to_datetime(wild_muse_df1['TimeStamp']))
                wild_df1_bvp.rename(columns={wild_df1_bvp.columns[1]: 'BVP'}, inplace=True)
                start_time_wild_1 = max(wild_df1_bvp['Timestamp'].iloc[0], wild_df1_eda['Timestamp'].iloc[0], wild_df1_temp['Timestamp'].iloc[0], wild_muse_df1['Timestamp'].iloc[0])
                end_time_wild_1 = min(wild_df1_bvp['Timestamp'].iloc[-1], wild_df1_eda['Timestamp'].iloc[-1], wild_df1_temp['Timestamp'].iloc[-1], wild_muse_df1['Timestamp'].iloc[-1])
                temp_start_time_wild_1 = start_time_wild_1
                temp_end_time_wild_1 = end_time_wild_1
                if start_time_wild_1 > end_time_wild_1:
                    start_time_wild_1 = end_time_wild_1
                    end_time_wild_1 = temp_start_time_wild_1
                print('For this dataset\'s wild data [Wild %s], these are the times: [%s - %s]' % (wilds[wild_idx], start_time_wild_1, end_time_wild_1))
                wilds_files[wild_idx] = [wild_muse_df1, wild_df1_acc, wild_df1_bvp, wild_df1_eda, wild_df1_temp, start_time_wild_1, end_time_wild_1]

        elif use_e4_data:
            for wild_idx, (wild_df1_acc, wild_df1_bvp, wild_df1_eda, wild_df1_temp) in enumerate(wilds_files):
                wild_df1_bvp.rename(columns={wild_df1_bvp.columns[1]: 'BVP'}, inplace=True)
                start_time_wild_1 = max(wild_df1_bvp['Timestamp'].iloc[0], wild_df1_eda['Timestamp'].iloc[0], wild_df1_temp['Timestamp'].iloc[0])
                end_time_wild_1 = min(wild_df1_bvp['Timestamp'].iloc[-1], wild_df1_eda['Timestamp'].iloc[-1], wild_df1_temp['Timestamp'].iloc[-1])
                print('For this dataset\'s wild data [Wild %s], these are the times: [%s - %s]' % (wilds[wild_idx], start_time_wild_1, end_time_wild_1))
                wilds_files[wild_idx] = [wild_df1_acc, wild_df1_bvp, wild_df1_eda, wild_df1_temp, start_time_wild_1, end_time_wild_1]

        elif use_muse_data:
            for wild_idx, wild_muse_raw_df1 in enumerate(wilds_files):
                if isinstance(wild_muse_raw_df1, list):
                    wild_muse_raw_df1 = pd.DataFrame(wild_muse_raw_df1[0])
                wild_muse_df1 = wild_muse_raw_df1[['TimeStamp','RAW_AF7','RAW_AF8','RAW_TP9','RAW_TP10','Accelerometer_X','Accelerometer_Y','Accelerometer_Z']]
                wild_muse_df1 = wild_muse_df1.assign(Timestamp=pd.to_datetime(wild_muse_df1['TimeStamp']))
                start_time_wild_1 = wild_muse_df1['Timestamp'].iloc[0]
                end_time_wild_1 = wild_muse_df1['Timestamp'].iloc[-1]
                print('For this dataset\'s wild data [Wild %s], these are the times: [%s - %s]' % (wilds[wild_idx], start_time_wild_1, end_time_wild_1))
                wilds_files[wild_idx] = [wild_muse_df1, start_time_wild_1, end_time_wild_1]

        #-#   ### NOW DATA CONCATENATING OF WILD DATA
        if use_e4_data or use_both_data:
            wild_df_hr = None
            wild_df_eda = None
            wild_df_temp = None
        if use_muse_data or use_both_data:
            wild_df_raw_eeg = None

        if len(wilds_files) >= 1:

            if use_both_data:
                for wild_file in wilds_files:
                    if (len(wild_file) == 7):
                        (wild_muse_df1, wild_df1_acc, wild_df1_bvp, wild_df1_eda, wild_df1_temp, start_time_wild_1, end_time_wild_1) = wild_file
                        if wild_df_hr is not None:
                            if len(wild_df_hr) == 1:
                                wild_df_hr = pd.concat([wild_df_hr[0], bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_hr) == 2:
                                wild_df_hr = pd.concat([wild_df_hr[1], bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_hr = bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]

                        if wild_df_eda is not None:
                            if len(wild_df_eda) == 1:
                                wild_df_eda = pd.concat([wild_df_eda[0], wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_eda) == 2:
                                wild_df_eda = pd.concat([wild_df_eda[1], wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_eda = wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]

                        if wild_df_temp is not None:
                            if len(wild_df_temp) == 1:
                                wild_df_temp = pd.concat([wild_df_temp[0], wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_temp) == 2:
                                wild_df_temp = pd.concat([wild_df_temp[1], wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_temp = wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]

                    elif (len(wild_file) == 6):
                        _, wild_df1_bvp, wild_df1_eda, wild_df1_temp, start_time_wild_1, end_time_wild_1 = wild_file
                        if wild_df_hr is not None:
                            if len(wild_df_hr) == 1:
                                wild_df_hr = pd.concat([wild_df_hr[0], bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_hr) == 2:
                                wild_df_hr = pd.concat([wild_df_hr[1], bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_hr = bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]
                        
                        if wild_df_eda is not None:
                            if len(wild_df_eda) == 1:
                                wild_df_eda = pd.concat([wild_df_eda[0], wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_eda) == 2:
                                wild_df_eda = pd.concat([wild_df_eda[1], wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_eda = wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]
                       
                        if wild_df_temp is not None:
                            if len(wild_df_temp) == 1:
                                wild_df_temp = pd.concat([wild_df_temp[0], wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_temp) == 2:
                                wild_df_temp = pd.concat([wild_df_temp[1], wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_temp = wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]

            elif use_e4_data:
                for wild_file in wilds_files:
                    if (len(wild_file) == 7):
                        (wild_muse_df1, wild_df1_acc, wild_df1_bvp, wild_df1_eda, wild_df1_temp, start_time_wild_1, end_time_wild_1) = wild_file
                        if wild_df_hr is not None:
                            if len(wild_df_hr) == 1:
                                wild_df_hr = pd.concat([wild_df_hr[0], bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_hr) == 2:
                                wild_df_hr = pd.concat([wild_df_hr[1], bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_hr = bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]
                        
                        if wild_df_eda is not None:
                            if len(wild_df_eda) == 1:
                                wild_df_eda = pd.concat([wild_df_eda[0], wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_eda) == 2:
                                wild_df_eda = pd.concat([wild_df_eda[1], wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_eda = wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]
                       
                        if wild_df_temp is not None:
                            if len(wild_df_temp) == 1:
                                wild_df_temp = pd.concat([wild_df_temp[0], wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_temp) == 2:
                                wild_df_temp = pd.concat([wild_df_temp[1], wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_temp = wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]
                    
                    elif (len(wild_file) == 6):
                        wild_df1_acc, wild_df1_bvp, wild_df1_eda, wild_df1_temp, start_time_wild_1, end_time_wild_1 = wild_file
                        if wild_df_hr is not None:
                            if len(wild_df_hr) == 1:
                                wild_df_hr = pd.concat([wild_df_hr[0], bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_hr) == 2:
                                wild_df_hr = pd.concat([wild_df_hr[1], bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_hr = bvp_to_hr(wild_df1_bvp)[bvp_to_hr(wild_df1_bvp)['Timestamp'].between(start_time_wild_1, end_time_wild_1)]
                        
                        if wild_df_eda is not None:
                            if len(wild_df_eda) == 1:
                                wild_df_eda = pd.concat([wild_df_eda[0], wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_eda) == 2:
                                wild_df_eda = pd.concat([wild_df_eda[1], wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_eda = wild_df1_eda[wild_df1_eda['Timestamp'].between(start_time_wild_1, end_time_wild_1)]
                        
                        if wild_df_temp is not None:
                            if len(wild_df_temp) == 1:
                                wild_df_temp = pd.concat([wild_df_temp[0], wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                            elif len(wild_df_temp) == 2:
                                wild_df_temp = pd.concat([wild_df_temp[1], wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]], ignore_index=True)
                        else:
                            wild_df_temp = wild_df1_temp[wild_df1_temp['Timestamp'].between(start_time_wild_1, end_time_wild_1)]

                wild_df_hr = wild_df_hr.bfill()
                wild_df_eda = wild_df_eda.bfill()
                wild_df_temp = wild_df_temp.bfill()
                wild_df_eda.rename(columns={wild_df_eda.columns[1]: 'EDA'}, inplace=True)
                wild_df_temp.rename(columns={wild_df_temp.columns[1]: 'TEMP'}, inplace=True)
                wild_df_hr = wild_df_hr.set_index('Timestamp')
                wild_df_eda = wild_df_eda.set_index('Timestamp')
                wild_df_temp = wild_df_temp.set_index('Timestamp')

            if use_both_data:
                wild_df_raw_eeg = pd.concat([
                    wild_muse_df1[wild_muse_df1['Timestamp'].between(start_time_wild_1, end_time_wild_1)]
                    for (wild_muse_df1, _, _, _, _, start_time_wild_1, end_time_wild_1) in wilds_files], ignore_index=True)
                wild_df_raw_eeg = wild_df_raw_eeg.bfill()
                wild_df_raw_eeg = wild_df_raw_eeg.set_index('Timestamp')

            elif use_muse_data:
                wild_df_raw_eeg = pd.concat([
                    wild_muse_df1[wild_muse_df1['Timestamp'].between(start_time_wild_1, end_time_wild_1)]
                    for (wild_muse_df1, start_time_wild_1, end_time_wild_1) in wilds_files], ignore_index=True)
                wild_df_raw_eeg = wild_df_raw_eeg.bfill()
                wild_df_raw_eeg = wild_df_raw_eeg.set_index('Timestamp')

        # Done concatenating all lab and wild data each into one df for lab, one for wild
        # Now take care of splitting lab and wild data dfs each into samples of unified length
        # Remain only valid samples! Therefore, calculate sfreq * seconds, and check that all samples have the same required length
        window_in_seconds = 60 # For windows in 30 seconds length, for participant 7 we run into issues! ValueError: zero-size array to reduction operation maximum which has no identity
        eeg_tolerance = 10

        if use_e4_data:
            eda_resampler = df_eda.resample('%ds' % window_in_seconds)
            eda_samples = [window for _, window in eda_resampler if ((not window.empty) and (len(window) == window_in_seconds * EDA_SFREQ))]
            bvp_resampler = df_hr.resample('%ds' % window_in_seconds)
            hr_samples = [window for _, window in bvp_resampler if ((not window.empty) and (len(window) == window_in_seconds * BVP_SFREQ))]
            temp_resampler = df_temp.resample('%ds' % window_in_seconds)
            temp_samples = [window for _, window in temp_resampler if ((not window.empty) and (len(window) == window_in_seconds * TEMP_SFREQ))]

        if use_muse_data:
            eeg_resampler = df_raw_eeg.resample('%ds' % window_in_seconds)
            eeg_samples = [window for _, window in eeg_resampler if ((not window.empty) and ((window_in_seconds - eeg_tolerance) * EEG_SFREQ <= len(window) <= (window_in_seconds + eeg_tolerance) * EEG_SFREQ))]

        if use_both_data:
            print('After resampling for windows of length %d seconds, these are the amounts of samples for the LAB data:\nEDA: %d, HR: %d, TEMP: %d, EEG: %d.\nNo worries if there are small differences in amount. We\'ll take proper care of it later.' 
                % (window_in_seconds, len(eda_samples), len(hr_samples), len(temp_samples), len(eeg_samples)))
        elif use_e4_data:
            print('After resampling for windows of length %d seconds, these are the amounts of samples for the LAB data:\nEDA: %d, HR: %d, TEMP: %d.\nNo worries if there are small differences in amount. We\'ll take proper care of it later.' 
                % (window_in_seconds, len(eda_samples), len(hr_samples), len(temp_samples)))
        elif use_muse_data:
            print('After resampling for windows of length %d seconds, these are the amounts of samples for the LAB data:\nEEG: %d.\nNo worries if there are small differences in amount. We\'ll take proper care of it later.' 
                % (window_in_seconds, len(eeg_samples)))

        if use_e4_data or use_both_data:
            wild_eda_samples = None
            wild_hr_samples = None
            wild_temp_samples = None
        if use_muse_data or use_both_data:
            wild_eeg_samples = None

        if len(wilds_files) >= 1:
            if use_e4_data or use_both_data:
                try:
                    wild_df_eda.set_index('Timestamp', inplace=True)
                    wild_df_hr.set_index('Timestamp', inplace=True)
                    wild_df_temp.set_index('Timestamp', inplace=True)
                except KeyError:
                    print('Probably the key Timestamp is already the index')

                wild_eda_resampler = wild_df_eda.resample('%ds' % window_in_seconds)
                wild_eda_samples = [window for _, window in wild_eda_resampler if ((not window.empty) and (len(window) == window_in_seconds * EDA_SFREQ))]
                wild_bvp_resampler = wild_df_hr.resample('%ds' % window_in_seconds)
                wild_hr_samples = [window for _, window in wild_bvp_resampler if ((not window.empty) and (len(window) == window_in_seconds * BVP_SFREQ))]
                wild_temp_resampler = wild_df_temp.resample('%ds' % window_in_seconds)
                wild_temp_samples = [window for _, window in wild_temp_resampler if ((not window.empty) and (len(window) == window_in_seconds * TEMP_SFREQ))]
            if use_muse_data or use_both_data:
                wild_eeg_resampler = wild_df_raw_eeg.resample('%ds' % window_in_seconds)
                wild_eeg_samples = [window for _, window in wild_eeg_resampler if ((not window.empty) and ((window_in_seconds - eeg_tolerance) * EEG_SFREQ <= len(window) <= (window_in_seconds + eeg_tolerance) * EEG_SFREQ))]

            if use_both_data:
                print('After resampling for windows of length %d seconds, these are the amounts of samples for the WILD data:\nEDA: %d, HR: %d, TEMP: %d, EEG: %d.\nNo worries if there are small differences in amount. We\'ll take proper care of it later.' 
                    % (window_in_seconds, len(wild_eda_samples), len(wild_hr_samples), len(wild_temp_samples), len(wild_eeg_samples)))
            elif use_e4_data:
                print('After resampling for windows of length %d seconds, these are the amounts of samples for the WILD data:\nEDA: %d, HR: %d, TEMP: %d.\nNo worries if there are small differences in amount. We\'ll take proper care of it later.' 
                    % (window_in_seconds, len(wild_eda_samples), len(wild_hr_samples), len(wild_temp_samples)))
            elif use_muse_data:
                print('After resampling for windows of length %d seconds, these are the amounts of samples for the WILD data:\nEEG: %d.\nNo worries if there are small differences in amount. We\'ll take proper care of it later.' 
                    % (window_in_seconds, len(wild_eeg_samples)))

        # Done splitting lab and wild data dfs each into samples of unified length
        # Now calculate features for the valid samples using earlier defined methods to extract DWT features

        print('STARTING THE HAND-CRAFTED FEATURE EXTRACTION FOR LAB!')

        if use_e4_data and (eda_samples is not None) and use_muse_data and (eeg_samples is not None):
            min_len_samples = min(len(eda_samples), len(hr_samples), len(temp_samples), len(eeg_samples))
        elif use_muse_data and eeg_samples is not None:
            min_len_samples = len(eeg_samples)
        elif use_e4_data and eda_samples is not None:
            min_len_samples = min(len(eda_samples), len(hr_samples), len(temp_samples))

        if use_both_data:
            handcrafted_features = handcrafted_features_extraction(participant_number=participant_number, eeg_samples=eeg_samples, eda_samples=eda_samples, hr_samples=hr_samples, temp_samples=temp_samples, min_len_samples=min_len_samples)
        elif use_e4_data:
            handcrafted_features = handcrafted_features_extraction(participant_number=participant_number, eeg_samples=None, eda_samples=eda_samples, hr_samples=hr_samples, temp_samples=temp_samples, min_len_samples=min_len_samples)
        elif use_muse_data:
            handcrafted_features = handcrafted_features_extraction(participant_number=participant_number, eeg_samples=eeg_samples, eda_samples=None, hr_samples=None, temp_samples=None, min_len_samples=min_len_samples)

        if len(wilds_files) >= 1:
            if use_e4_data and (wild_eda_samples is not None) and use_muse_data and (wild_eeg_samples is not None):
                wild_min_len_samples = min(len(wild_eda_samples), len(wild_hr_samples), len(wild_temp_samples), len(wild_eeg_samples))
            elif use_muse_data and wild_eeg_samples is not None:
                wild_min_len_samples = len(wild_eeg_samples)
            elif use_e4_data and wild_eda_samples is not None:
                wild_min_len_samples = min(len(wild_eda_samples), len(wild_hr_samples), len(wild_temp_samples))

            if use_both_data:
                wild_handcrafted_features = handcrafted_features_extraction(participant_number=participant_number, eeg_samples=wild_eeg_samples, eda_samples=wild_eda_samples, hr_samples=wild_hr_samples, temp_samples=wild_temp_samples, min_len_samples=wild_min_len_samples)
            elif use_e4_data:
                wild_handcrafted_features = handcrafted_features_extraction(participant_number=participant_number, eeg_samples=None, eda_samples=wild_eda_samples, hr_samples=wild_hr_samples, temp_samples=wild_temp_samples, min_len_samples=wild_min_len_samples)
            elif use_muse_data:
                wild_handcrafted_features = handcrafted_features_extraction(participant_number=participant_number, eeg_samples=wild_eeg_samples, eda_samples=None, hr_samples=None, temp_samples=None, min_len_samples=wild_min_len_samples)

        # Next, take care of assigning correct labels to the data
        if use_both_data:
            lab_tasks_array = []
            lab_logs_files_dicts = []
            for lab_file in labs_files:
                if len(lab_file) == 9: # [raw_df1, df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary]
                    (_, _, _, _, _, _, log_file, lab, log_file_dict) = lab_file
                    tmp_routine_timestamps = extract_routine_timestamps(log_file, lab)
                    lab_tasks_array.append(tmp_routine_timestamps)
                    lab_logs_files_dicts.append(log_file_dict)
                elif len(lab_file) == 8:
                    # Len 8: [df1_bvp_01, df1_eda_01, df1_temp_01, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary]
                    (_, _, _, _, _, log_file, lab, log_file_dict) = lab_file
                    tmp_routine_timestamps = extract_routine_timestamps(log_file, lab)
                    lab_tasks_array.append(tmp_routine_timestamps)
                    lab_logs_files_dicts.append(log_file_dict)
                elif len(lab_file) == 6:   # [raw_df1, start_time_lab_1, end_time_lab_1, log_file, lab, lab_log_dictionary] 
                    (_, _, _, log_file, lab, log_file_dict) = lab_file
                    tmp_routine_timestamps = extract_routine_timestamps(log_file, lab)
                    lab_tasks_array.append(tmp_routine_timestamps)
                    lab_logs_files_dicts.append(log_file_dict)
                else: 
                    print('Some issue on the length counter for labs! Check this!')
                    exit(-1)

        elif use_e4_data:
            lab_tasks_array = [extract_routine_timestamps(log_file, lab) for (_, _, _, _, _, log_file, lab, _) in labs_files]
            lab_logs_files_dicts = [log_file_dict for (_, _, _, _, _, _, _, log_file_dict) in labs_files]

        elif use_muse_data:
            lab_tasks_array = [extract_routine_timestamps(log_file, lab) for (_, _, _, log_file, lab, _) in labs_files]
            lab_logs_files_dicts = [log_file_dict for (_, _, _, _, _, log_file_dict) in labs_files]

        lab_tasks = lab_tasks_array[0] if len(lab_tasks_array) == 1 else {**lab_tasks_array[0], **lab_tasks_array[1]}

        if len(wilds_files) >= 1:
            wild_tasks = participants_wild_tasks_dir[participant_number - 1]    # -1 as participants range 1 to 10, but stored in array 0 to 9

        feature_labels_to_use = handcrafted_features[0].keys()
        # print('SANITY CHECK - POSSIBLE FEATURES TO USE BEFORE FEATURE SELECTION:')
        # print(feature_labels_to_use)
        aligned_lab_features_comb = handcrafted_features.copy()
        lab_timestamps_and_features_and_labels = combine_timestamps_and_features_and_labels(aligned_lab_features_comb, lab_tasks, True, lab_logs_files_dicts)

        aligned_wild_features_comb = None
        wild_timestamps_and_features_and_labels = None
        if len(wilds_files) >= 1:
            aligned_wild_features_comb = wild_handcrafted_features.copy()
            wild_timestamps_and_features_and_labels = combine_timestamps_and_features_and_labels(aligned_wild_features_comb, wild_tasks, False)

        wild_class_counter = {'vlw_mw': 0, 'lw_mw': 0, 'nor_mw': 0, 'hg_mw': 0, 'vhg_mw': 0}
        lab_class_counter = {'vlw_mw': 0, 'lw_mw': 0, 'nor_mw': 0, 'hg_mw': 0, 'vhg_mw': 0}
        vhg_mw = []
        hg_mw = []
        nor_mw = []
        lw_mw = []
        vlw_mw = []   

        if len(wilds_files) >= 1:
            for key, value in wild_timestamps_and_features_and_labels.items():
                label = value[0]
                if label == 1:
                    tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                    tmp_data_arr_to_be_np_arr = []
                    for dat_to_be_np in tmp_data_arr:
                        if dat_to_be_np is not None:
                            tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                    vlw_mw.append(tmp_data_arr_to_be_np_arr)
                    wild_class_counter['vlw_mw'] += 1

                elif label == 2:
                    tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                    tmp_data_arr_to_be_np_arr = []
                    for dat_to_be_np in tmp_data_arr:
                        if dat_to_be_np is not None:
                            tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                    lw_mw.append(tmp_data_arr_to_be_np_arr)
                    wild_class_counter['lw_mw'] += 1

                elif label == 3:
                    tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                    tmp_data_arr_to_be_np_arr = []
                    for dat_to_be_np in tmp_data_arr:
                        if dat_to_be_np is not None:
                            tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                    nor_mw.append(tmp_data_arr_to_be_np_arr)
                    wild_class_counter['nor_mw'] += 1

                elif label == 4:
                    tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                    tmp_data_arr_to_be_np_arr = []
                    for dat_to_be_np in tmp_data_arr:
                        if dat_to_be_np is not None:
                            tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                    hg_mw.append(tmp_data_arr_to_be_np_arr)
                    wild_class_counter['hg_mw'] += 1

                elif label == 5:
                    tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                    tmp_data_arr_to_be_np_arr = []
                    for dat_to_be_np in tmp_data_arr:
                        if dat_to_be_np is not None:
                            tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                    vhg_mw.append(tmp_data_arr_to_be_np_arr)
                    wild_class_counter['vhg_mw'] += 1

        for key, value in lab_timestamps_and_features_and_labels.items():
            label = value[0]
            if isinstance(label, list):
                label = label[0]
            lab_class_counter[label] += 1
            if label == 'vlw_mw':
                tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                tmp_data_arr_to_be_np_arr = []
                for dat_to_be_np in tmp_data_arr:
                    if dat_to_be_np is not None:
                        tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                vlw_mw.append(tmp_data_arr_to_be_np_arr)

            elif label == 'lw_mw':
                tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                tmp_data_arr_to_be_np_arr = []
                for dat_to_be_np in tmp_data_arr:
                    if dat_to_be_np is not None:
                        tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                lw_mw.append(tmp_data_arr_to_be_np_arr)

            elif label == 'nor_mw':
                tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                tmp_data_arr_to_be_np_arr = []
                for dat_to_be_np in tmp_data_arr:
                    if dat_to_be_np is not None:
                        tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                nor_mw.append(tmp_data_arr_to_be_np_arr)

            elif label == 'hg_mw':
                tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                tmp_data_arr_to_be_np_arr = []
                for dat_to_be_np in tmp_data_arr:
                    if dat_to_be_np is not None:
                        tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                hg_mw.append(tmp_data_arr_to_be_np_arr)

            elif label == 'vhg_mw':
                tmp_data_arr = [np.float32(item) if (type(item) == np.float64 or type(item) == np.int64 or type(item) == float or type(item) == int) else None for key, item in value[1].items()]
                tmp_data_arr_to_be_np_arr = []
                for dat_to_be_np in tmp_data_arr:
                    if dat_to_be_np is not None:
                        tmp_data_arr_to_be_np_arr.append(dat_to_be_np)
                vhg_mw.append(tmp_data_arr_to_be_np_arr)

        print('For Lab, this is the class distribution: %s' % lab_class_counter)
        print('For Wild, this is the class distribution: %s' % wild_class_counter)
        print('Overall, for Lab and Wild combined, this is the class distribution: %s' % ({k : lab_class_counter[k] + wild_class_counter[k] for k in lab_class_counter}))

        if three_class_classification:
            print('### ### ### ### ### LABEL SHIFTING TO THREE-CLASS PROBLEM ### ### ### ### ###')
            print('This shall be a three-class classification problem, so the classes will be modified in the following way:')

            vhg_mw_ctr = 0
            hg_mw_ctr = 0
            nor_mw_ctr = 0
            lw_mw_ctr = 0
            vlw_mw_ctr = 0
            for diction in [lab_class_counter, wild_class_counter]:
                for lbl, num_vals in diction.items():
                    if lbl == 'vhg_mw':
                        vhg_mw_ctr += num_vals
                    elif lbl == 'hg_mw':
                        hg_mw_ctr += num_vals
                    elif lbl == 'nor_mw':
                        nor_mw_ctr += num_vals
                    elif lbl == 'lw_mw':
                        lw_mw_ctr += num_vals
                    elif lbl == 'vlw_mw':
                        vlw_mw_ctr += num_vals

            num_total_labels = sum([vhg_mw_ctr, hg_mw_ctr, nor_mw_ctr, lw_mw_ctr, vlw_mw_ctr])
            approximate_third = 0.33 * num_total_labels

            num_zero_lbls_ctr = 0

            for lbl_ctr in [vhg_mw_ctr, hg_mw_ctr, nor_mw_ctr, lw_mw_ctr, vlw_mw_ctr]:
                if lbl_ctr == 0:
                    num_zero_lbls_ctr += 1

            if (num_zero_lbls_ctr == 0) or (num_zero_lbls_ctr == 1):
                shift_num = 0.33*sum([vhg_mw_ctr, hg_mw_ctr, nor_mw_ctr, lw_mw_ctr, vlw_mw_ctr])    # 0.33 as we want to reach roughly balanced class distribution!
                ctr_np = np.asarray([vhg_mw_ctr, hg_mw_ctr, nor_mw_ctr, lw_mw_ctr, vlw_mw_ctr])
                ctr_df = pd.DataFrame([vhg_mw_ctr, hg_mw_ctr, nor_mw_ctr, lw_mw_ctr, vlw_mw_ctr])
                previous_labels_data = [(vhg_mw, 'vhg_mw'), (hg_mw, 'hg_mw'), (nor_mw, 'nor_mw'), (lw_mw, 'lw_mw'), (vlw_mw, 'vlw_mw')]
                new_hg_mw = [[], []]
                new_nor_mw = [[], []]
                new_lw_mw = [[], []]            
                shifted_multiplier = 1
                split_1 = 0
                split_2 = 0
                for idx, amount in ctr_df.cumsum().iterrows():
                    print(amount[0])
                    if amount[0] >= (shift_num * shifted_multiplier):
                        print('Will need to split in the area of %d' % idx)
                        if shifted_multiplier == 1:
                            split_1 = idx
                        elif shifted_multiplier == 2:
                            split_2 = idx
                        else:
                            print('WARNING: SOMEHOW SOMETHING MIGHT HAVE GONE WRONG WITH THE DATA RE-LABELLING! IGNORING FOR NOW AND HAVING TO CHECK IN THE RESULTS LATER!')
                        shifted_multiplier += 1

                option_a_std = np.std([sum([i for i in ctr_np[:split_1]]), sum([i for i in ctr_np[split_1:split_2]]), sum([i for i in ctr_np[split_2:]])])
                option_b_std = np.std([sum([i for i in ctr_np[:split_1+1]]), sum([i for i in ctr_np[split_1+1:split_2]]), sum([i for i in ctr_np[split_2:]])])
                option_c_std = np.std([sum([i for i in ctr_np[:split_1+1]]), sum([i for i in ctr_np[split_1+1:split_2+1]]), sum([i for i in ctr_np[split_2+1:]])])
                option_d_std = np.std([sum([i for i in ctr_np[:split_1]]), sum([i for i in ctr_np[split_1:split_2+1]]), sum([i for i in ctr_np[split_2+1:]])])

                smallest_std = None
                smallest_option_id = None

                if (option_a_std <= option_b_std) and (option_a_std <= option_c_std) and (option_a_std <= option_d_std):
                    smallest_std = option_a_std
                    smallest_option_id = 'A'
                    actually_to_use_split_1 = split_1
                    actually_to_use_split_2 = split_2
                elif (option_b_std <= option_a_std) and (option_b_std <= option_c_std) and (option_b_std <= option_d_std):
                    smallest_std = option_b_std
                    smallest_option_id = 'B'
                    actually_to_use_split_1 = split_1+1
                    actually_to_use_split_2 = split_2
                elif (option_c_std <= option_a_std) and (option_c_std <= option_b_std) and (option_c_std <= option_d_std):
                    smallest_std = option_c_std
                    smallest_option_id = 'C'
                    actually_to_use_split_1 = split_1+1
                    actually_to_use_split_2 = split_2+1
                else:
                    smallest_std = option_d_std
                    smallest_option_id = 'D'
                    actually_to_use_split_1 = split_1
                    actually_to_use_split_2 = split_2+1

                print('For the four different options, these are the STDs: [%s], recommending option %s!' % (('A: %f, B: %f, C: %f, D: %f' % (option_a_std, option_b_std, option_c_std, option_d_std)), smallest_option_id))

                for i in range(0, actually_to_use_split_1):
                    new_hg_mw[0].append(previous_labels_data[i][0])
                    new_hg_mw[1].append(previous_labels_data[i][1])
                for i in range(actually_to_use_split_1, actually_to_use_split_2):
                    new_nor_mw[0].append(previous_labels_data[i][0])
                    new_nor_mw[1].append(previous_labels_data[i][1])
                for i in range(actually_to_use_split_2, len(previous_labels_data)):
                    new_lw_mw[0].append(previous_labels_data[i][0])
                    new_lw_mw[1].append(previous_labels_data[i][1])

                if len(new_hg_mw[0]) >= 2:
                    new_hg_mw[0] = [dat_to_be_np for sublist in new_hg_mw[0] for dat_to_be_np in sublist]
                    was_string = ''
                    for was_identifier in new_hg_mw[1]:
                        was_string += ('-and-' + str(was_identifier))
                    new_hg_mw[1] = was_string
                else:
                    new_hg_mw[0] = [dat_to_be_np for sublist in new_hg_mw[0] for dat_to_be_np in sublist]
                    new_hg_mw[1] = new_hg_mw[1][0]

                if len(new_nor_mw[0]) >= 2:
                    new_nor_mw[0] = [dat_to_be_np for sublist in new_nor_mw[0] for dat_to_be_np in sublist]
                    was_string = ''
                    for was_identifier in new_nor_mw[1]:
                        was_string += ('-and-' + str(was_identifier))
                    new_nor_mw[1] = was_string
                else:
                    new_nor_mw[0] = [dat_to_be_np for sublist in new_nor_mw[0] for dat_to_be_np in sublist]
                    new_nor_mw[1] = new_nor_mw[1][0]

                if len(new_lw_mw[0]) >= 2:
                    new_lw_mw[0] = [dat_to_be_np for sublist in new_lw_mw[0] for dat_to_be_np in sublist]
                    was_string = ''
                    for was_identifier in new_lw_mw[1]:
                        was_string += ('-and-' + str(was_identifier))
                    new_lw_mw[1] = was_string
                else:
                    new_lw_mw[0] = [dat_to_be_np for sublist in new_lw_mw[0] for dat_to_be_np in sublist]
                    new_lw_mw[1] = new_lw_mw[1][0]

            elif num_zero_lbls_ctr == 2:
                new_hg_mw = [None, None]
                new_nor_mw = [None, None]
                new_lw_mw = [None, None]
                amount_defined = 0
                for (lbl_ctr, new_lbls, was_formerly) in [(vhg_mw_ctr, vhg_mw, 'vhg_mw'), (hg_mw_ctr, hg_mw, 'hg_mw'), (nor_mw_ctr, nor_mw, 'nor_mw'), (lw_mw_ctr, lw_mw, 'lw_mw'), (vlw_mw_ctr, vlw_mw, 'vlw_mw')]:
                    if lbl_ctr != 0:
                        if amount_defined == 0:
                            new_hg_mw = [new_lbls, was_formerly]
                            amount_defined += 1
                        elif amount_defined == 1:
                            new_nor_mw = [new_lbls, was_formerly]
                            amount_defined += 1
                        elif amount_defined == 2:
                            new_lw_mw = [new_lbls, was_formerly]
                            amount_defined += 1
                        else: 
                            print('SOMETHING WENT REALLY WRONG!')
                            exit(-10)

            elif num_zero_lbls_ctr == 3:
                new_hg_mw = [None, None]
                new_nor_mw = [[], 'not existent in previous labels']
                new_lw_mw = [None, None]
                amount_defined = 0
                for (lbl_ctr, new_lbls, was_formerly) in [(vhg_mw_ctr, vhg_mw, 'vhg_mw'), (hg_mw_ctr, hg_mw, 'hg_mw'), (nor_mw_ctr, nor_mw, 'nor_mw'), (lw_mw_ctr, lw_mw, 'lw_mw'), (vlw_mw_ctr, vlw_mw, 'vlw_mw')]:
                    if lbl_ctr != 0:
                        if amount_defined == 0:
                            new_hg_mw = [new_lbls, was_formerly]
                            amount_defined += 1
                        elif amount_defined == 1:
                            new_lw_mw = [new_lbls, was_formerly]
                            amount_defined += 1
                        else: 
                            print('SOMETHING WENT REALLY WRONG!')
                            exit(-10)

            elif num_zero_lbls_ctr == 4:
                new_hg_mw = [[], 'not existent in previous labels']
                new_nor_mw = [None, None]
                new_lw_mw = [[], 'not existent in previous labels']
                amount_defined = 0
                for (lbl_ctr, new_lbls, was_formerly) in [(vhg_mw_ctr, vhg_mw, 'vhg_mw'), (hg_mw_ctr, hg_mw, 'hg_mw'), (nor_mw_ctr, nor_mw, 'nor_mw'), (lw_mw_ctr, lw_mw, 'lw_mw'), (vlw_mw_ctr, vlw_mw, 'vlw_mw')]:
                    if lbl_ctr != 0:
                        if amount_defined == 0:
                            new_nor_mw = [new_lbls, was_formerly]
                            amount_defined += 1
                        else: 
                            print('SOMETHING WENT REALLY WRONG!')
                            exit(-10)

            else:
                print('It cant be that no class has any instance... Something went really wrong... Investigate!')
                exit(-10)

            print('From a total of %d labels, the best new balance will be reached by defining %s as low, %s as neither low nor high, and %s as high...' % (num_total_labels, new_lw_mw[1], new_nor_mw[1], new_hg_mw[1]))

            vhg_mw = []
            hg_mw = new_hg_mw[0]
            nor_mw = new_nor_mw[0]
            lw_mw = new_lw_mw[0]
            vlw_mw = []

        elif binary_classification:
            print('### ### ### ### ### LABEL SHIFTING TO BINARY PROBLEM ### ### ### ### ###')
            print('We want this as a binary classification problem, so the classes will be modified in the following way:')

            vhg_mw_ctr = 0
            hg_mw_ctr = 0
            nor_mw_ctr = 0
            lw_mw_ctr = 0
            vlw_mw_ctr = 0
            for diction in [lab_class_counter, wild_class_counter]:
                for lbl, num_vals in diction.items():
                    if lbl == 'vhg_mw':
                        vhg_mw_ctr += num_vals
                    elif lbl == 'hg_mw':
                        hg_mw_ctr += num_vals
                    elif lbl == 'nor_mw':
                        nor_mw_ctr += num_vals
                    elif lbl == 'lw_mw':
                        lw_mw_ctr += num_vals
                    elif lbl == 'vlw_mw':
                        vlw_mw_ctr += num_vals

            num_total_labels = sum([vhg_mw_ctr, hg_mw_ctr, nor_mw_ctr, lw_mw_ctr, vlw_mw_ctr])
            print('Sum total labels: %d' % num_total_labels)
            approximate_half = 0.4 * num_total_labels
            print('approximate_half: %f' % approximate_half)

            splits = [
                [[vhg_mw_ctr], [hg_mw_ctr, nor_mw_ctr, lw_mw_ctr, vlw_mw_ctr]],
                [[vhg_mw_ctr, hg_mw_ctr], [nor_mw_ctr, lw_mw_ctr, vlw_mw_ctr]],
                [[vhg_mw_ctr, hg_mw_ctr, nor_mw_ctr], [lw_mw_ctr, vlw_mw_ctr]],
                [[vhg_mw_ctr, hg_mw_ctr, nor_mw_ctr, lw_mw_ctr], [vlw_mw_ctr]],
            ]

            splits_data = [
                [[vhg_mw], [hg_mw, nor_mw, lw_mw, vlw_mw]],
                [[vhg_mw, hg_mw], [nor_mw, lw_mw, vlw_mw]],
                [[vhg_mw, hg_mw, nor_mw], [lw_mw, vlw_mw]],
                [[vhg_mw, hg_mw, nor_mw, lw_mw], [vlw_mw]],
            ]

            distribution_to_use = None

            for split_idx, (split_option, high_class_members) in enumerate(splits):
                potential_low_class_labels = sum(split_option)
                print('For this split %d, this is the potential_low_class_labels: %d' % (split_idx, potential_low_class_labels))
                if potential_low_class_labels >= approximate_half:
                    # Found a potential new distribution
                    distribution_to_use = [split_option, high_class_members]
                    split_option, high_class_members = splits_data[split_idx]
                    new_hg_mw = []
                    new_lw_mw = []
                    for elems_container in split_option:
                        for elem in elems_container:
                            new_hg_mw.append(elem)
                    for elems_container in high_class_members:
                        for elem in elems_container:
                            new_lw_mw.append(elem)
                    vhg_mw = []
                    hg_mw = []
                    nor_mw = []
                    lw_mw = []
                    vlw_mw = []
                    hg_mw = new_hg_mw
                    lw_mw = new_lw_mw
                    break

        overall_class_counter = {'vlw_mw': len(vlw_mw), 'lw_mw': len(lw_mw), 'nor_mw': len(nor_mw), 'hg_mw': len(hg_mw), 'vhg_mw': len(vhg_mw)}
        print('Overall, for Lab and Wild combined, this is the NEW class distribution: %s' % overall_class_counter)
        print('### ### ### ### ### LABEL SHIFTING TO BINARY PROBLEM ### ### ### ### ###')

        if use_both_data:
            columns_np_df = ['EngagementIndex', 'BrainBeat', 'CLI', 'AsymmetryIndexAllChannels', 'AsymmetryIndexPrefrontal', 'AsymmetryIndexTemporal', 'PrefrontalDeltaPower', 'PrefrontalThetaPower', 'PrefrontalAlphaPower', 'PrefrontalBetaPower', 'PrefrontalGammaPower', 'TemporalDeltaPower', 'TemporalThetaPower', 'TemporalAlphaPower', 'TemporalBetaPower', 'TemporalGammaPower','min_scr', 'max_scr', 'num_scr_peaks', 'min_scl', 'max_scl', 'mean_scl', 'min_skt', 'max_skt', 'mean_skt', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_LF', 'HRV_HF', 'HRV_ratio_LF_HF', 'Min_Heart-Rate', 'Mean_Heart-Rate', 'Max_Heart-Rate']
        elif use_muse_data:
            columns_np_df = ['EngagementIndex', 'BrainBeat', 'CLI', 'AsymmetryIndexAllChannels', 'AsymmetryIndexPrefrontal', 'AsymmetryIndexTemporal', 'PrefrontalDeltaPower', 'PrefrontalThetaPower', 'PrefrontalAlphaPower', 'PrefrontalBetaPower', 'PrefrontalGammaPower', 'TemporalDeltaPower', 'TemporalThetaPower', 'TemporalAlphaPower', 'TemporalBetaPower', 'TemporalGammaPower']
        elif use_e4_data:
            columns_np_df = ['min_scr', 'max_scr', 'num_scr_peaks', 'min_scl', 'max_scl', 'mean_scl', 'min_skt', 'max_skt', 'mean_skt', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_LF', 'HRV_HF', 'HRV_ratio_LF_HF', 'Min_Heart-Rate', 'Mean_Heart-Rate', 'Max_Heart-Rate']

        row_lengths_vhg_mw = [len(row) for row in vhg_mw] if len(vhg_mw) >= 1 else None
        row_lengths_hg_mw = [len(row) for row in hg_mw] if len(hg_mw) >= 1 else None
        row_lengths_nor_mw = [len(row) for row in nor_mw] if len(nor_mw) >= 1 else None
        row_lengths_lw_mw = [len(row) for row in lw_mw] if len(lw_mw) >= 1 else None
        row_lengths_vlw_mw = [len(row) for row in vlw_mw] if len(vlw_mw) >= 1 else None

        row_lengths_values_available = []
        lengts_of_rows_over_all_data = []
        for vals in [row_lengths_vhg_mw, row_lengths_hg_mw, row_lengths_nor_mw, row_lengths_lw_mw, row_lengths_vlw_mw]:
            if vals is not None:
                for row_len in vals:
                    lengts_of_rows_over_all_data.append(row_len)

        most_common_length = most_common(lengts_of_rows_over_all_data)

        vhg_mw_same_length = []
        vhg_mw_not_considered_ctr = 0
        for row in vhg_mw:
            if len(row) == most_common_length:
                vhg_mw_same_length.append(row)
            else:
                vhg_mw_not_considered_ctr += 1
        print('Put to equal length vhg_mw and thereby dropped %d rows' % vhg_mw_not_considered_ctr)
        
        hg_mw_same_length = []
        hg_mw__not_considered_ctr = 0
        for row in hg_mw:
            if len(row) == most_common_length:
                hg_mw_same_length.append(row)
            else:
                hg_mw__not_considered_ctr += 1
        print('Put to equal length hg_mw and thereby dropped %d rows' % hg_mw__not_considered_ctr)
        
        nor_mw_same_length = []
        nor_mw_not_considered_ctr = 0
        for row in nor_mw:
            if len(row) == most_common_length:
                nor_mw_same_length.append(row)
            else:
                nor_mw_not_considered_ctr += 1
        print('Put to equal length nor_mw and thereby dropped %d rows' % nor_mw_not_considered_ctr)
        
        lw_mw_same_length = []
        lw_mw__not_considered_ctr = 0
        for row in lw_mw:
            if len(row) == most_common_length:
                lw_mw_same_length.append(row)
            else:
                lw_mw__not_considered_ctr += 1
        print('Put to equal length lw_mw and thereby dropped %d rows' % lw_mw__not_considered_ctr)
        
        vlw_mw_same_length = []
        vlw_mw_not_considered_ctr = 0
        for row in vlw_mw:
            if len(row) == most_common_length:
                vlw_mw_same_length.append(row)
            else:
                vlw_mw_not_considered_ctr += 1
        print('Put to equal length vlw_mw and thereby dropped %d rows' % vlw_mw_not_considered_ctr)

        try:
            vhg_mw = np.asarray(vhg_mw_same_length, dtype=np.float32)
            hg_mw = np.asarray(hg_mw_same_length, dtype=np.float32)
            nor_mw = np.asarray(nor_mw_same_length, dtype=np.float32)
            lw_mw = np.asarray(lw_mw_same_length, dtype=np.float32)
            vlw_mw = np.asarray(vlw_mw_same_length, dtype=np.float32)
        except Exception as exc:
            print('Issue in making np arrays of the data')
            print(exc)

        try:
            if ((lab_class_counter['vhg_mw'] + wild_class_counter['vhg_mw']) >= 1) and (len(vhg_mw) >= 1):
                col_means = np.mean(vhg_mw, axis=0)
                vhg_mw = (vhg_mw - col_means) / np.std(vhg_mw, axis=0)
            if ((lab_class_counter['hg_mw'] + wild_class_counter['hg_mw']) >= 1) and (len(hg_mw) >= 1):
                col_means = np.mean(hg_mw, axis=0)
                hg_mw = (hg_mw - col_means) / np.std(hg_mw, axis=0)
            if ((lab_class_counter['nor_mw'] + wild_class_counter['nor_mw']) >= 1) and (len(nor_mw) >= 1):
                col_means = np.mean(nor_mw, axis=0)
                nor_mw = (nor_mw - col_means) / np.std(nor_mw, axis=0)
            if ((lab_class_counter['lw_mw'] + wild_class_counter['lw_mw']) >= 1) and (len(lw_mw) >= 1):
                col_means = np.mean(lw_mw, axis=0)
                lw_mw = (lw_mw - col_means) / np.std(lw_mw, axis=0)
            if ((lab_class_counter['vlw_mw'] + wild_class_counter['vlw_mw']) >= 1) and (len(vlw_mw) >= 1):
                col_means = np.mean(vlw_mw, axis=0)
                vlw_mw = (vlw_mw - col_means) / np.std(vlw_mw, axis=0)
        except Exception as exc:
            print('Issue in normalizing the data')
            print(exc)
        
        temp_exes = [data_features if (len(data_features) > 1) else None for data_features in [vhg_mw, hg_mw, nor_mw, lw_mw, vlw_mw]]
        temp_exes_without_Nones = []

        for temp_ex in temp_exes:
            if temp_ex is not None:
                temp_exes_without_Nones.append(temp_ex)

        if three_class_classification:
            print('USED LABELS FOR THREE CLASS CLASSIFICATION!')
            temp_eyes = [np.ones(len(data_labels), dtype=np.int8) * multiplier if (len(data_labels) > 1) else None for data_labels, multiplier in [(hg_mw, 1), (nor_mw, 2), (lw_mw, 3)]]
        elif binary_classification:
            print('USED LABELS FOR BINARY CLASSIFICATION!')
            temp_eyes = [np.ones(len(data_labels), dtype=np.int8) * multiplier if (len(data_labels) > 1) else None for data_labels, multiplier in [(hg_mw, 1), (lw_mw, 2)]]
            print(temp_eyes)
            print('FINDMEEEE')
        elif five_class_classification:
            print('USED LABELS FOR FIVE-CLASS CLASSIFICATION!')
            temp_eyes = [np.ones(len(data_labels), dtype=np.int8) * multiplier if (len(data_labels) > 1) else None for data_labels, multiplier in [(vhg_mw, 0), (hg_mw, 1), (nor_mw, 2), (lw_mw, 3), (vlw_mw, 4)]]
        temp_eyes_without_Nones = []

        for temp_ey in temp_eyes:
            if temp_ey is not None:
                temp_eyes_without_Nones.append(temp_ey)

        try:
            X = np.concatenate([data_features for data_features in temp_exes_without_Nones])
            y = np.concatenate([data_labels for data_labels in temp_eyes_without_Nones])
        except:
            for data_features in temp_exes_without_Nones:
                print_full(pd.DataFrame(data_features))
            for data_labels in temp_eyes_without_Nones:
                print_full(pd.DataFrame(data_labels))
            exit(-1)

        np.random.seed(24)
        np.random.shuffle(X)
        np.random.shuffle(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print('Len of X_train and X_test: %d to %d' % (len(X_train), len(X_test)))

        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
        data_tuple_for_task_classification = [X_train, X_test, y_train, y_test]
        participant_data.append(data_tuple_for_task_classification)

    return participant_data, feature_labels_to_use


participant_data, feature_labels_probably_to_use = get_data_all_participants_and_loo_participant()

if use_activity_labels:
    # Activity labels included here:    'Relaxation', 'LoadTask', 'Summary', 'Reading', 'Game'
    if use_both_data:
        columns_np_df_w_labels = ['db2-Mean-cD1-Kurtosis', 'db2-Notchd_AF7-cD1-MEAN', 'db2-Notchd_AF7-cD3-ZeroCrossing', 'db2-Notchd_AF8-cD1-MAX', 'db2-Notchd_AF8-cD2-MAX', 'db2-Notchd_TP10-cD1-Kurtosis', 'db2-Notchd_TP10-cD8-MIN', 'db2-Notchd_TP9-cA8-ZeroCrossing', 'db2-Notchd_TP9-cD7-Skewness', 'haar-Notchd_AF7-cD1-MEAN', 'haar-Notchd_AF7-cD7-ZeroCrossing', 'haar-Notchd_TP10-cD8-ZeroCrossing', 'max_scr', 'HRV_SDNN', 'HRV_RMSSD', 'Max_Heart-Rate', 'max_skt', 'max_scl', 'HRV_MeanNN', 'HRV_ratio_LF_HF', 'Relaxation', 'LoadTask', 'Summary', 'Reading', 'Game']
    elif use_muse_data:
        columns_np_df_w_labels = ['db2-Mean-cD1-Kurtosis', 'db2-Notchd_AF7-cD1-MEAN', 'db2-Notchd_AF7-cD3-ZeroCrossing', 'db2-Notchd_AF8-cD1-MAX', 'db2-Notchd_AF8-cD2-MAX', 'db2-Notchd_TP10-cD1-Kurtosis', 'db2-Notchd_TP10-cD8-MIN', 'db2-Notchd_TP9-cA8-ZeroCrossing', 'db2-Notchd_TP9-cD7-Skewness', 'haar-Notchd_AF7-cD1-MEAN', 'haar-Notchd_AF7-cD7-ZeroCrossing', 'haar-Notchd_TP10-cD8-ZeroCrossing', 'TemporalDeltaPower', 'PrefrontalGammaPower', 'TemporalThetaPower', 'TemporalBetaPower', 'TemporalGammaPower', 'EngagementIndex', 'AsymmetryIndexPrefrontal', 'CLI', 'Relaxation', 'LoadTask', 'Summary', 'Reading', 'Game']
    elif use_e4_data:
        columns_np_df_w_labels = ['max_scr', 'HRV_SDNN', 'HRV_RMSSD', 'Max_Heart-Rate', 'max_skt', 'max_scl', 'HRV_MeanNN', 'HRV_ratio_LF_HF', 'min_scr', 'num_scr_peaks', 'min_scl', 'mean_scl', 'min_skt', 'mean_skt', 'HRV_LF', 'HRV_HF', 'Min_Heart-Rate', 'Mean_Heart-Rate', 'Relaxation', 'LoadTask', 'Summary', 'Reading', 'Game']
else:
    # Activity labels NOT included here
    if use_both_data:
        columns_np_df_w_labels = ['db2-Mean-cD1-Kurtosis', 'db2-Notchd_AF7-cD1-MEAN', 'db2-Notchd_AF7-cD3-ZeroCrossing', 'db2-Notchd_AF8-cD1-MAX', 'db2-Notchd_AF8-cD2-MAX', 'db2-Notchd_TP10-cD1-Kurtosis', 'db2-Notchd_TP10-cD8-MIN', 'db2-Notchd_TP9-cA8-ZeroCrossing', 'db2-Notchd_TP9-cD7-Skewness', 'haar-Notchd_AF7-cD1-MEAN', 'haar-Notchd_AF7-cD7-ZeroCrossing', 'haar-Notchd_TP10-cD8-ZeroCrossing', 'max_scr', 'HRV_SDNN', 'HRV_RMSSD', 'Max_Heart-Rate', 'max_skt', 'max_scl', 'HRV_MeanNN', 'HRV_ratio_LF_HF']
    elif use_muse_data:
        columns_np_df_w_labels = ['db2-Mean-cD1-Kurtosis', 'db2-Notchd_AF7-cD1-MEAN', 'db2-Notchd_AF7-cD3-ZeroCrossing', 'db2-Notchd_AF8-cD1-MAX', 'db2-Notchd_AF8-cD2-MAX', 'db2-Notchd_TP10-cD1-Kurtosis', 'db2-Notchd_TP10-cD8-MIN', 'db2-Notchd_TP9-cA8-ZeroCrossing', 'db2-Notchd_TP9-cD7-Skewness', 'haar-Notchd_AF7-cD1-MEAN', 'haar-Notchd_AF7-cD7-ZeroCrossing', 'haar-Notchd_TP10-cD8-ZeroCrossing', 'TemporalDeltaPower', 'PrefrontalGammaPower', 'TemporalThetaPower', 'TemporalBetaPower', 'TemporalGammaPower', 'EngagementIndex', 'AsymmetryIndexPrefrontal', 'CLI']
    elif use_e4_data:
        columns_np_df_w_labels = ['max_scr', 'HRV_SDNN', 'HRV_RMSSD', 'Max_Heart-Rate', 'max_skt', 'max_scl', 'HRV_MeanNN', 'HRV_ratio_LF_HF', 'min_scr', 'num_scr_peaks', 'min_scl', 'mean_scl', 'min_skt', 'mean_skt', 'HRV_LF', 'HRV_HF', 'Min_Heart-Rate', 'Mean_Heart-Rate']

feature_labels_probably_to_use = list(feature_labels_probably_to_use)

if 'Start_Time_EEG' in feature_labels_probably_to_use:
    feature_labels_probably_to_use.remove('Start_Time_EEG')
if 'skt_start_time' in feature_labels_probably_to_use:
    feature_labels_probably_to_use.remove('skt_start_time')

if top_stat_features_only:
    elems_to_use = [elem in columns_np_df_w_labels for elem in feature_labels_probably_to_use]
else:
    elems_to_use = [True for elem in feature_labels_probably_to_use]

if top_stat_features_only:
    feature_labels_to_use = columns_np_df_w_labels.copy()
else:
    feature_labels_to_use = feature_labels_probably_to_use.copy()

feature_labels_to_use.append('label')   # This was added only for the correlation analysis but will not be included in the actual classification!
do_correlation_analysis = True
correlation_scores_pearson = []
correlation_scores_spearman = []

for loo_participant in range(0,10):
    X_train, X_test, y_train, y_test = None, None, None, None
    print('LOO %d ML and Analysis Started' % (loo_participant + 1))
    for participant_number in range(0,10):
        print('STARTING WITH PARTICIPANT %d, WHO %s THE LOO PARTICIPANT!' % ((participant_number + 1), 'IS' if participant_number == loo_participant else 'IS NOT'))
        if loo_participant == participant_number:
            loo_X_train = participant_data[loo_participant][0][:, elems_to_use]
            loo_X_test = participant_data[loo_participant][1][:, elems_to_use]
            loo_y_train = participant_data[loo_participant][2]
            loo_y_test = participant_data[loo_participant][3]
            loo_y_train = np.expand_dims(loo_y_train, axis=-1)
            loo_y_test = np.expand_dims(loo_y_test, axis=-1)
            X_test = np.concatenate((loo_X_train, loo_X_test), axis=0)
            y_test = np.concatenate((loo_y_train, loo_y_test), axis=0)

            if do_correlation_analysis:
                print('PANDAS -- Feature Analysis and Correlation Analysis For Participant %d' % (participant_number + 1))
                try:
                    all_data = np.concatenate((X_test, y_test), axis=1)
                    all_data_df = pd.DataFrame(all_data, columns=feature_labels_to_use)
                    correlation_scores_pearson.append(all_data_df[all_data_df.columns[:]].corr()['label'][:])
                    correlation_scores_spearman.append(all_data_df[all_data_df.columns[:]].corr(method='spearman')['label'][:])
                except Exception as exc:
                    print(loo_X_train.shape)
                    print(loo_X_test.shape)
                    print(loo_y_train.shape)
                    print(loo_y_test.shape)
                    print(X_test.shape)
                    print(y_test.shape)
                    print(all_data.shape)
                    print('Exception:')
                    print(exc)

        else:
            participant_X_train = participant_data[participant_number][0][:, elems_to_use]
            participant_X_test = participant_data[participant_number][1][:, elems_to_use]
            participant_y_train = participant_data[participant_number][2]
            participant_y_test = participant_data[participant_number][3]
            participant_y_train = np.expand_dims(participant_y_train, axis=-1)
            participant_y_test = np.expand_dims(participant_y_test, axis=-1)
            participant_data_X_data = np.concatenate((participant_X_train, participant_X_test), axis=0)
            participant_data_y_data = np.concatenate((participant_y_train, participant_y_test), axis=0)
            if X_train is None:
                X_train = participant_data_X_data
                y_train = participant_data_y_data
            else:
                X_train = np.concatenate((X_train, participant_data_X_data), axis=0)
                y_train = np.concatenate((y_train, participant_data_y_data), axis=0)

            if do_correlation_analysis:
                try:
                    print('PANDAS -- Feature Analysis and Correlation Analysis For Participant %d' % (participant_number + 1))
                    all_data = np.concatenate((participant_data_X_data, participant_data_y_data), axis=1)
                    all_data_df = pd.DataFrame(all_data, columns=feature_labels_to_use)
                    correlation_scores_pearson.append(all_data_df[all_data_df.columns[:]].corr()['label'][:])
                    correlation_scores_spearman.append(all_data_df[all_data_df.columns[:]].corr(method='spearman')['label'][:])
                except Exception as exc: 
                    print(participant_X_train.shape)
                    print(participant_X_test.shape)
                    print(participant_y_train.shape)
                    print(participant_y_test.shape)
                    print(participant_data_X_data.shape)
                    print(participant_data_y_data.shape)
                    print(all_data.shape)
                    print('Exception:')
                    print(exc)

    if do_correlation_analysis:
        correlation_cols_pearson = correlation_scores_pearson[0].index
        correlation_cols_spearman = correlation_scores_spearman[0].index

        averaged_results_pearson = (np.sum(np.asarray([np.abs(arr) for arr in correlation_scores_pearson]), axis=0)) / len(correlation_scores_pearson)
        averaged_results_spearman = (np.sum(np.asarray([np.abs(arr) for arr in correlation_scores_spearman]), axis=0)) / len(correlation_scores_spearman)

        np.savetxt(("./stats_results/correlation_results_pearson_on_%s_for_%s_classes_%s_and_%s.csv" % (sensors, classification_here, feature_reduction_str, with_activity_labels_str)), np.asarray(correlation_scores_pearson), delimiter=",")
        np.savetxt(("./stats_results/correlation_results_spearman_on_%s_for_%s_classes_%s_and_%s.csv" % (sensors, classification_here, feature_reduction_str, with_activity_labels_str)), np.asarray(correlation_scores_spearman), delimiter=",")

        np.savetxt(("./stats_results/averaged_correlation_results_pearson_on_%s_for_%s_classes_%s_and_%s.csv" % (sensors, classification_here, feature_reduction_str, with_activity_labels_str)), averaged_results_pearson, delimiter=",")
        np.savetxt(("./stats_results/averaged_correlation_results_spearman_on_%s_for_%s_classes_%s_and_%s.csv" % (sensors, classification_here, feature_reduction_str, with_activity_labels_str)), averaged_results_spearman, delimiter=",")

    print('For this run started at %d its current ms time: %d' % (start_time, int(time.time())))

    print('Len of y_train and y_test: %d to %d' % (len(y_train), len(y_test)))
    unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
    unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
    print('For y_train, this is the label-distribution: %s' % dict(zip(unique_y_train, counts_y_train)))
    print('For y_test, this is the label-distribution: %s' % dict(zip(unique_y_test, counts_y_test)))

    ### ### ### ### DO THE REAL CLASSIFICATION! ### ### ### ###
    f1 = make_scorer(f1_score, average='weighted')

    all_epochs = np.append(X_train, X_test, axis=0)
    all_epochs = np.nan_to_num(all_epochs)
    all_labels = np.append(y_train, y_test)

    print('Shape of all_epochs: %s' % str(all_epochs.shape))
    print('Shape of all_labels: %s' % str(all_labels.shape))

    csv_results_header = ['Best_Params', 'Heldout_Test_F1_or_ACC', 'best_predictor_f1_score', 'y_test', 'best_predictor_predictions']
    n_jobs = -1
    min_max_scaler = preprocessing.MinMaxScaler()
    standard_scaler = StandardScaler()
    standard_scaler.fit(X_train)
    X_train = standard_scaler.transform(X_train)
    X_test = standard_scaler.transform(X_test)

    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    parameter_grid_lr = [
        {'solver': ['lbfgs'], 'penalty': ['l2', None]},
        {'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
        {'solver': ['sag'], 'penalty': ['l2', None]},
        {'solver': ['saga'], 'penalty': ['l1', 'l2', None]}
    ]

    parameter_grid_dt = [
        {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': np.concatenate((np.arange(5, 305, 5), np.asarray([None])), axis=0)}
    ]

    parameter_grid_svm = [
        {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    ]

    parameter_grid_nn = [
        {'leaf_size': list(range(1,50)), 'n_neighbors': list(range(1,30)), 'p': [1, 2]}
    ]

    parameter_grid_mlp = [
        {'activation': ['logistic', 'tanh', 'relu'], 'hidden_layer_sizes': [(3,), (10,), (30,), (50,)]}
    ]

    model_cv_definitions = [
        [parameter_grid_nn, 'NN-LABELS'],
        [parameter_grid_svm, 'SVM-LABELS'],
        [parameter_grid_dt, 'DT-LABELS'],
        [parameter_grid_lr, 'LR-LABELS'],
        [parameter_grid_mlp, 'MLP-LABELS'],
    ]

    results_per_model = []

    for j, (grid, abbrev) in enumerate(model_cv_definitions):
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        models_skf = []

        if abbrev == 'LR-LABELS':
            classifier = LogisticRegression(random_state=42, max_iter=10000)
        elif abbrev == 'DT-LABELS':
            classifier = DecisionTreeClassifier()
        elif abbrev == 'SVM-LABELS':
            classifier = LinearSVC(dual=False, max_iter=10000)
        elif abbrev == 'NN-LABELS':
            classifier = KNeighborsClassifier()
        elif abbrev == 'MLP-LABELS':
            classifier = MLPClassifier(random_state=42, max_iter=10000, early_stopping=True)

        gridsearch_cv = GridSearchCV(classifier, param_grid=grid, n_jobs=n_jobs, scoring=f1).fit(X=X_train, y=y_train)
        print('CV done! Best estimator: %s and best score: %f and best params: %s' % (gridsearch_cv.best_estimator_, gridsearch_cv.best_score_, gridsearch_cv.best_params_))
        optimized_clf_score = gridsearch_cv.best_estimator_.score(X=X_test, y=y_test)
        best_predictor_predictions = gridsearch_cv.best_estimator_.predict(X_test)
        best_predictor_f1_score = f1_score(y_test, best_predictor_predictions, average='weighted')
        print('Split %d test scoring done! Score: %f and F1: %f' % (i, optimized_clf_score, best_predictor_f1_score))

        results_per_model.append([j, [gridsearch_cv.best_estimator_, gridsearch_cv.best_params_, optimized_clf_score, best_predictor_f1_score, y_test, best_predictor_predictions], abbrev])

    for (run, cv_results, abbrev) in results_per_model:
        mean_f1 = 0
        mean_value = 0
        mean_value_ctr = 0
        best_model_score_over_runs = 0
        best_model_params = None
        _, best_params_of_best_model, best_model_score_this_run, f1, _, _ = cv_results
        mean_f1 += f1
        mean_value += best_model_score_this_run
        mean_value_ctr += 1
        if best_model_score_this_run >= best_model_score_over_runs:
            best_model_score_over_runs = best_model_score_this_run
            best_model_params = best_params_of_best_model
        print('%s: F1: %f, %s,' % (abbrev, f1, best_params_of_best_model))
        print('RESULTS: For %s the mean nested-cv score is: %f and mean nested-F1 is: %f' % (abbrev, mean_value / mean_value_ctr, mean_f1 / mean_value_ctr))
        print('RESULTS: Best Score of %f achieved for %s with params %s' % (best_model_score_over_runs, abbrev, best_model_params))

    for (run, cv_results, abbrev) in results_per_model:
        model_storage_path = reults_path + model_name % ((loo_participant + 1), abbrev, classification_here, feature_reduction_str, with_activity_labels_str)
        _, best_params_of_best_model, best_model_score_this_run, f1, y_test, best_predictor_predictions = cv_results
        with open(model_storage_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_results_header)
            writer.writerow([str(best_params_of_best_model), str(best_model_score_this_run), str(f1), str(y_test), str(best_predictor_predictions)])
        print('Stored results for %s in csv-file %s' % (abbrev, model_storage_path))

    print('LOO %d ML and Analysis Finished' % (loo_participant + 1))
    do_correlation_analysis = False
    
print('For this run started at %d its current ms time: %d' % (start_time, int(time.time())))
### ### ### ### END OF REAL CLASSIFICATION! ### ### ### ###