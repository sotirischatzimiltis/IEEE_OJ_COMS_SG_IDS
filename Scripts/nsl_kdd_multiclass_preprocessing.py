import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings("ignore")

"""# Read Datasets"""

TRAIN_FILE = "../Datasets/KDDTrain.txt"
TEST_FILE = "../Datasets/KDDTest.txt"

dataset_train = pd.read_csv(TRAIN_FILE, sep=",", header=None)
dataset_train = dataset_train.iloc[:, :-1]  # remove difficulty column

dataset_test = pd.read_csv(TEST_FILE, sep=",", header=None)
dataset_test = dataset_test.iloc[:, :-1]  # remove difficulty column

"""Columns names of Training and Test Datasets"""

col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

"""Shape of Training and Test Datasets"""

print("Shape of Training Dataset:", dataset_train.shape)
print("Shape of Testing Dataset:", dataset_test.shape)

"""Columns Assignment"""

dataset_train.columns = col_names  # append labels
dataset_test.columns = col_names


"""Labels of Training and Test Datasets"""

# label distribution of Training set and testing set
print('Label distribution Training set:')
print(dataset_train['label'].value_counts())
train_labels = dataset_train['label'].unique()
print()
print('Label distribution Test set:')
print(dataset_test['label'].value_counts())

""" Data Pre-processing """

#  Drop Records Containing Infinite Values, are Null and Nan Values
# Replace infinite number with NaN values
dataset_train.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset_test.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop null and NaN values
print("Corrupted records were deleted in training dataset")
print("Corrupted records were deleted in test dataset")
dataset_train = dataset_train.dropna().reset_index(drop=True)
dataset_test = dataset_test.dropna().reset_index(drop=True)

print("Shape of Training Dataset:", dataset_train.shape)
print("Shape of Testing Dataset:", dataset_test.shape)

"""## Label-Encoding to Replace Categorical Values with Numerical"""
# columns that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
# explore categorical features

print('Training set:')
for col_name in dataset_train.columns:
    if dataset_train[col_name].dtypes == 'object':
        unique_cat = dataset_train[col_name].nunique()
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

# Test set
print('Test set:')
for col_name in dataset_test.columns:
    if dataset_test[col_name].dtypes == 'object':
        unique_cat = len(dataset_test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

"""One-hot-Encoding"""

one_hot_encoder = preprocessing.OneHotEncoder()  # create one hot encoder

"""Protocol Type"""
# fit one hot encoder
one_hot_encoder.fit(dataset_train['protocol_type'].values.reshape(-1, 1))
# train dataset
train_transform = one_hot_encoder.transform(dataset_train['protocol_type'].values.reshape(-1, 1))
train_transform = train_transform.toarray()
train_df = pd.DataFrame(train_transform, columns=one_hot_encoder.get_feature_names_out())
del dataset_train['protocol_type']
dataset_train = pd.concat([dataset_train, train_df], axis=1)
# test dataset
test_transform = one_hot_encoder.transform(dataset_test['protocol_type'].values.reshape(-1, 1))
test_transform = test_transform.toarray()
test_df = pd.DataFrame(test_transform, columns=one_hot_encoder.get_feature_names_out())
del dataset_test['protocol_type']
dataset_test = pd.concat([dataset_test, test_df], axis=1)

"""Service"""
# fit one hot encoder
one_hot_encoder.fit(dataset_train['service'].values.reshape(-1, 1))
# train dataset
train_transform = one_hot_encoder.transform(dataset_train['service'].values.reshape(-1, 1))
train_transform = train_transform.toarray()
train_df = pd.DataFrame(train_transform, columns=one_hot_encoder.get_feature_names_out())
del dataset_train['service']
dataset_train = pd.concat([dataset_train, train_df], axis=1)
# test dataset
test_transform = one_hot_encoder.transform(dataset_test['service'].values.reshape(-1, 1))
test_transform = test_transform.toarray()
test_df = pd.DataFrame(test_transform, columns=one_hot_encoder.get_feature_names_out())
del dataset_test['service']
dataset_test = pd.concat([dataset_test, test_df], axis=1)

"""Flag"""
# fit one hot encoder
one_hot_encoder.fit(dataset_train['flag'].values.reshape(-1, 1))
# train dataset
train_transform = one_hot_encoder.transform(dataset_train['flag'].values.reshape(-1, 1))
train_transform = train_transform.toarray()
train_df = pd.DataFrame(train_transform, columns=one_hot_encoder.get_feature_names_out())
del dataset_train['flag']
dataset_train = pd.concat([dataset_train, train_df], axis=1)
# test dataset
test_transform = one_hot_encoder.transform(dataset_test['flag'].values.reshape(-1, 1))
test_transform = test_transform.toarray()
test_df = pd.DataFrame(test_transform, columns=one_hot_encoder.get_feature_names_out())
del dataset_test['flag']
dataset_test = pd.concat([dataset_test, test_df], axis=1)


print("Shape of Training Dataset:", dataset_train.shape)
print("Shape of Testing Dataset:", dataset_test.shape)

""" Find and remove features that only contribute zero values """

zero_columns = dataset_train.columns[dataset_train.eq(0).all()]
print("Features with only zero values:")
print(zero_columns)
dataset_train = dataset_train.drop(columns=zero_columns)
dataset_test = dataset_test.drop(columns=zero_columns)

print("Shape of Training Dataset:", dataset_train.shape)
print("Shape of Testing Dataset:", dataset_test.shape)

"""## Multiclass Classification Dataset

Rename every attack label: 0=normal, 1=DoS, 2=Probe, 3=R2L and 4=U2R.
Replace labels column with new labels column
"""

dataset_train_multiclass = dataset_train.copy()
dataset_train_multiclass = dataset_train_multiclass.replace({'normal': 0, 'neptune': 1, 'back': 1, 'land': 1, 'pod': 1,
                                                             'smurf': 1, 'teardrop': 1, 'mailbomb': 1, 'apache2': 1,
                                                             'processtable': 1, 'udpstorm': 1, 'worm': 1, 'ipsweep': 2,
                                                             'nmap': 2, 'portsweep': 2,'satan': 2, 'mscan': 2,
                                                             'saint': 2, 'ftp_write': 3, 'guess_passwd': 3, 'imap': 3,
                                                             'multihop': 3, 'phf': 3, 'spy': 3, 'warezclient': 3,
                                                             'warezmaster': 3, 'sendmail': 3, 'named': 3,
                                                             'snmpgetattack': 3, 'snmpguess': 3, 'xlock': 3, 'xsnoop': 3
                                                            , 'httptunnel': 3, 'buffer_overflow': 4, 'loadmodule': 4,
                                                             'perl': 4, 'rootkit': 4, 'ps': 4, 'sqlattack': 4,
                                                             'xterm': 4})
print(dataset_train_multiclass['label'])

dataset_test_multiclass = dataset_test.copy()
dataset_test_multiclass = dataset_test_multiclass.replace({'normal': 0, 'neptune': 1, 'back': 1, 'land': 1, 'pod': 1,
                                                           'smurf': 1, 'teardrop': 1, 'mailbomb': 1, 'apache2': 1,
                                                           'processtable': 1, 'udpstorm': 1, 'worm': 1, 'ipsweep': 2,
                                                           'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2,
                                                           'saint': 2, 'ftp_write': 3, 'guess_passwd': 3, 'imap': 3,
                                                           'multihop': 3, 'phf': 3, 'spy': 3, 'warezclient': 3,
                                                           'warezmaster': 3, 'sendmail': 3, 'named': 3,
                                                           'snmpgetattack': 3, 'snmpguess': 3 ,'xlock': 3, 'xsnoop': 3,
                                                           'httptunnel': 3, 'buffer_overflow': 4, 'loadmodule': 4,
                                                           'perl': 4, 'rootkit': 4, 'ps': 4, 'sqlattack': 4,
                                                           'xterm': 4})


print(dataset_test_multiclass['label'])

"""Split features and labels"""
x_train = dataset_train_multiclass.drop('label', 1)
col_names = x_train.columns
y_train = dataset_train_multiclass.label

x_test = dataset_test_multiclass.drop('label', 1)
y_test = dataset_test_multiclass.label


"""## Save initial dataset without normalization csv"""
x_train = pd.DataFrame(x_train)
x_train.columns = col_names
y_train = pd.DataFrame(y_train)
y_train.columns = ['label']
train_frame = [x_train, y_train]
train_final = pd.concat(train_frame, axis=1)
train_final.to_csv('train_multiclass20.csv', index=False)

x_test = pd.DataFrame(x_test)
x_test.columns = col_names
y_test = pd.DataFrame(y_test)
y_test.columns = ['label']
test_frame = [x_test, y_test]
test_final = pd.concat(test_frame, axis=1)
test_final.to_csv('test_multiclass20.csv', index=False)


""" Min Max Scaling """
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


""" Save dataset after normalization"""
x_train = pd.DataFrame(x_train)
x_train.columns = col_names
y_train = pd.DataFrame(y_train)
y_train.columns = ['label']
train_frame = [x_train, y_train]
train_final = pd.concat(train_frame, axis=1)
train_final.to_csv('train_multiclass_scaled20.csv', index=False)

x_test = pd.DataFrame(x_test)
x_test.columns = col_names
y_test = pd.DataFrame(y_test)
y_test.columns = ['label']
test_frame = [x_test, y_test]
test = pd.concat(test_frame, axis=1)
test.to_csv('test_multiclass_scaled20.csv', index=False)



