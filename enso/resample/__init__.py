from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import ADASYN


def resample(resample_type, train_data, train_labels):
    if resample_type.lower() == 'none':
        return train_data, train_labels
    else:
        instance = globals()[resample_type]()
        try:
            return instance.fit_sample(train_data, train_labels)
        except ValueError:
            # this error occurs if there are already even number of samples
            return train_data, train_labels