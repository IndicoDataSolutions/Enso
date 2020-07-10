import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import numpy as np

from enso.utils import labels_to_binary

l = np.load('test_labels.npy', allow_pickle=True)
t = np.load('test_set.npy', allow_pickle=True)
p = np.load('test_pred.npy')

labels = [el[1] for el in l]
binary_labels = labels_to_binary(labels)
predicted = pd.DataFrame(p, columns=['APP',  'QA',  'QUOTE',  'SA',  'SKIP'])
idx = np.where(binary_labels.idxmax(axis=1)!= predicted.idxmax(axis=1))
print(predicted.idxmax(axis=1).values[idx].tolist())
print([el[1] for el in l[idx]])
print(idx)
import ipdb; ipdb.set_trace()
