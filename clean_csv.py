import os
import pandas as pd


if __name__ == "__main__":
    dirs = ['Results/DocRepXLM', 'Results/XlmNer']
    for d in dirs:
        df = pd.read_csv(os.path.join(d, 'Results.csv'), index_col=0)
        df['base_model'] = df.base_model.apply(lambda s: s.split('.')[-1].split("'")[0].split(' ')[0])
        df.to_csv(os.path.join(d, 'Results.csv'))