from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    dataset = pd.DataFrame(columns=['morph_path', 'mtype'])
    dataset.index.name = 'morph_name'
    morph_path = Path("morphologies")
    for morph in morph_path.iterdir():
        if morph.suffix in ['.asc', '.h5', '.swc']:
            dataset.loc[morph.stem, 'morph_path'] = morph
            dataset.loc[morph.stem, 'mtype'] = 'L1_AAA:C'
    dataset.loc['AA0319', 'mtype'] = 'L6_TPC:A'
    dataset.loc['rp100427-123_idC', 'mtype'] = 'L4_UPC'
    dataset.loc['C270106A', 'mtype'] = 'L1_DAC'
    dataset.sort_index(inplace=True)
    dataset.reset_index().to_csv('dataset.csv', index=False)
