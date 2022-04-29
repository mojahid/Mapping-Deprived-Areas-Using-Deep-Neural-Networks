# project_root.py
import git
from pathlib import Path

def get_project_root():
    return Path(git.Repo('.', search_parent_directories=True).working_tree_dir)

from project_root import get_project_root
root = get_project_root()

df = pd.read_csv(root / '1.Data' / 'Contextual_data.csv')
print(df.head())

# write dataframe to csv
filename = 'test.csv'
df.to_csv(root / '1.Data' / f'{filename}', index=False)