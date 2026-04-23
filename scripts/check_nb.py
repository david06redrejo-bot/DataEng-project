import json
nb = json.load(open('notebooks/01_exploratory_data_analysis.ipynb', encoding='utf-8'))
for c in nb['cells']:
    ct = c['cell_type']
    out = len(c.get('outputs', []))
    first = c['source'][0][:70].strip() if c['source'] else ''
    print(f'{ct:8s} | outputs={out:2d} | {first}')
