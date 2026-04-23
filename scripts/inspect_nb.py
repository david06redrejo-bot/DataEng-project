import json
from pathlib import Path

nb = json.loads(Path(r"C:\Users\david\Downloads\Labs 1 - Pandas (1)\PROJECT\notebooks\01_exploratory_data_analysis.ipynb").read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "def recommend" in src or ("recommend(" in src and "def " not in src):
        print(f"--- Cell {i} ---")
        print(src)
        print()
