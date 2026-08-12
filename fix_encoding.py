import os
import re

files = [
    'src/data_loader.py',
    'src/preprocessing.py', 
    'src/segmentation.py',
    'src/prediction.py',
    'src/explainability.py',
    'src/recommendation.py',
    'src/config.py',
    'src/train_model.py',
]

replacements = {
    '\u2192': '->',
    '\u2190': '<-',
    '\u2713': 'OK',
    '\u2714': 'OK',
    '\u2705': '[OK]',
    '\u26a0': '[WARN]',
    '\u274c': '[ERR]',
    '\u00d7': 'x',
    '\u00e9': 'e',
    '\u2014': '--',
    '\u2013': '-',
}

for fname in files:
    if not os.path.exists(fname):
        continue
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        # Only fix lines that are NOT pure comments or docstrings
        stripped = line.strip()
        is_comment = stripped.startswith('#')
        is_docstring = stripped.startswith('"""') or stripped.startswith("'''")
        
        if is_comment:
            # Comments: safe, just keep but replace arrows in print statements embedded in comments = fine
            new_lines.append(line)
        else:
            # Code lines: replace problematic chars that may appear in string literals
            new_line = line
            for old, new in replacements.items():
                new_line = new_line.replace(old, new)
            new_lines.append(new_line)
    
    new_content = '\n'.join(new_lines)
    if new_content != original:
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed: {fname}")
    else:
        print(f"Clean: {fname}")
        
print("Done.")
