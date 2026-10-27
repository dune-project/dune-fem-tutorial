import ast
import jupytext
import sys
script = sys.argv[1]

nb = jupytext.read(script+".py")
lines = [ cell.source for cell in nb.cells if cell.cell_type == "code" ]
lines = [ line
    for code in lines
    for line in code.splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
"""
isMainImport = lambda line: "import " in line and line==line.lstrip()
code = "\n".join([line for line in lines if isMainImport(line)])
code += "\n\n"
code += "\n".join([line for line in lines if not isMainImport(line)])
"""

source = "\n".join(lines)

tree = ast.parse(source)
import_nodes = [
    node for node in tree.body
    if isinstance(node, (ast.Import, ast.ImportFrom))
]

# Use source line ranges so multiline imports move as a unit.
import_ranges = [
    (node.lineno - 1, node.end_lineno)
    for node in import_nodes
]

import_lines = []
remaining_lines = []
line_number = 0

for start, end in import_ranges:
    remaining_lines.extend(lines[line_number:start])
    import_lines.extend(lines[start:end])
    line_number = end

remaining_lines.extend(lines[line_number:])

code = "\n".join(import_lines) + "\n\n" + "\n".join(remaining_lines)

with open(script+"_script.py","w") as f:
    f.write(code)
