import fileinput
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('filename', type=str, help='enter tex file to add python highlighting')
args = parser.parse_args()
filename = args.filename

count = 0
input = open(filename, 'r')
output = open('output', 'w')
lines = input.readlines()
for i in range(len(lines)):
    # remove pre-existing code highlighting
    lines[i] = lines[i].replace('[language=Python]', '')
    # add pythonstyle highlighting if not already done
    if not 'style' in lines[i]:
        lines[i] = lines[i].replace('begin{lstlisting}', 'begin{lstlisting}[style=pythonstyle]')
    # rescale images to fit pagewidth
    if filename != 'laplace-la.tex':
        lines[i] = lines[i].replace('includegraphics{', 'includegraphics[width=\\textwidth, height=7cm, keepaspectratio]{')
    elif filename == 'laplace-la.tex':
        lines[i] = lines[i].replace('includegraphics{', 'includegraphics[width=0.5\\textwidth]{')
    # add captions to images
    if filename == 'battery.tex' and lines[i].startswith('\includegraphics'):
        if count == 0:
            output.write(lines[i])
        elif count == 1:
            output.write(lines[i])
            output.write('\caption{The initial plot of $c$ and $\phi$}\n')
        elif count ==2:
            output.write(lines[i])
            output.write('\caption{The plot after the final timestep}\n')
        count += 1
    elif filename == 'mcf.tex' and lines[i].startswith('\includegraphics'):
        if count == 0:
            output.write(lines[i])
            output.write('\caption{The plot of the surface at three different timesteps}\n')
        elif count == 1:
            output.write(lines[i])
            output.write('\caption{Comparison of the error over time for varying levels of refinement}\n')
        count += 1
    elif filename == 'crystal.tex' and lines[i].startswith('\includegraphics'):
        if count == 0:
            output.write(lines[i])
            output.write('\caption{The initial adapted grid and phase field}\n')
        elif count == 1:
            output.write(lines[i])
            output.write('\caption{The grid, phase field and temperature after the final timestep}\n')
        count += 1
    elif filename == 'laplace-adaptive.tex' and lines[i].startswith('\includegraphics'):
        if count == 0:
            output.write(lines[i])
            output.write('\caption{Three plots of the solution on the locally adapted grid}\n')
        elif count == 1:
            output.write(lines[i])
            output.write('\caption{Zoom into the corner singularity (left to right). Top row shows solution and the bottom row the refinement level of each cell}\n')
        elif count == 2:
            output.write(lines[i])
            output.write('\caption{Error on sequence globally refined and adaptive grid}\n')
        elif count == 3:
            output.write(lines[i])
            output.write('\caption{Zooming in on the re-entrant corner}\n')
        elif count == 4:
            output.write(lines[i])
            output.write('\caption{Plot of the level function of the grid}\n')
        count += 1
    elif filename == 'elasticity.tex' and lines[i].startswith('\includegraphics'):
        if count == 0:
            output.write(lines[i])
            output.write('\caption{The magnitude of the displacement field (left) and stress (right)}\n')
        elif count == 1:
            output.write(lines[i])
            output.write('\caption{The displaced beam}\n')
        count += 1
    elif filename == 'wave.tex' and lines[i].startswith('\includegraphics'):
        if count == 0:
            output.write(lines[i])
            output.write('\caption{Wave amplitude at the final time}\n')
        count += 1
    # remove empty captions
    elif 'caption{svg}' in lines[i]:
        pass
    elif lines[i] == '\n':
        # remove empty lines either side of equations
        equations = ('equation', 'eqnarray', 'gather', 'align')
        try:
            if any(eqn in lines[i-1] for eqn in equations):
                pass
            elif any(eqn in lines[i+1] for eqn in equations):
                pass
            else:
                output.write(lines[i])
        except:
            pass
    else:
        output.write(lines[i])
os.rename("output", filename)

print('added lstlisting tags')
