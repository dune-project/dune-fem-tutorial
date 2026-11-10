import fileinput
from argparse import ArgumentParser
import os, re, itertools, sys

parser = ArgumentParser()
parser.add_argument('filename', type=str, help='enter tex file to add gitlab math')
args = parser.parse_args()
filename = args.filename

# these math environments are not supported - replaced with a simple
# ```math gitlab environment. All alignment tabs will cause problems
# use aligned or alignedat instead
# see https://katex.org/docs/supported.html
mathEnvs = ["gather","eqnarray","align","equation"]

count = 0
input = open(filename, 'r')
output = open('output', 'w')
lines = input.readlines()

for i in range(len(lines)):
    # remove pre-existing code highlighting
    lines[i] = lines[i].replace('[language=Python]', '')
    if "$$" in lines[i]:
        print("can not handle $$..$$ math environments yet please use"+\
              "equation environment instead... need to bail out")
        sys.exit(1)
    # add pythonstyle highlighting if not already done
    # if "\\begin{aligned" in lines[i]: # also covers {alignedat}{n}
    #     output.write("```math\n")
    #     output.write(lines[i])
    # if "\\end{aligned" in lines[i]:
    #     output.write(lines[i])
    #     output.write("```math\n")
    if any("\\begin{"+me+"}" in lines[i] for me in mathEnvs) or\
       any("\\begin{"+me+"*}" in lines[i] for me in mathEnvs):
        output.write("```math\n")
    elif any("\\end{"+me+"}" in lines[i] for me in mathEnvs) or\
         any("\\end{"+me+"*}" in lines[i] for me in mathEnvs):
        output.write("```\n")
    else:
        counter = itertools.count(1)
        output.write( re.sub('\$',
              lambda m: "$`" if next(counter)%2==1 else "`$",
              lines[i])
            )
os.rename("output", filename)
