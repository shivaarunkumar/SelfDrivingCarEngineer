from pprint import pprint

p = [.2] * 5
color = ['g', 'r', 'r', 'g', 'g']
for i, c in enumerate(color):
    if c == 'g':
        p[i] *= .2
    else:
        p[i] *= .6
pprint(p)
