final_res = set()

import sys
fn1 = sys.argv[1]
fn2 = sys.argv[2]

with open(fn1, 'r') as f:
    for l in f:
        l = l.strip()
        x = l.split('\t')
        final_res.add((x[0], x[1]))

with open(fn2, 'r') as f:
    for l in f:
        l = l.strip()
        x = l.split('\t')
        final_res.add((x[0], x[1]))

print(len(final_res))
fn3 = sys.argv[3]
of = open(fn3, 'w')
for x, y in final_res:
    of.write(x+'\t'+y+'\n')