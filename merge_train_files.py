#command to run: python3 merge_train_files.py

final_res = set()

fn1 = 'pretrain_hd_train_hi_cm.tsv'
fn2 = 'pretrain_hd_dev_hi_cm.tsv'
fn3 = 'finetune_hd_train_hi_cm.tsv'

for fn in [fn1, fn2, fn3]:
    with open(fn, 'r') as f:
        for l in f:
            l = l.strip()
            x = l.split('\t')
            final_res.add((x[0], x[1]))

print(len(final_res))
fn4 = 'cmi_control_train.tsv'
of = open(fn4, 'w')
for x, y in final_res:
    of.write(x+'\t'+y+'\n')