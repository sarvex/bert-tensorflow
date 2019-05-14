
#%%
import pysnooper
from pprint import pprint

#%%
EVAL_FILE = "eval_results.txt"
lines = [l[:-1] for l in open(EVAL_FILE)]
grouped_lines = [lines[x:x+9] for x in range(0, len(lines), 9)]
pprint(grouped_lines)

#%%
def parse_group(group):
    stats = {}
    for line in group:
        try:
            field, value = line.split("=")
            stats[field.strip()] = value
        except ValueError:
            continue
    stats["global_step"] = int(group[0])
    return stats


#%%
parsed_groups = [parse_group(gl) for gl in grouped_lines]
pprint(parsed_groups)

#%%
import matplotlib.pyplot as plt
losses = ["loss", "masked_lm_loss", "next_sentence_loss"]
accuracy = ["masked_lm_accuracy", "next_sentence_accuracy"]

for a in accuracy:
    coords = [(int(pg["global_step"]), float(pg[a])) for pg in parsed_groups]
    coords = sorted(coords, key=lambda x: x[0])
    x, y = zip(*coords)
    print(x)
    print(y)
    plt.plot(x, y, label=a)
plt.legend()
plt.show()
plt.clf()

for l in losses:
    coords = [(int(pg["global_step"]), float(pg[l])) for pg in parsed_groups]
    coords = sorted(coords, key=lambda x: x[0])
    x, y = zip(*coords)
    print(x)
    print(y)
    plt.plot(x, y, label=l)
plt.legend()
plt.show()
plt.clf()




#%%


#%%
