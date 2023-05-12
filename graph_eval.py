
import pysnooper
from pprint import pprint
import argparse
import matplotlib.pyplot as plt


def graph(eval_file):

    lines = [l[:-1] for l in open(eval_file)]
    grouped_lines = [lines[x:x+9] for x in range(0, len(lines), 9)]
    parsed_groups = [parse_group(gl) for gl in grouped_lines]

    losses = ["loss", "masked_lm_loss", "next_sentence_loss"]
    accuracy = ["masked_lm_accuracy", "next_sentence_accuracy"]

    for a in accuracy:
        coords = [(int(pg["global_step"]), float(pg[a])) for pg in parsed_groups]
        coords = sorted(coords, key=lambda x: x[0])
        x, y = zip(*coords)
        plt.plot(x, y, label=a)
    plt.legend()
    save_file = f"{eval_file[:-4]}_accuracy.png"
    plt.savefig(save_file)
    plt.clf()

    for l in losses:
        coords = [(int(pg["global_step"]), float(pg[l])) for pg in parsed_groups]
        coords = sorted(coords, key=lambda x: x[0])
        x, y = zip(*coords)
        plt.plot(x, y, label=l)
    plt.legend()
    save_file = f"{eval_file[:-4]}_loss.png"
    plt.savefig(save_file)
    plt.clf()

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, action="store",
                        help='txt file where eval results are saved')
    args = parser.parse_args()

    graph(args.eval_file)