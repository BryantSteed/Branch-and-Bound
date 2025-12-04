import matplotlib.pyplot as plt
import numpy as np
from tsp_solve import branch_and_bound, branch_and_bound_smart
from tsp_full_details import Timer, generate_network

seeds = list(range(20))
time_limit = 0.2
size = 10

smart_scores = []
stupid_scores = []

for seed in seeds:
    locations, edges = generate_network(
        size,
        euclidean=True,
        reduction=0.2,
        normal=False,
        seed=seed,
    )

    timer1 = Timer(time_limit)
    stats_stupid = branch_and_bound(edges, timer1)
    score_stupid = stats_stupid[-1].score if stats_stupid else float('inf')
    stupid_scores.append(score_stupid)

    timer2 = Timer(time_limit)
    stats_smart = branch_and_bound_smart(edges, timer2)
    score_smart = stats_smart[-1].score if stats_smart else float('inf')
    smart_scores.append(score_smart)

# convert infinities to NaN so they don't distort autoscaling
stupid_plot = [np.nan if s == float('inf') else s for s in stupid_scores]
smart_plot = [np.nan if s == float('inf') else s for s in smart_scores]

x = np.arange(len(seeds))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width / 2, stupid_plot, width, label='Stupid', color='#d62728')
ax.bar(x + width / 2, smart_plot, width, label='Smart', color='#1f77b4')

ax.set_xlabel('Seed')
ax.set_ylabel('Score (lower is better)')
ax.set_title(f'Stupid vs Smart scores (n={size}, time_limit={time_limit}s)')
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in seeds])
ax.legend()
plt.tight_layout()


def gen_markdown_table(seeds, stupid_scores, smart_scores):
    header = "| Seed | Stupid Score | Smart Score |\n|------|--------------|-------------|\n"
    rows = ""
    for seed, stupid, smart in zip(seeds, stupid_scores, smart_scores):
        stupid_str = "inf" if stupid == float('inf') else f"{stupid:.2f}"
        smart_str = "inf" if smart == float('inf') else f"{smart:.2f}"
        rows += f"| {seed} | {stupid_str} | {smart_str} |\n"
    return header + rows

markdown_table = gen_markdown_table(seeds, stupid_scores, smart_scores)
with open("other_smart_stupid.md", "w") as f:
    f.write(markdown_table)