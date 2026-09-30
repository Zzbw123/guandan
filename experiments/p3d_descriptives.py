"""EDA of original-deal group win fractions; no new hypothesis tests."""
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    directory=Path(sys.argv[1]).resolve()
    rows=[json.loads(line) for line in (directory/"results.jsonl").read_text().splitlines()]
    output=directory/"descriptives";output.mkdir()
    all_values={}
    for opponent in ("greedy","random","team"):
        subset=[r for r in rows if r["matchup_id"]==f"dmc|dmc|{opponent}"]
        assert len(subset)==520 and all(r["status"]=="ok" for r in subset)
        all_values[opponent]=[sum(r["win"] for r in subset if r["deal_seed"]==seed)/8 for seed in range(201000,201065)]
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),sharex=True,sharey=True,layout="constrained")
    for ax,(opponent,values) in zip(axes,all_values.items()):
        counts=[values.count(i/8) for i in range(9)]
        ax.bar([i/8 for i in range(9)],counts,width=.09,color="#4477AA",edgecolor="white")
        ax.set_title(f"vs {opponent}",loc="left",fontweight="bold")
        ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlim(-.08,1.08);ax.set_ylim(0,65)
        ax.set_xlabel("Win fraction in 8 paired variants")
        ax.text(.98,.96,f"65 original deals\nMean = {sum(values)/65:.1%}",ha="right",va="top",transform=ax.transAxes,fontsize=9)
    axes[0].set_ylabel("Number of original deals")
    fig.suptitle("Fixed 400-hand GPU candidate: validation group distributions",fontsize=12,fontweight="bold")
    fig.savefig(output/"group-distributions.png",dpi=180)
    fig.savefig(output/"group-distributions.svg")
    with (output/"groups.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["opponent","deal_seed","level","games","win_fraction"])
        for opponent,values in all_values.items():
            for i,value in enumerate(values):writer.writerow([opponent,201000+i,2+i%13,8,value])
    receipt=dict(kind="descriptive EDA only; no independent-game interpretation",matplotlib=matplotlib.__version__,
                 source_results_sha256=sha256((directory/"results.jsonl").read_bytes()).hexdigest(),
                 script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                 group_counts={opponent:{str(i/8):values.count(i/8) for i in range(9)} for opponent,values in all_values.items()})
    (output/"trace.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
