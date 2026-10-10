"""Run the interactive task-state analysis headlessly and export a browsable report.

python -m analysis.eigenposture_task_report --output /path/to/results
Use --from-results to rebuild only the HTML from an existing completed run.
"""

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def run_analysis(output):
    source_path = Path(__file__).with_name("eigenposture_interactive.py")
    source = (
        "\n".join(
            line
            for line in source_path.read_text().splitlines()
            if not line.lstrip().startswith("%matplotlib")
        )
        + "\n"
    )
    namespace = {"__name__": "__main__", "__file__": str(source_path)}
    plt.close("all")
    exec(compile(source, str(source_path), "exec"), namespace)
    n = namespace
    assert n["task_input"].shape[1] == 6
    assert len(n["assignments"]) == len(n["ants"])
    np.testing.assert_array_equal(n["tasks"] >= 0, n["valid"])
    np.testing.assert_array_equal(
        n["tasks"][n["valid"]],
        n["state_remap"][n["task_model"].predict(n["task_input"])],
    )
    np.testing.assert_allclose(n["proportions"][n["ants"].eligible].sum(axis=1), 1)
    output.mkdir(parents=True, exist_ok=True)
    (output / "interactive_task_states.py").write_text(source)
    for key, filename in [
        ("task_k_table", "task_k_selection.csv"),
        ("task_summary", "task_summary.csv"),
        ("proportion_table", "ant_task_proportions.csv"),
    ]:
        n[key].to_csv(output / filename)
    n["bin_table"].to_csv(output / "five_minute_tasks.csv.gz", index=False)
    n["assignments"].to_csv(output / "ant_assignments.csv", index=False)
    pd.DataFrame(n["spatial_results"]).to_csv(
        output / "spatial_comparison.csv", index=False
    )
    for side, table in n["ant_k_tables"].items():
        table.to_csv(output / f"{side}_ant_k_selection.csv")
    np.savez_compressed(
        output / "task_states.npz",
        ants=n["ants"].ant.to_numpy(str),
        **{
            key: n[key]
            for key in (
                "binned",
                "clip_counts",
                "tasks",
                "proportions",
                "ant_groups",
                "task_means",
                "feature_center",
                "feature_scale",
            )
        },
        posture_center=n["coordinate_center"],
        posture_modes=n["coordinate_modes"],
    )
    summary = dict(
        n_identities=len(n["ants"]),
        valid_bins=int(n["valid"].sum()),
        n_task_states=n["task_k"],
        posture_variance=float(n["posture_variance"][:4].sum()),
        ant_groups={
            side: dict(
                k=m.n_components,
                n=int((n["ants"].side.eq(side) & n["ants"].eligible).sum()),
            )
            for side, m in n["ant_models"].items()
        },
        spatial=n["spatial_results"],
        bootstraps=n["BOOTSTRAPS"],
        minimum_clips=n["MIN_CLIPS"],
        minimum_ant_hours=n["MIN_ANT_HOURS"],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    with PdfPages(output / "task_states.pdf") as pdf:
        for number in plt.get_fignums():
            figure = plt.figure(number)
            figure.savefig(figures / f"figure_{number:02d}.png", dpi=150)
            pdf.savefig(figure)
    plt.close("all")


def build_html(output):
    summary = json.loads((output / "summary.json").read_text())
    ants = pd.read_csv(output / "ant_assignments.csv")
    means = pd.read_csv(output / "task_summary.csv")
    with np.load(output / "task_states.npz") as cache:
        np.testing.assert_array_equal(cache["ants"], ants.ant)
        features = np.round(cache["binned"], 3).astype(object)
        features[~np.isfinite(cache["binned"])] = None
        proportions = np.round(cache["proportions"], 4).astype(object)
        proportions[~np.isfinite(cache["proportions"])] = None
        payload = dict(
            ants=ants[["ant", "side", "ant_group", "observed_bin_hours"]].to_dict(
                "records"
            ),
            tasks=cache["tasks"].tolist(),
            counts=cache["clip_counts"].tolist(),
            features=features.tolist(),
            proportions=proportions.tolist(),
            task_means=cache["task_means"].tolist(),
        )
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    group_notes, spatial_tables = [], []
    for side, result in summary["ant_groups"].items():
        table = pd.read_csv(output / f"{side}_ant_k_selection.csv").set_index("k")
        note = f"{side.title()}: K={result['k']} among {result['n']} ants."
        if result["k"] == 3:
            note += f" K=3 improves BIC over K=2 by {table.loc[2, 'bic']-table.loc[3, 'bic']:.2f}, a small preference."
        group_notes.append(html.escape(note))
        subset = ants[ants.side.eq(side) & ants.eligible]
        spatial_tables.append(
            f"<h3>{html.escape(side.title())}</h3>"
            + pd.crosstab(subset.ant_group, subset.spatial_cluster).to_html()
        )
    page = TEMPLATE.replace("__DATA__", data)
    page = page.replace(
        "__COUNTS__",
        f"{summary['n_identities']} identities · {summary['valid_bins']:,} valid five-minute bins · {summary['n_task_states']} behavioral states",
    )
    page = page.replace("__GROUP_NOTES__", "<br>".join(group_notes))
    page = page.replace("__STATE_TABLE__", means.round(3).to_html(index=False))
    page = page.replace("__SPATIAL_TABLES__", "".join(spatial_tables))
    (output / "index.html").write_text(page)


TEMPLATE = r"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>July 24 · Five-minute behavior states</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;color:#172333;background:#fafbfc;margin:24px auto;padding:0 24px;max-width:1450px}
h1{font-size:28px;margin-bottom:6px}h2{margin-top:36px}p{max-width:1050px}.muted{color:#566475}
.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:center;margin:18px 0}select,input{font:inherit;padding:5px}
#wrap{overflow:auto;background:white;border:1px solid #d9dfe5;padding:8px}canvas{max-width:100%;display:block}
#tip{position:fixed;display:none;pointer-events:none;background:#172333;color:white;padding:10px;border-radius:5px;white-space:pre-line;font-size:13px;z-index:2;max-width:340px}
#legend{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0}.swatch{display:inline-block;width:14px;height:14px;margin-right:5px;vertical-align:middle}
img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:6px 12px;border-bottom:1px solid #d9dfe5;text-align:right}a{color:#1263a0}details{margin:18px 0}summary{cursor:pointer;font-weight:600}.tables{display:flex;gap:36px;flex-wrap:wrap}
</style>
<h1>July 24: behavior states through time</h1>
<div class="muted">__COUNTS__ · July 24 10:00 – July 26 10:00 JST</div>
<p>Each column is five minutes. Colors come from clustering four posture-PC scores and unsigned forward/lateral peak speeds across all ants. Gray bins lack sufficient observations. Ant groups are fitted afterward from each ant's task proportions. Hover over a bin to inspect it.</p>
<div class="controls"><label>Colony <select id="side"><option value="all">Both</option><option value="left">Left</option><option value="right">Right</option></select></label>
<label>Order <select id="order"><option value="group">Ant group, then activity</option><option value="identity">Ant identity</option></select></label>
<label>Find ant <input id="find" placeholder="e.g. left:004" size="16"></label>
<label><input id="eligible" type="checkbox"> Show only ants in the division-of-labor fit</label></div>
<div id="legend"></div><div id="wrap"><canvas id="timeline"></canvas></div><div id="tip"></div>
<p class="muted">Speed features are maxima of unsigned, smoothed motion samples within each bin. Each minute contains one sampled 2.5-second clip; these are not continuous five-minute recordings. At least three jointly observed clips are required per bin. Unassigned ants have less than 12 observed bin-hours over the full recording.</p>
<h2>Division of labor from task proportions</h2><p>__GROUP_NOTES__</p>
<p>The current groups mainly describe low, intermediate, and high use of the faster state. The preference for three groups is modest; it does not establish three biological castes. Task-state selection considers K=2–10, so its two-state result is a descriptive resolution, not proof of exactly two biological tasks.</p>
<img src="figures/figure_06.png" alt="Per-ant task proportions, ordered by ant group">
<details><summary>How the behavioral states were formed</summary><p>Log-transformed unsigned speeds and linear posture-PC scores are balanced by feature family. K-means uses all six dimensions; silhouette chooses the state resolution. Biological task names require further observation.</p>
<img src="figures/figure_02.png" alt="Task-state resolution and feature centroids">__STATE_TABLE__</details>
<details><summary>Ant-group number and stability</summary><p>Shared-covariance Gaussian mixtures use all square-root task proportions. K=1 is allowed. Accepted splits must improve BIC, contain at least five ants per group, and pass ant-bootstrap and disjoint-time stability thresholds. Stability conditions on the pooled task dictionary.</p><img src="figures/figure_04.png" alt="Ant-group BIC and rejected candidates"></details>
<details><summary>Spatial correspondence, inspected after activity clustering</summary><p>These tables compare three activity groups with two spatial classes. No spatial label selected either clustering or merged the activity groups.</p><div class="tables">__SPATIAL_TABLES__</div>
<img src="figures/figure_07.png" alt="Left colony occupancy"><img src="figures/figure_08.png" alt="Right colony occupancy"></details>
<p><a href="task_states.pdf">All plots (PDF)</a> · <a href="interactive_task_states.py">Interactive Python script</a> · <a href="five_minute_tasks.csv.gz">Every five-minute assignment (CSV.gz)</a> · <a href="ant_task_proportions.csv">Ant task proportions</a> · <a href="README.md">Methods and results</a></p>
<script type="application/json" id="data">__DATA__</script>
<script>
const d=JSON.parse(document.getElementById('data').textContent);
const colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
const canvas=document.getElementById('timeline'),ctx=canvas.getContext('2d'),tip=document.getElementById('tip');
const width=1390,left=165,plotTop=40,rowHeight=18,cellWidth=2;let rows=[],height=0;
const label=document.getElementById('legend');
d.task_means.forEach((v,i)=>{const span=document.createElement('span');span.innerHTML='<i class="swatch" style="background:'+colors[i]+'"></i>T'+i+' · mean forward peak '+v[0].toFixed(2)+' mm/s';label.appendChild(span)});
const missing=document.createElement('span');missing.innerHTML='<i class="swatch" style="background:#ddd"></i>Missing';label.appendChild(missing);
function expected(i){return d.proportions[i].reduce((s,p,k)=>s+(p||0)*d.task_means[k][0],0)}
function draw(){
const side=document.getElementById('side').value,order=document.getElementById('order').value,find=document.getElementById('find').value.trim().toLowerCase(),eligible=document.getElementById('eligible').checked;
rows=d.ants.map((a,i)=>i).filter(i=>(side==='all'||d.ants[i].side===side)&&d.ants[i].ant.toLowerCase().includes(find)&&(!eligible||d.ants[i].ant_group>=0));
rows.sort((i,j)=>d.ants[i].side.localeCompare(d.ants[j].side)||(order==='group'?((d.ants[i].ant_group<0?99:d.ants[i].ant_group)-(d.ants[j].ant_group<0?99:d.ants[j].ant_group)||expected(i)-expected(j)):0)||d.ants[i].ant.localeCompare(d.ants[j].ant));
height=plotTop+Math.max(rows.length,1)*rowHeight+18;const ratio=window.devicePixelRatio||1;
canvas.width=width*ratio;canvas.height=height*ratio;canvas.style.width=width+'px';canvas.style.height='auto';ctx.scale(ratio,ratio);
ctx.fillStyle='white';ctx.fillRect(0,0,width,height);ctx.font='12px system-ui';ctx.fillStyle='#172333';ctx.textAlign='center';
['Jul24 10:00','Jul24 22:00','Jul25 10:00','Jul25 22:00','Jul26 10:00'].forEach((s,k)=>ctx.fillText(s,left+k*144*cellWidth,24));
rows.forEach((i,r)=>{const a=d.ants[i],y=plotTop+r*rowHeight;ctx.fillStyle='#172333';ctx.textAlign='right';ctx.fillText(a.ant+(a.ant_group>=0?' G'+a.ant_group:' (unassigned)'),left-8,y+13);
d.tasks[i].forEach((k,b)=>{ctx.fillStyle=k<0?'#ddd':colors[k];ctx.fillRect(left+b*cellWidth,y,cellWidth,rowHeight)});
if(r>0&&(a.side!==d.ants[rows[r-1]].side||(order==='group'&&a.ant_group!==d.ants[rows[r-1]].ant_group))){ctx.strokeStyle='#172333';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+576*cellWidth,y);ctx.stroke()}});
ctx.strokeStyle='#172333';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(left+288*cellWidth,plotTop);ctx.lineTo(left+288*cellWidth,height-18);ctx.stroke();ctx.setLineDash([]);
if(!rows.length){ctx.textAlign='left';ctx.fillText('No matching ants',left,plotTop+14)}tip.style.display='none';
}
for(const id of ['side','order','find','eligible'])document.getElementById(id).addEventListener('input',draw);
const clock=new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Tokyo',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
canvas.addEventListener('mousemove',event=>{const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*width/rect.width,y=(event.clientY-rect.top)*height/rect.height,b=Math.floor((x-left)/cellWidth),r=Math.floor((y-plotTop)/rowHeight);
if(b<0||b>=576||r<0||r>=rows.length){tip.style.display='none';return}const i=rows[r],a=d.ants[i],k=d.tasks[i][b],v=d.features[i][b];
let text=a.ant+' · '+(a.ant_group<0?'unassigned':'ant group G'+a.ant_group)+'\n'+clock.format(new Date(Date.UTC(2026,6,24,1)+b*300000))+' JST\n'+(k<0?'Missing bin':'Task T'+k)+' · '+d.counts[i][b]+'/5 valid clips';
if(k>=0)text+='\nForward peak '+v[0].toFixed(3)+' mm/s; lateral peak '+v[1].toFixed(3)+' mm/s\nPC1–4: '+v.slice(2).map(x=>x.toFixed(3)).join(', ');
tip.textContent=text;tip.style.display='block';tip.style.left=Math.min(event.clientX+15,window.innerWidth-360)+'px';tip.style.top=Math.min(event.clientY+15,window.innerHeight-165)+'px';});
canvas.addEventListener('mouseleave',()=>tip.style.display='none');draw();
</script></html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-results", action="store_true")
    args = parser.parse_args()
    if not args.from_results:
        run_analysis(args.output)
    build_html(args.output)
