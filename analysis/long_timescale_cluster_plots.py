"""Static and interactive views of fixed-reference occupancy assignments."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex
import numpy as np
import pandas as pd

COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]


def color(label):
    return COLORS[int(str(label).rsplit("_", 1)[1]) % len(COLORS)]


def number(value):
    return f"{value:.2g}" if pd.notna(value) else "NA"


def short(block):
    p = Path(block)
    return p.parent.name[4:6] + "/" + p.parent.name[6:8] + " " + p.name.replace("block", "b")


def save(fig, output, name):
    fig.savefig(Path(output) / (name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fixed_map(tables, ref_index, output):
    ref = tables["reference"].query("reference_index == @ref_index")
    ass = tables["assignments"].query("reference_index == @ref_index")
    for side in ("left", "right"):
        fig, ax = plt.subplots(figsize=(12, 7))
        r, a = ref[ref.side.eq(side)], ass[ass.side.eq(side)]
        for label, part in r.groupby("original_cluster"):
            ax.scatter(part.umap_x, part.umap_y, c=color(label), label=f"{label} (n={len(part)})", s=55, alpha=.7)
            ax.text(part.umap_x.mean(), part.umap_y.mean(), label, fontsize=11, weight="bold")
        examples = a[a.example_switch].sort_values(["vote", "distance_ratio"], ascending=[False, True]).drop_duplicates("ant").head(3)
        for row in examples.itertuples():
            base = r[r.ant.eq(row.ant)].iloc[0]
            ax.annotate("", xy=(row.umap_x, row.umap_y), xytext=(base.umap_x, base.umap_y), arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
            ax.scatter(row.umap_x, row.umap_y, c=color(row.prediction), marker="D", s=70, edgecolors="k")
            ax.annotate(f"{row.ant}\n{short(row.source_block)}", (row.umap_x, row.umap_y), xytext=(5, 6), textcoords="offset points", fontsize=8)
        ax.set(title=f"{side}: {short(r.source_block.iloc[0])} reference", xlabel="Reference UMAP 1", ylabel="Reference UMAP 2")
        ax.legend(fontsize=8, loc="best")
        fig.suptitle(f"{side.capitalize()} colony\n" + "Fixed reference occupancy space — supported, well-observed example changes", fontsize=14)
        fig.text(.5, .005, "Circles: original labeled ants. Diamonds: later KNN projections. Labels and distances use full grid features; arrows do not show a continuous path.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        save(fig, output, f"reference_{ref_index}_{side}_fixed_space")


def switching_matrix(tables, ref_index, output):
    ref = tables["reference"].query("reference_index == @ref_index")
    ass = tables["assignments"].query("reference_index == @ref_index")
    for side in ("left", "right"):
        fig, ax = plt.subplots(figsize=(10, 12))
        r = ref[ref.side.eq(side)].sort_values(["original_cluster", "ant"])
        a = ass[ass.side.eq(side) & ass.reference_cluster.notna()]
        blocks = sorted(a.block_index.unique())
        labels = sorted(r.original_cluster.unique())
        codes = {label: i for i, label in enumerate(labels)}
        values = np.full((len(r), len(blocks)+1), np.nan)
        values[:, 0] = r.original_cluster.map(codes)
        lookup = {(row.ant, row.block_index): row for row in a.itertuples()}
        for i, ant in enumerate(r.ant):
            for j, block in enumerate(blocks, 1):
                row = lookup.get((ant, block))
                if row is not None and pd.notna(row.prediction):
                    values[i, j] = codes[row.prediction]
                    if row.switch:
                        ax.text(j, i, "●" if row.example_switch else "○", ha="center", va="center", fontsize=10, color="black")
        ax.imshow(values, cmap=ListedColormap([color(label) for label in labels]), vmin=-.5, vmax=len(labels)-.5, aspect="auto", alpha=.7)
        ax.set_yticks(range(len(r)), r.ant, fontsize=7)
        names = [short(r.source_block.iloc[0])] + [short(a.loc[a.block_index.eq(block), "source_block"].iloc[0]) for block in blocks]
        ax.set_xticks(range(len(names)), names, rotation=22, ha="right", fontsize=9)
        ax.set_title(side + ": every originally labeled ant")
        for label in labels:
            ax.plot([], [], "s", color=color(label), label=label)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.09), ncol=3, fontsize=8)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Same-ant cluster assignments over successive recorded blocks", fontsize=14)
        fig.text(.5, .017, "● supported switch with ≥40% detection and ≥1 h detected at both endpoints; ○ other label change; white = no assignment.\nColumns are separate observed windows; gaps and unequal durations are not represented by column width.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .96))
        save(fig, output, f"reference_{ref_index}_{side}_all_ant_assignments")


def transitions(tables, ref_index, output):
    ass = tables["assignments"].query("reference_index == @ref_index and reference_cluster.notna() and prediction.notna()")
    ref = tables["reference"].query("reference_index == @ref_index")
    blocks = sorted(ass.block_index.unique())
    for i, side in enumerate(("left", "right")):
        fig, axes = plt.subplots(1, len(blocks), figsize=(5*len(blocks), 5), squeeze=False)
        labels = sorted(ref.loc[ref.side.eq(side), "original_cluster"].unique())
        for j, block in enumerate(blocks):
            part = ass[ass.side.eq(side) & ass.block_index.eq(block)]
            counts = pd.crosstab(part.reference_cluster, part.prediction).reindex(index=labels, columns=labels, fill_value=0)
            supported = pd.crosstab(part.loc[part.example_switch, "reference_cluster"], part.loc[part.example_switch, "prediction"]).reindex(index=labels, columns=labels, fill_value=0)
            ax = axes[0, j]
            ax.imshow(counts, cmap="Blues", vmin=0)
            for y in range(len(labels)):
                for x in range(len(labels)):
                    n, s = int(counts.iloc[y, x]), int(supported.iloc[y, x])
                    ax.text(x, y, str(n) + (f"\n({s} supported)" if x != y else ""), ha="center", va="center", fontsize=10,
                            color="white" if n > counts.to_numpy().max()*.65 else "black")
            ax.set_xticks(range(len(labels)), labels, rotation=25)
            ax.set_yticks(range(len(labels)), labels)
            ax.set(title=f"{side}: {short(part.source_block.iloc[0])}\n{len(part)} same ants", xlabel="Later KNN assignment", ylabel="Original cluster")
        fig.suptitle(f"{side.capitalize()} colony\n" + "Transition counts relative to the fixed reference", fontsize=14)
        fig.text(.5, .005, "All assignable same-ant profiles counted. Parentheses: switches passing neighbor, baseline, distance and endpoint observation checks.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        save(fig, output, f"reference_{ref_index}_{side}_transitions")


def diagnostics(tables, ref_index, output):
    ass = tables["assignments"].query("reference_index == @ref_index and reference_cluster.notna()")
    cal = tables["calibration"].query("reference_index == @ref_index")
    for i, side in enumerate(("left", "right")):
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), squeeze=False)
        part = ass[ass.side.eq(side)]
        ax = axes[0, 0]
        for switched, label, marker in [(False, "Unchanged label", "o"), (True, "Changed label", "D")]:
            rows = part[part.switch.eq(switched)]
            ax.scatter(rows.distance_ratio, rows.vote, c=rows.coverage, vmin=0, vmax=1, cmap="viridis", marker=marker, alpha=.7, label=label)
        strong = part[part.example_switch]
        ax.scatter(strong.distance_ratio, strong.vote, s=120, facecolors="none", edgecolors="black", label="Supported + observed")
        ax.axvline(1, c="gray", ls="--"); ax.axhline(.8, c="gray", ls="--")
        ax.set(title=side, xlabel="Mean KNN distance / reference LOO 95th percentile", ylabel="Distance-weighted neighbor vote", ylim=(-.02, 1.04))
        ax.legend(fontsize=8)
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"), ax=ax, label="Later detected-frame fraction")
        ax = axes[1, 0]
        for label, rows in cal[cal.side.eq(side)].groupby("cluster"):
            ax.plot(rows.k, rows.loo_accuracy, "o-", c=color(label), label=f"{label} (n={rows.n_reference.iloc[0]})")
        ax.set(ylim=(0, 1.04), xlabel="Number of neighbors", ylabel="Leave-one-ant-out label recovery", title="How well KNN recovers original labels")
        ax.set_xticks([3, 5, 10]); ax.legend(fontsize=8)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Assignment support and reference calibration", fontsize=14)
        fig.text(.5, .005, "Neighbor vote is not a calibrated probability. Reference distance cutoff is a descriptive novelty flag, not a significance test.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        save(fig, output, f"reference_{ref_index}_{side}_diagnostics")


def examples(tables, maps, edges, ref_index, output):
    ref = tables["reference"].query("reference_index == @ref_index")
    ass = tables["assignments"].query("reference_index == @ref_index")
    for side in ("left", "right"):
        part = ass[ass.side.eq(side)]
        picked = part[part.example_switch].sort_values(["vote", "distance_ratio"], ascending=[False, True]).drop_duplicates("ant").head(3)
        stable = part[~part.switch & part.supported & part.well_observed].sort_values("vote", ascending=False).head(1)
        picked = pd.concat([picked, stable])
        if picked.empty:
            continue
        fig, axes = plt.subplots(len(picked), 4, figsize=(15, 4.1*len(picked)), squeeze=False)
        for i, row in enumerate(picked.itertuples()):
            base = ref[ref.ant.eq(row.ant)].iloc[0]
            old = ref[ref.original_cluster.eq(base.original_cluster)]
            new = ref[ref.original_cluster.eq(row.prediction)]
            histograms = [maps[base.profile_key], maps[row.profile_key],
                          np.mean([maps[key] for key in old.profile_key], axis=0),
                          np.mean([maps[key] for key in new.profile_key], axis=0)]
            transformed = [np.sqrt(h) for h in histograms]
            vmax = max(float(x.max()) for x in transformed)
            titles = [f"{row.ant}: original {base.original_cluster}\n{short(base.source_block)}; detected {base.coverage:.0%}",
                      f"Later {row.prediction}; vote {row.vote:.0%}\n{short(row.source_block)}; detected {row.coverage:.0%}",
                      f"Original cluster prototype\n{base.original_cluster} (n={len(old)})",
                      f"Assigned cluster prototype\n{row.prediction} (n={len(new)})"]
            x, y = edges[side]
            for j, (hist, title) in enumerate(zip(transformed, titles)):
                im = axes[i, j].imshow(hist, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], vmin=0, vmax=vmax, cmap="magma", aspect="equal")
                axes[i, j].set(title=title, xlabel="Arena x (mm)", ylabel="Arena y (mm)")
                fig.colorbar(im, ax=axes[i, j], fraction=.03, pad=.02, label="√ occupancy")
            axes[i, 0].text(0, -.24, ("Supported switch" if row.switch else "Stable control") +
                             f"; colony {base.colony_percent:.1f}% → {row.colony_percent:.1f}%; trip mean {number(base.mean_trip_minutes)} → {number(row.mean_trip_minutes)} min", transform=axes[i, 0].transAxes, fontsize=9)
        fig.suptitle(f"{side}: example ant maps and fixed reference prototypes", fontsize=14)
        fig.tight_layout(rect=(0, .03, 1, .96))
        save(fig, output, f"reference_{ref_index}_{side}_example_maps")


def behavior_check(tables, manifest, ref_index, output):
    data = tables["clock_matched_behavior"].query("reference_index == @ref_index")
    picked = tables["examples"].query("reference_index == @ref_index").drop_duplicates("ant").head(6)
    if picked.empty:
        return
    for side, side_picked in picked.groupby("side"):
        fig, axes = plt.subplots(len(side_picked), 2, figsize=(13, 3.8*len(side_picked)), squeeze=False)
        for i, row in enumerate(side_picked.itertuples()):
            raw = tables["profiles"][tables["profiles"].ant.eq(row.ant) & tables["profiles"].block_index.ge(ref_index)].sort_values("block_index")
            for j, (metric, label) in enumerate([("colony_percent", "Inside colony (%)"), ("mean_speed_mm_s", "Mean speed (mm/s)")]):
                matched = data[data.ant.eq(row.ant) & data.metric.eq(metric)].sort_values("block_index")
                ax = axes[i, j]
                ax.plot(raw.block_index, raw[metric], "o--", c="0.6", label="Whole observed block")
                ax.plot(matched.block_index, matched.value, "o-", c=color(row.prediction), label="Equal weight to shared clock slots")
                ax.set_xticks(raw.block_index, [short(manifest["windows"][b]["block"]) for b in raw.block_index], rotation=10)
                ax.set(ylabel=label, title=f"{row.ant}; {matched.shared_clock_hours.iloc[0]:g} shared clock hours" if len(matched) else row.ant)
                ax.legend(fontsize=8)
                if metric == "colony_percent": ax.set_ylim(0, 103)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Do example changes persist at matching times of day?", fontsize=14)
        fig.text(.5, .005, "Existing behavior bins only; no refitting or clock matching of the KNN occupancy maps. Matching is per ant/metric; observation weights combine repeated cycles.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .95))
        save(fig, output, f"reference_{ref_index}_{side}_example_clock_check")


def save_figures(tables, maps, edges, manifest, output):
    for index in sorted(tables["reference"].reference_index.unique()):
        fixed_map(tables, index, output)
        switching_matrix(tables, index, output)
        transitions(tables, index, output)
        diagnostics(tables, index, output)
        examples(tables, maps, edges, index, output)
        behavior_check(tables, manifest, index, output)


def records(table):
    return json.loads(table.to_json(orient="records"))


def write_explorer(tables, maps, edges, manifest, output):
    from plotly.offline import get_plotlyjs
    data = dict(reference=records(tables["reference"]), assignments=records(tables["assignments"]),
                profiles=records(tables["profiles"]), matched=records(tables["clock_matched_behavior"]), windows=manifest["windows"], colors=COLORS,
                magma=[[float(t), to_hex(plt.get_cmap("magma")(t))] for t in np.linspace(0, 1, 20)],
                maps={key: np.round(value, 8).tolist() for key, value in maps.items()},
                edges={side: [a.tolist() for a in pair] for side, pair in edges.items()})
    page = '''<!doctype html><html><head><meta charset="utf-8"><title>Ant occupancy cluster switching</title>
<style>body{font:15px system-ui;margin:24px;color:#263238;background:#fafbfc}select{padding:8px;margin:6px}h1{font-size:26px}.note{max-width:1100px;line-height:1.5}.plots{background:white;margin:18px 0}#space{height:580px}#maps{height:540px}#metrics{height:520px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ccd}#error{color:#b00}a{color:#246}</style>
<script>__PLOTLY__</script></head><body><a href="index.html">Report and saved figures</a>
<h1>Does the same ant change its spatial occupancy cluster?</h1>
<p class="note">Reference labels and √ occupancy features are fixed. KNN uses Euclidean distances in the full grid, with 5 inverse-distance-weighted neighbors. The 2D display uses a reference-only UMAP and KNN-weighted projection for later maps. Projection stays inside the reference map even for novel profiles: consult distance and support below.</p>
<label>Reference <select id="reference"></select></label><label>Colony <select id="side"><option>left</option><option>right</option></select></label>
<label>Ant <select id="ant"></select></label><p id="status"></p><div id="space" class="plots"></div>
<div id="table"></div><div id="maps" class="plots"></div>
<p class="note">Behavior panels: gray lines show whole-block means. Orange dashed lines compare colony use and speed at shared clock times. Trip metrics are NA when the existing optional trip analysis did not include this ant; NA does not mean zero trips.</p><div id="metrics" class="plots"></div>
<p class="note">Maps average whole recorded blocks of unequal durations and clock coverage; see the actual intervals above. A recording directory date can differ from its tracked window's dates. Spatial cluster switching is a candidate behavior change, not a demonstrated task reassignment. Side + tag ID defines identity. Missing observations and unsampled gaps do not imply inactivity.</p><p id="error"></p>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const col=label=>D.colors[Number(label.split('_').pop())%D.colors.length];
const name=b=>b.split('/').slice(-2).join('/');
const pct=x=>x==null?'NA':(100*x).toFixed(0)+'%';
const number=x=>x==null?'NA':x.toFixed(2);
for(const index of [...new Set(D.reference.map(r=>r.reference_index))]) $('reference').add(new Option(name(D.windows[index].block),index));
function ants(){const old=$('ant').value;const ref=Number($('reference').value),side=$('side').value;const rows=D.reference.filter(r=>r.reference_index===ref&&r.side===side);$('ant').replaceChildren();for(const r of rows) $('ant').add(new Option(r.ant+' · '+r.original_cluster,r.ant));if(rows.some(r=>r.ant===old))$('ant').value=old;else {const example=D.assignments.find(r=>r.reference_index===ref&&r.side===side&&r.example_switch);if(example)$('ant').value=example.ant;}}
async function render(){window.__READY__=false;try{
const ref=Number($('reference').value),side=$('side').value,ant=$('ant').value;
const R=D.reference.filter(r=>r.reference_index===ref&&r.side===side),base=R.find(r=>r.ant===ant);
const A=D.assignments.filter(r=>r.reference_index===ref&&r.ant===ant).sort((a,b)=>a.block_index-b.block_index);
const points=[{...base,prediction:base.original_cluster,vote:base.loo_vote},...A];
const traces=[];for(const label of [...new Set(R.map(r=>r.original_cluster))]){const p=R.filter(r=>r.original_cluster===label);traces.push({x:p.map(r=>r.umap_x),y:p.map(r=>r.umap_y),text:p.map(r=>r.ant),type:'scatter',mode:'markers',marker:{color:col(label),size:10,opacity:.5},name:label,hovertemplate:'%{text}<extra>'+label+'</extra>'});}
traces.push({x:points.map(r=>r.umap_x),y:points.map(r=>r.umap_y),text:points.map(r=>name(r.source_block)),type:'scatter',mode:'lines+markers+text',textposition:'top center',line:{color:'#333',dash:'dot'},marker:{size:14,color:points.map(r=>r.prediction?col(r.prediction):'#aaa'),symbol:points.map((r,i)=>i?'diamond':'star'),line:{color:'black',width:1}},name:ant,connectgaps:false,hovertemplate:'%{text}<extra>'+ant+'</extra>'});
await Plotly.react('space',traces,{title:{text:ant+' in fixed '+name(base.source_block)+' space'},xaxis:{title:{text:'Reference UMAP 1'}},yaxis:{title:{text:'Reference UMAP 2'}},margin:{t:65},legend:{orientation:'h'},uirevision:'space-'+ref+'-'+side},{responsive:true});
$('status').textContent=ant+': original '+base.original_cluster+'. Baseline leave-one-out recovery: '+base.loo_prediction+' ('+pct(base.loo_vote)+' vote). '+A.filter(r=>r.switch).length+' later label changes, '+A.filter(r=>r.example_switch).length+' supported and well-observed.';
let rows='<table><tr><th>Observed window</th><th>Assignment</th><th>Detected</th><th>Detected hours</th><th>Vote</th><th>Distance / reference P95</th><th>k=3/5/10 agree</th><th>Excluding own baseline</th><th>Switch evidence</th></tr>';
for(const r of points){const isBase=r.block_index===ref;rows+='<tr><td>'+name(r.source_block)+'<br>'+r.start+' → '+r.stop+'</td><td>'+(r.prediction??'NA')+'</td><td>'+pct(r.coverage)+'</td><td>'+number(r.detected_hours)+'</td><td>'+pct(r.vote)+'</td><td>'+(isBase?'reference':number(r.distance_ratio))+'</td><td>'+(isBase?'LOO: '+r.loo_k_agreement:r.k_agreement)+'</td><td>'+(isBase?r.loo_prediction:r.without_own_prediction)+'</td><td>'+(isBase?'original label':r.example_switch?'supported + observed':r.supported_switch?'supported; limited observation':r.switch?'candidate only':'no label change')+'</td></tr>';}
$('table').innerHTML=rows+'</table>';
const count=points.length,imgs=[],layout={title:{text:'Original and later cached occupancy — same color scale',y:.98,yanchor:'top'},annotations:[],margin:{t:110,b:55},height:540};
const hist=points.map(r=>D.maps[r.profile_key].map(line=>line.map(Math.sqrt)));let vmax=0;for(const m of hist)for(const row of m)for(const x of row)vmax=Math.max(vmax,x);
const xe=D.edges[side][0],ye=D.edges[side][1],x=xe.slice(1).map((v,i)=>(v+xe[i])/2),y=ye.slice(1).map((v,i)=>(v+ye[i])/2);
for(let i=0;i<count;i++){const suffix=i?String(i+1):'',start=i/count+.015,stop=(i+1)/count-.035;layout['xaxis'+suffix]={domain:[start,stop],title:{text:'Arena x (mm)'},range:[xe[0],xe.at(-1)],constrain:'domain'};layout['yaxis'+suffix]={anchor:'x'+suffix,title:{text:i?'':'Arena y (mm)'},range:[ye[0],ye.at(-1)],scaleanchor:'x'+suffix,scaleratio:1};imgs.push({type:'heatmap',z:hist[i],x,y,xaxis:'x'+suffix,yaxis:'y'+suffix,zmin:0,zmax:vmax,colorscale:D.magma,showscale:i===count-1,colorbar:{title:{text:'√ occupancy'},len:.9},hovertemplate:'x=%{x} mm<br>y=%{y} mm<br>√ occupancy=%{z:.3f}<extra></extra>'});layout.annotations.push({xref:'paper',yref:'paper',x:(start+stop)/2,y:1.04,text:name(points[i].source_block)+'<br>'+points[i].prediction,showarrow:false,xanchor:'center',yanchor:'bottom'});}
await Plotly.react('maps',imgs,layout,{responsive:true});
const specs=[['colony_percent','Inside colony (%)'],['mean_speed_mm_s','Mean speed (mm/s)'],['mean_trip_minutes','Mean completed trip (min)'],['trip_rate','Trips / observed hour']];
const mt=[],ml={title:{text:'Behavior summaries from the same recorded blocks'},grid:{rows:2,columns:2,pattern:'independent'},showlegend:false,margin:{t:55,b:70},annotations:[]};
for(let i=0;i<specs.length;i++){const [metric,title]=specs[i],suffix=i?String(i+1):'';mt.push({x:points.map(r=>name(r.source_block)),y:points.map(r=>r[metric]),type:'scatter',mode:'lines+markers',connectgaps:false,xaxis:'x'+suffix,yaxis:'y'+suffix,marker:{color:points.map(r=>r.prediction?col(r.prediction):'#aaa'),size:10},line:{color:'#777'},name:'Whole observed block'});const matched=D.matched.filter(r=>r.reference_index===ref&&r.ant===ant&&r.metric===metric).sort((a,b)=>a.block_index-b.block_index);if(matched.length)mt.push({x:matched.map(r=>name(D.windows[r.block_index].block)),y:matched.map(r=>r.value),type:'scatter',mode:'lines+markers',xaxis:'x'+suffix,yaxis:'y'+suffix,line:{dash:'dash',color:'#e69f00'},name:'Shared clock slots',hovertemplate:'%{y:.2f}<extra>Clock matched ('+matched[0].shared_clock_hours+' h)</extra>'});ml['yaxis'+suffix]={title:{text:title}};ml['xaxis'+suffix]={tickangle:15,tickfont:{size:9}};}
await Plotly.react('metrics',mt,ml,{responsive:true});window.__READY__=true;
}catch(e){$('error').textContent=e.stack;window.__ERROR__=e.stack;throw e;}}
$('reference').onchange=()=>{ants();render()};$('side').onchange=()=>{ants();render()};$('ant').onchange=render;ants();render();
</script></body></html>'''
    (Path(output) / "individual_cluster_explorer.html").write_text(page.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))


def write_report(tables, manifest, output):
    output = Path(output)
    ref_names = ", ".join(short(manifest["windows"][i]["block"]) for i in sorted(tables["reference"].reference_index.unique()))
    lines = ["# Fixed-reference occupancy cluster switching", "",
             f"The saved grid_occupancy cluster IDs from each requested reference block are used unchanged. Reference in this run: {ref_names}. If additional references are requested, each is an independent analysis: cluster numbers across references are not equivalent.", "",
             "KNN sees exactly the original square-root occupancy features (bin counts divided by detected frames), using Euclidean distance, separately per colony. Five inverse-distance-weighted neighbors vote on later profiles. No later data trains either the classifier or UMAP. The reference UMAP is regenerated with the original parameters; later points are neighbor-weighted barycenters. Projection cannot display novelty outside the reference map, so high-dimensional distance is reported explicitly.", "",
             "## Counts", "", "| Reference | Colony | Later block | Same ants assigned | Changed label | Supported switches | Supported + observed | Outside reference range |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in tables["transitions"].itertuples():
        lines.append(f"| {short(manifest['windows'][r.reference_index]['block'])} | {r.side} | {short(manifest['windows'][r.block_index]['block'])} | {r.same_ant_assigned} | {r.raw_switches} | {r.supported_switches} | {r.well_observed_supported_switches} | {r.outside_reference_range} |")
    lines += ["", "A supported switch requires the original label to be recovered by leave-one-ant-out KNN at k=3,5,10 with k=5 vote ≥80%; the later assignment must also have ≥80% vote, agree at k=3,5,10, survive exclusion of that ant's baseline map, and have mean 5-neighbor distance no greater than the reference leave-one-out 95th percentile. The observation check additionally requires ≥40% detected frames and ≥1 detected hour at both endpoints. These are descriptive checks, not statistical significance or calibrated probabilities. All maps and predictions remain in the audit tables, including low-coverage, ambiguous, novel and originally unclustered ants.", "",
              "## Concrete examples", "", "| Reference | Ant | Later block | Change | Vote | Distance / P95 | Detection: first → later | Colony %: first → later |", "|---|---|---|---|---:|---:|---|---|"]
    for keys, part in tables["examples"].groupby(["reference_index", "side"]):
        for r in part.drop_duplicates("ant").head(5).itertuples():
            lines.append(f"| {short(manifest['windows'][r.reference_index]['block'])} | {r.ant} | {short(r.source_block)} | {r.reference_cluster} → {r.prediction} | {r.vote:.0%} | {r.distance_ratio:.2f} | {r.baseline_coverage:.0%} → {r.coverage:.0%} | {r.baseline_colony_percent:.1f} → {r.colony_percent:.1f} |")
    if tables["examples"].empty:
        lines += ["", "No candidate passed every support and observation check. Inspect the raw transitions and diagnostics rather than interpreting the forced assignments as strong switches."]
    lines += ["", "## Shared-clock behavior check", "",
              "This supplementary check uses existing half-hour behavior summaries. Within each block, observed-frame weights combine repeated cycles at the same clock time. Each clock slot observed in all compared blocks then receives equal weight, separately per ant and metric. Partial boundary slots contribute where observed. It does not refit or clock-match the KNN grid maps.", "",
              "| Ant | Metric | Shared clock hours | Reference → successive later blocks |", "|---|---|---:|---|"]
    for r in tables["examples"].drop_duplicates(["reference_index", "ant"]).itertuples():
        for metric in ("colony_percent", "mean_speed_mm_s"):
            part = tables["clock_matched_behavior"].query("reference_index == @r.reference_index and ant == @r.ant and metric == @metric").sort_values("block_index")
            if len(part):
                lines.append(f"| {r.ant} | {metric} | {part.shared_clock_hours.iloc[0]:g} | " + " → ".join(f"{v:.2f}" for v in part.value) + " |")
    lines += ["", "## Interpretation and scope", "",
              "These are shifts among spatial occupancy patterns. Clusters have not been independently validated as biological tasks. Matching is by colony side and tag ID. Every block is a whole-window average: the ~14-hour July 23 block02 reference, the 49-hour July 24 recording and the ~24-hour July 30–31 tracked window differ in clock coverage and duration. The reference spans 19:31–09:30, primarily nighttime plus morning. A switch can reflect circadian occupancy, a broader average of multiple states, environmental changes, or task allocation. This analysis does not clock-match histograms or establish when within a block a switch occurred or what happened in gaps.", "",
              "Original independent labels from later blocks are included only for provenance; they are not used to detect switches. Original labeled reference cohorts are preserved; future maps are never subject to the original grid selection filter. Ants without an original label can be classified but cannot count as an original-label switch. Zero-occupancy profiles remain unassigned.", "",
              "Trip metrics are unknown (NA) for ants omitted from the existing optional trip analysis, including some colony-resident reference ants. NA does not mean zero trips. The source trip selection limits before/after trip comparisons for newly roaming ants; colony occupancy and activity are available independently.", "",
              "Interactive explorer: individual_cluster_explorer.html. Tables: profiles (all maps and behavior), reference (original labels/LOO diagnostics), assignments (every later prediction), neighbors (exact five contributing identities/distances/weights), calibration (LOO recovery by cluster/k), transitions, examples. occupancy_maps.npz and grid_edges.npz preserve the loaded arrays. run_manifest.json fingerprints inputs and code. In example figures, each row uses one shared occupancy scale, with fixed reference prototypes and an unchanged-label control when available."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    table_html = tables["transitions"].to_html(index=False)
    figures = "".join(f'<h2>{html.escape(p.stem.replace("_", " "))}</h2><a href="{p.name}"><img loading="lazy" src="{p.name}"></a>' for p in sorted(output.glob("*.png")))
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Occupancy cluster switching</title><style>body{font:16px system-ui;margin:30px;line-height:1.5}img{max-width:1100px;width:100%}td,th{padding:6px}pre{white-space:pre-wrap;max-width:1150px}</style><h1>Fixed-reference occupancy cluster switching</h1><p><a href="individual_cluster_explorer.html">Open interactive per-ant explorer</a> · <a href="report.md">Analysis notes and example table</a></p>'+table_html+'<pre>'+html.escape("\n".join(lines))+"</pre>"+figures)
