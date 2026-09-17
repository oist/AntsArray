"""Alternative views of trip length, clock effects, and within-ant changes."""
import html
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import long_timescale_trip_diagnostics as td
from analysis.long_timescale_plots import COLORS, SIDES, save

FIGURES = {
    "01_same_ants_duration_and_tail": ("Do the same ants take longer trips?", "Mean, median, upper tail, and ≥10-minute-trip frequency across all three recordings. Thin lines retain all available ant estimates; the thick line summarizes only ants with completed trips in every recording."),
    "02_clock_matched_durations": ("Does the shift survive clock matching?", "Each ant contributes equally across the same 2-hour departure-clock strata in every recording. Separate light/dark panels distinguish longer excursions from changes in the hours observed."),
    "03_day_night_cycles": ("Is this a gradual change or a nighttime effect?", "Medians by light-on cycle. Each whole night belongs to the date darkness begins. Labels give recorded phase hours; partial phases are retained, and unrecorded days remain blank."),
    "04_trip_duration_survival": ("Are most trips longer, or only a long tail?", "The fraction of completed trips exceeding each duration, giving each ant equal weight. Logarithmic duration axes show short trips and the long tail together."),
    "05_fewer_longer_trips": ("Do longer trips accompany fewer departures?", "Within-ant fold changes from first to last recording, using matched clock strata. Upper-left points mean longer trips and fewer departures per observed hour."),
    "06_coverage_and_boundary_sensitivity": ("Could tracking coverage or recording edges explain it?", "Primary results alongside explicitly labeled high-coverage and recording-edge checks. Bars are descriptive ant-bootstrap intervals, not independent-colony uncertainty."),
    "07_two_complete_nights": ("Does it happen within one continuous recording?", "Matched ants across the fully recorded nights of 0724. This avoids comparing a short nighttime calendar-day fragment with a whole day."),
    "08_sleep_during_excursions": ("Does outside time include sleep?", "Sleep fraction during each completed trip, using existing classifier contexts. This separates duration from an assumption that an entire excursion is active work."),
}


def centers(manifest):
    return pd.to_datetime([manifest["recording_centers"][r] for r in manifest["recordings"]])


def trajectory(ax, table, recordings, x, metric, color, *, show_all=True):
    wide = table.pivot(index="ant", columns="recording", values=metric).reindex(columns=recordings)
    fixed = wide.dropna()
    for _, row in (wide if show_all else fixed).iterrows():
        ax.plot(x, row, color="#8797a6", alpha=.35, lw=.65, marker=".", markersize=3)
    if len(fixed):
        ax.plot(x, fixed.median(), color=color, lw=2.4, marker="o", markersize=5, label=f"Median: same {len(fixed)} ants")
        ax.fill_between(x, fixed.quantile(.25), fixed.quantile(.75), color=color, alpha=.12, label="Ant interquartile range")
    ax.set_xticks(x, [r + "\n" + t.strftime("%b %d") for r, t in zip(recordings, x)])
    ax.grid(alpha=.15); ax.legend(fontsize=8)
    return fixed


def same_ants(tables, manifest, output):
    for j, side in enumerate(SIDES):
        fig, axes = plt.subplots(1, 4, figsize=(22, 6), squeeze=False, layout="constrained")
        data = tables["observed"].query("phase == 'all'")
        for i, (metric, label) in enumerate(td.MEASURES.items()):
            ax = axes[0, i]
            trajectory(ax, data[data.side.eq(side)], manifest["recordings"], centers(manifest), metric, COLORS[j])
            ax.set(title=f"{side}: {label}", ylabel=label)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Same-ant trajectories across recording dates\nMean versus typical trip length and long-trip frequency; gray = all observed ants, colored = fixed cohort")
        save(fig, output, f"01_same_ants_duration_and_tail_{side}")


def clock_plot(tables, manifest, output):
    for i, side in enumerate(SIDES):
        fig, axes = plt.subplots(1, 3, figsize=(19, 6), squeeze=False, layout="constrained")
        for j, phase in enumerate(("all", "light", "dark")):
            part = tables["clock_matched"].query("side == @side and phase == @phase")
            trajectory(axes[0, j], part, manifest["recordings"], centers(manifest), "median", COLORS[j], show_all=False)
            axes[0, j].set(title=f"{side}: {phase}", ylabel="Clock-matched median trip duration (min)")
        fig.suptitle(f"{side.capitalize()} colony\n" + "Do longer trips persist when comparing the same departure-clock strata?\nEqual weight per 2 h clock stratum within ant; then equal weight per ant; no minimum number of trips")
        save(fig, output, f"02_clock_matched_durations_{side}")


def cycle_plot(tables, manifest, output):
    for i, side in enumerate(SIDES):
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), squeeze=False, layout="constrained")
        cov = tables["phase_coverage"]
        days = sorted(cov.cycle_day.unique())
        x = pd.to_datetime(days)
        baseline = tables["observed"].query("side == @side and phase == 'all'").pivot(index="ant", columns="recording", values="median").dropna().index
        for j, phase in enumerate(("light", "dark")):
            ax = axes[0, j]
            part = tables["cycle_profiles"].query("side == @side and phase == @phase")
            wide = part.pivot(index="ant", columns="cycle_day", values="median").reindex(columns=days)
            for _, row in wide.iterrows():
                ax.plot(x, row, lw=.7, color="#8b9ba9", alpha=.4, marker=".", markersize=3)
            fixed = wide.reindex(baseline)
            ax.plot(x, fixed.median(), color=COLORS[j], marker="o", lw=2.2, label=f"Median of fixed cohort ({len(baseline)} ants)")
            hours = cov[cov.phase.eq(phase)].set_index("cycle_day").reindex(days).recording_hours
            ax.set_xticks(x, [d[5:]+f"\n{h:.1f} h" for d, h in zip(days, hours)])
            ax.set(title=f"{side}: {'lights on' if phase == 'light' else 'night beginning on this date'}", ylabel="Median completed-trip duration (min)", xlabel="Cycle date; recorded phase hours")
            ax.grid(alpha=.15); ax.legend(fontsize=8)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Whole light/dark phases reveal the nighttime pattern\nAll available ants and partial phases retained; fixed cohort summary uses available observations and leaves missing days blank")
        save(fig, output, f"03_day_night_cycles_{side}")


def survival(tables, manifest, output):
    events = tables["events"]
    grid = np.geomspace(.5, max(events.duration_minutes.max(), 60), 220)
    for i, side in enumerate(SIDES):
        fig, axes = plt.subplots(1, 2, figsize=(17, 6), squeeze=False, layout="constrained")
        for j, phase in enumerate(("light", "dark")):
            ax = axes[0, j]
            part = events.query("side == @side and phase == @phase")
            counts = part.groupby(["ant", "recording"]).size().unstack().reindex(columns=manifest["recordings"])
            ants = counts.dropna().index
            for k, recording in enumerate(manifest["recordings"]):
                lines = []
                for ant in ants:
                    values = np.sort(part.loc[part.ant.eq(ant) & part.recording.eq(recording), "duration_minutes"])
                    lines.append(100 * (len(values)-np.searchsorted(values, grid, side="right"))/len(values))
                if lines:
                    ax.plot(grid, np.mean(lines, axis=0), color=COLORS[k], lw=2, label=f"{recording}: {len(ants)} same ants")
            ax.axvline(10, color="#777", ls=":", lw=1)
            ax.set(xscale="log", ylim=(0, 100), title=f"{side}: {phase}", xlabel="Trip duration (min, logarithmic axis)", ylabel="Completed trips longer than this (%)")
            ax.grid(alpha=.15); ax.legend(fontsize=8)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Typical trips versus the long tail\nEach ant receives equal weight; light/dark defined by departure time; all trips in the fixed cohort retained")
        save(fig, output, f"04_trip_duration_survival_{side}")


def rate_length(tables, manifest, output):
    for side in SIDES:
        fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
        first, last = manifest["recordings"][0], manifest["recordings"][-1]
        data = tables["clock_matched"].query("side == @side and phase == 'all'")
        med = data.pivot(index="ant", columns="recording", values="median")
        rate = data.pivot(index="ant", columns="recording", values="trip_rate")
        x, y = rate[last]/rate[first], med[last]/med[first]
        ax.scatter(x, y, s=42, color=COLORS[0], alpha=.8)
        for ant in x.index:
            ax.annotate(ant.split(":")[-1], (x[ant], y[ant]), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.axvline(1, ls="--", color="#777", lw=1); ax.axhline(1, ls="--", color="#777", lw=1)
        ax.set(xscale="log", yscale="log", xlabel="Later / first: completed-trip rate", ylabel="Later / first: median trip duration", title=f"{side}: {(x.lt(1) & y.gt(1)).sum()}/{len(x)} fewer and longer")
        ax.grid(alpha=.15)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Are the same ants making fewer, longer excursions?\nClock-matched first-to-last fold changes; labels are tag IDs; 1 means no change")
        save(fig, output, f"05_fewer_longer_trips_{side}")


def sensitivity(tables, manifest, output):
    data = tables["sensitivity"].query("phase == 'all'")
    names = {"all_completed": "All completed trips", "coverage_at_least_95_percent": "Position coverage ≥95%",
             "one_hour_edge_buffer_and_duration_cap": "≥1 h from block edges; trip ≤1 h"}
    for i, side in enumerate(SIDES):
        fig, axes = plt.subplots(1, 2, figsize=(17, 6), squeeze=False, layout="constrained")
        for j, metric in enumerate(("mean", "median")):
            ax = axes[0, j]
            part = data.query("side == @side and metric == @metric").set_index("method").reindex(names)
            y = np.arange(len(part))
            ax.errorbar(part.median_fold, y, xerr=[part.median_fold-part.fold_ci_low, part.fold_ci_high-part.median_fold], fmt="o", color=COLORS[i], capsize=4)
            ax.set_yticks(y, [names[k]+f"\n{r.increased}/{r.n_ants} increased; {r.retained_trips:,} trips retained overall" for k, r in part.iterrows()])
            ax.axvline(1, color="#888", ls="--"); ax.set(xlabel="Median within-ant later / first fold", title=f"{side}: {metric} duration")
            ax.grid(axis="x", alpha=.15)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Does the increase survive coverage and recording-edge checks?\nExplicit sensitivity subsets; bars = 95% descriptive ant-bootstrap interval; primary figures remain unfiltered")
        save(fig, output, f"06_coverage_and_boundary_sensitivity_{side}")


def two_nights(tables, manifest, output):
    table = tables["complete_nights"]
    if table.empty:
        return
    rec = table.source_recording.iloc[0]
    table = table[table.source_recording.eq(rec)]
    days = sorted(table.recording.unique())
    for i, side in enumerate(SIDES):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False, layout="constrained")
        for j, metric in enumerate(("mean", "median", "long10")):
            ax = axes[0, j]
            fixed = trajectory(ax, table[table.side.eq(side)], days, pd.to_datetime(days), metric, COLORS[i])
            ax.set(title=f"{side}: {(fixed.iloc[:, -1]>fixed.iloc[:, 0]).sum()}/{len(fixed)} increased", ylabel=td.MEASURES[metric], xlabel="Night beginning on this date")
        fig.suptitle(f"{side.capitalize()} colony\n" + f"Two fully recorded nights within {rec}\nSame-ant comparison within one tracking block; each night lasts 10 hours")
        save(fig, output, f"07_two_complete_nights_{side}")


def sleep_plot(tables, manifest, output):
    data = tables["events"]
    for i, side in enumerate(SIDES):
        fig, axes = plt.subplots(1, 2, figsize=(17, 6), squeeze=False, layout="constrained")
        part = data[data.side.eq(side)]
        for j, phase in enumerate(("light", "dark")):
            ax = axes[0, j]
            points = part[part.phase.eq(phase)]
            for k, rec in enumerate(manifest["recordings"]):
                v = points[points.recording.eq(rec)]
                ax.scatter(v.duration_minutes, v.sleep_percent, s=7, color=COLORS[k], alpha=.15, rasterized=True)
                edges = np.array([.5, 1, 2, 5, 10, 20, 60, 240])
                medians = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    g = v[v.duration_minutes.ge(lo) & v.duration_minutes.lt(hi)].groupby("ant").sleep_percent.median()
                    medians.append(g.median())
                ax.plot(np.sqrt(edges[:-1]*edges[1:]), medians, color=COLORS[k], lw=2.2, marker="o", label=rec)
            ax.set(xscale="log", ylim=(-2, 102), title=f"{side}: {phase}", xlabel="Completed-trip duration (min)", ylabel="Trip seconds classified asleep (%)")
            ax.legend(fontsize=8); ax.grid(alpha=.15)
        fig.suptitle(f"{side.capitalize()} colony\n" + "Long outside excursions need not consist entirely of active work\nDots = all completed trips with known sleep state; lines = median of ant medians in duration bands")
        save(fig, output, f"08_sleep_during_excursions_{side}")


def save_figures(tables, manifest, output):
    for function in (same_ants, clock_plot, cycle_plot, survival, rate_length, sensitivity, two_nights, sleep_plot):
        function(tables, manifest, output)
        print("Saved", function.__name__, flush=True)


def write_explorer(tables, manifest, output):
    from plotly.offline import get_plotlyjs
    events = tables["events"][["ant", "side", "recording", "exit_timestamp", "phase", "cycle_day", "duration_minutes", "observed_coverage", "sleep_percent"]].copy()
    events["exit_timestamp"] = events.exit_timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")
    payload = dict(events=json.loads(events.to_json(orient="records", double_precision=5)),
                   observed=json.loads(tables["observed"].to_json(orient="records", double_precision=5)),
                   matched=json.loads(tables["clock_matched"].to_json(orient="records", double_precision=5)),
                   cycles=json.loads(tables["cycle_profiles"].to_json(orient="records", double_precision=5)),
                   recordings=manifest["recordings"], centers=manifest["recording_centers"], metrics=td.MEASURES)
    text = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Individual trip duration explorer</title><script src="plotly.min.js"></script><style>body{font:15px system-ui;color:#243b4b;margin:24px;background:#f5f7fa}h1{font-size:27px}.controls{display:flex;gap:18px;flex-wrap:wrap;background:white;padding:16px;position:sticky;top:0;z-index:10}label{display:grid;gap:5px}select{padding:8px;font:inherit}.chart{background:white;height:470px;margin:20px 0;border:1px solid #ddd}p{max-width:1250px;line-height:1.5}a{color:#14779b}</style></head><body>
<h1>Do individual ants take longer trips over time?</h1><p><a href="index.html">Figure guide and findings</a> · <a href="report.md">Methods and numerical report</a></p>
<p>Follow the same ant across all three recordings. The mean is sensitive to a long tail; compare it with the median and the fraction of trips lasting at least 10 minutes. Day/night uses departure time. All saved completed trips remain available.</p>
<div class="controls"><label>Colony<select id="side"><option>left</option><option>right</option></select></label><label>Ant<select id="ant"></select></label><label>Duration summary<select id="metric"></select></label><label>Phase<select id="phase"><option value="all">All hours</option><option value="light">Lights on</option><option value="dark">Lights off</option></select></label><label>Clock weighting<select id="mode"><option value="observed">All observed trips</option><option value="matched">Same 2 h clock strata</option></select></label><span id="status"></span></div>
<h2>Within-ant trajectories across recordings</h2><p>Thin gray lines identify individual ants; the colored line is the median for ants with completed trips in every recording. Selecting an ant shows that ant alone. Lines connect recording summaries and do not fill the observation gaps.</p><div id="trends" class="chart"></div>
<h2>Every completed trip on the actual calendar</h2><p>One point per saved trip. Hover for ID, duration, position coverage and classified sleep. The logarithmic axis shows short trips and very long excursions together.</p><div id="events" class="chart"></div>
<h2>Typical trips and the long tail</h2><p>Each ant contributes equally after its trip-duration distribution is calculated. This panel uses observed trips within the selected phase; clock weighting applies to the trajectory panel above.</p><div id="survival" class="chart"></div>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),colors=['#187ca5','#d47830','#7566b0'];
const finite=x=>x!==null&&Number.isFinite(x);const median=a=>{if(!a.length)return null;const v=[...a].sort((a,b)=>a-b),i=Math.floor(v.length/2);return v.length%2?v[i]:(v[i-1]+v[i])/2};
function groups(rows,key){const m=new Map();for(const r of rows){let k=key(r);if(!m.has(k))m.set(k,[]);m.get(k).push(r)}return m}
for(const [key,label] of Object.entries(D.metrics))$('metric').add(new Option(label,key));$('metric').value='median';
function ants(){const old=$('ant').value,values=[...new Set(D.events.filter(r=>r.side===$('side').value).map(r=>r.ant))].sort();$('ant').replaceChildren(new Option('All observed ants','all'),...values.map(v=>new Option(v,v)));if(values.includes(old))$('ant').value=old}
function plot(id,traces,layout){return Plotly.react(id,traces,{margin:{l:80,r:35,t:35,b:65},font:{family:'system-ui'},...layout},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}})}
async function render(){window.__READY__=false;const side=$('side').value,ant=$('ant').value,phase=$('phase').value,metric=$('metric').value,mode=$('mode').value;
 const selected=(mode==='matched'?D.matched:D.observed).filter(r=>r.side===side&&r.phase===phase&&(ant==='all'||r.ant===ant)),ev=D.events.filter(r=>r.side===side&&(ant==='all'||r.ant===ant)&&(phase==='all'||r.phase===phase));
 const byant=groups(selected,r=>r.ant),x=D.recordings.map(r=>D.centers[r].replace(' ','T')),traces=[],fixed=[];
 for(const [id,rows] of byant){const map=new Map(rows.map(r=>[r.recording,r[metric]])),y=D.recordings.map(r=>map.get(r)??null);if(y.every(finite))fixed.push(y);if(y.some(finite))traces.push({x,y,name:id,mode:'lines+markers',showlegend:ant!=='all',line:{color:ant==='all'?'rgba(100,115,135,.35)':colors[0],width:ant==='all'?1:2},customdata:D.recordings.map(r=>r+' '+id),hovertemplate:'%{customdata}<br>%{y:.3f}<extra></extra>',connectgaps:false})}
 if(ant==='all')traces.push({x,y:D.recordings.map((r,i)=>median(fixed.map(v=>v[i]))),name:'Median of '+fixed.length+' same ants',mode:'lines+markers',line:{color:colors[0],width:3}});
 const eventTraces=D.recordings.map((rec,i)=>{const v=ev.filter(r=>r.recording===rec);return {x:v.map(r=>r.exit_timestamp),y:v.map(r=>r.duration_minutes),mode:'markers',type:'scattergl',name:rec,marker:{color:colors[i],size:ant==='all'?4:7,opacity:ant==='all'?.35:.8},customdata:v.map(r=>r.ant+'<br>'+r.phase+'<br>Position coverage '+(r.observed_coverage*100).toFixed(1)+'%<br>Sleep '+(finite(r.sleep_percent)?r.sleep_percent.toFixed(1)+'%':'unknown')),hovertemplate:'%{x}<br>%{y:.3f} min<br>%{customdata}<extra></extra>'}});
 const grid=Array.from({length:200},(_,i)=>.5*Math.pow(500,i/199)),curves=D.recordings.map((rec,i)=>{const ants=groups(ev.filter(r=>r.recording===rec),r=>r.ant),all=[...ants.values()];return {x:grid,y:grid.map(t=>all.length?100*all.reduce((s,v)=>s+v.filter(r=>r.duration_minutes>t).length/v.length,0)/all.length:null),name:rec+' ('+all.length+' ants)',mode:'lines',line:{color:colors[i],width:2}}});
 await Promise.all([plot('trends',traces,{xaxis:{type:'date',tickvals:x,ticktext:D.recordings},yaxis:{title:{text:D.metrics[metric]}},title:{text:phase+' · '+(mode==='matched'?'equal weight on shared clock strata':'all observed trips')}}),plot('events',eventTraces,{xaxis:{type:'date',range:[D.events[0].exit_timestamp,D.events.at(-1).exit_timestamp]},yaxis:{type:'log',tickmode:'array',tickvals:[.5,1,2,5,10,20,60,120,240],ticktext:['0.5','1','2','5','10','20','60','120','240'],title:{text:'Trip duration (min)'}}}),plot('survival',curves,{xaxis:{type:'log',tickmode:'array',tickvals:[.5,1,2,5,10,20,60,120,240],ticktext:['0.5','1','2','5','10','20','60','120','240'],title:{text:'Duration (min)'}},yaxis:{range:[0,100],title:{text:'Completed trips longer than this (%)'}}})]);
 $('status').textContent=ev.length.toLocaleString()+' completed trips · '+fixed.length+' ants comparable across all recordings';window.__READY__=true;
}
function fail(e){window.__ERROR__=String(e);$('status').textContent='Error: '+e.message;console.error(e)}
$('side').addEventListener('change',()=>{ants();render().catch(fail)});for(const id of ['ant','metric','phase','mode'])$(id).addEventListener('change',()=>render().catch(fail));ants();render().catch(fail);
</script></body></html>"""
    # Preserve calendar range even though context extraction grouped by ant.
    payload["events"].sort(key=lambda r: r["exit_timestamp"])
    embedded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (Path(output) / "individual_trip_explorer.html").write_text(text.replace("__DATA__", embedded))
    (Path(output) / "plotly.min.js").write_text(get_plotlyjs())


def write_index(output):
    root = Path(output)
    rows = []
    for name, (title, caption) in FIGURES.items():
        for side in SIDES:
            filename = f"{name}_{side}.png"
            heading = f"{side.capitalize()} colony — {title}"
            if (root / filename).is_file():
                rows.append(f'<section><h2>{html.escape(heading)}</h2><p>{html.escape(caption)}</p><a href="{filename}"><img loading="lazy" src="{filename}" alt="{html.escape(heading)}"></a></section>')
    report = (root / "report.md").read_text()
    text = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Trip duration over time</title><style>body{font:16px system-ui;line-height:1.55;color:#243b4b;background:#f5f7fa;max-width:1500px;margin:auto;padding:25px}section{background:white;padding:20px;margin:20px 0;border:1px solid #dde3e9}img{width:100%;height:auto}a{color:#14799d}pre{white-space:pre-wrap;font:inherit}h1{font-size:30px}h2{font-size:23px}</style></head><body><h1>Are ants taking longer trips over time?</h1><p><a href="individual_trip_explorer.html">Open interactive ant explorer</a> · <a href="report.md">Findings and methods</a> · <a href="observed_changes.csv">Within-ant changes</a> · <a href="clock_matched_changes.csv">Clock-matched changes</a> · <a href="sensitivity.csv">Coverage and edge checks</a> · <a href="run_manifest.json">Provenance</a></p><details><summary>Read numerical findings and interpretation</summary><pre>__REPORT__</pre></details>__FIGURES__</body></html>'''
    (root / "index.html").write_text(text.replace("__FIGURES__", "\n".join(rows)).replace("__REPORT__", html.escape(report)))
