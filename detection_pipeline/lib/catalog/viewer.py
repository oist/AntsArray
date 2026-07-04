"""Emit a self-contained catalog.html for browsing the catalog in a web browser.

The row data is embedded inline (no external fetch), so the file opens straight
from the bucket via file:// with no server and works offline. Vanilla JS/CSS,
no CDN. Colours follow the data-viz status palette; every chip carries a text
label so meaning is never colour-alone. Python 3.6-safe (no f-strings).
"""
import json

# --- HTML template. Braces are literal; only __TOKEN__ placeholders are filled.
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>basler catalog</title>
<style>
:root{
  --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --info:#2a78d6; --bar:#2a78d6; --barbg:#e1e0d9;
}
html[data-theme="dark"]{
  --plane:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --info:#3987e5; --bar:#3987e5; --barbg:#2c2c2a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:13px;line-height:1.4}
.wrap{max-width:1600px;margin:0 auto;padding:18px 20px 60px}
header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:14px}
h1{font-size:19px;margin:0;font-weight:650}
.sub{color:var(--muted);font-size:12px}
.spacer{flex:1}
button,select,input{font:inherit;color:inherit}
.btn{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  padding:5px 10px;cursor:pointer}
.btn:hover{border-color:var(--axis)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 14px}
.kpi .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.kpi .l{color:var(--ink2);font-size:12px;margin-top:2px}
.tabs{display:flex;gap:6px;margin-bottom:12px;border-bottom:1px solid var(--grid)}
.tab{background:none;border:none;padding:8px 12px;cursor:pointer;color:var(--ink2);
  border-bottom:2px solid transparent;margin-bottom:-1px}
.tab.active{color:var(--ink);border-bottom-color:var(--info);font-weight:600}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
.toolbar input[type=search]{background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:6px 10px;min-width:230px}
.toolbar select{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:6px 8px}
.toolbar label{color:var(--ink2);font-size:12px;display:inline-flex;gap:5px;align-items:center}
.count{color:var(--muted);font-size:12px}
.tablewrap{overflow:auto;border:1px solid var(--border);border-radius:12px;background:var(--surface);max-height:76vh}
table{border-collapse:collapse;width:100%}
th,td{padding:6px 9px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--grid);vertical-align:top}
th{position:sticky;top:0;background:var(--surface);cursor:pointer;font-weight:600;
  color:var(--ink2);user-select:none;z-index:1}
th:hover{color:var(--ink)}
th .arrow{color:var(--info);font-size:10px}
tr:hover td{background:color-mix(in srgb,var(--info) 6%,transparent)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.chip{display:inline-flex;align-items:center;gap:5px;padding:1px 8px;border-radius:999px;
  font-size:11px;border:1px solid;line-height:1.7}
.dot{width:7px;height:7px;border-radius:50%;flex:none}
.haz{display:inline-block;margin:1px 3px 1px 0;padding:0 6px;border-radius:6px;font-size:10.5px;
  border:1px solid;line-height:1.7;font-variant-numeric:tabular-nums}
.bar{display:inline-block;width:58px;height:7px;border-radius:4px;background:var(--barbg);
  overflow:hidden;vertical-align:middle;margin-right:6px}
.bar>i{display:block;height:100%;background:var(--bar)}
.muted{color:var(--muted)}
.mono{font-variant-numeric:tabular-nums}
.path{max-width:360px;overflow:hidden;text-overflow:ellipsis;color:var(--muted)}
.recbtn{cursor:pointer;border-radius:6px;font-size:10.5px;padding:1px 7px;border:1px solid;
  color:var(--ink);white-space:nowrap}
.recbtn:hover{filter:brightness(1.08)}
.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:10;
  align-items:center;justify-content:center;padding:20px}
.modal .box{background:var(--surface);border:1px solid var(--border);border-radius:12px;
  max-width:860px;width:100%;max-height:85vh;overflow:auto;padding:16px 18px}
.modal .mtitle{font-weight:650;font-size:15px;margin:0 0 10px}
.modal pre{white-space:pre-wrap;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;
  background:var(--plane);border:1px solid var(--border);border-radius:8px;padding:12px;overflow:auto}
.modal .row{display:flex;gap:8px;margin-top:10px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>basler catalog</h1>
  <span class="sub">__ROOT__ &middot; scanned __SCANNED_AT__</span>
  <span class="spacer"></span>
  <button class="btn" id="theme">◐ theme</button>
</header>
<div class="kpis" id="kpis"></div>
<div class="tabs" id="tabs"></div>
<div class="toolbar" id="toolbar"></div>
<div class="tablewrap"><table id="tbl"><thead></thead><tbody></tbody></table></div>
<p class="count" id="count"></p>
</div>
<div class="modal" id="modal"><div class="box">
  <p class="mtitle"></p>
  <pre></pre>
  <div class="row"><button class="btn copy">copy command</button><button class="btn close">close</button></div>
</div></div>
<script>
const DATA = {catalog: __CATALOG__, videos: __VIDEOS__, trials: __TRIALS__};
const C = {good:getVar('--good'),warn:getVar('--warn'),serious:getVar('--serious'),
           crit:getVar('--crit'),info:getVar('--info'),muted:getVar('--muted')};
function getVar(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()||'#888';}
function hexa(h,a){h=h.replace('#','');if(h.length===3)h=h.split('').map(c=>c+c).join('');
  const r=parseInt(h.slice(0,2),16),g=parseInt(h.slice(2,4),16),b=parseInt(h.slice(4,6),16);
  return 'rgba('+r+','+g+','+b+','+a+')';}
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}

const STATUS={complete:'good',partial:'warn',not_started:'muted',analysis_only:'info'};
const HEALTH={ok:'good',warn:'warn',bad:'crit',unknown:'muted','n/a':'muted'};
const HAZ_CRIT=new Set(['SLEAP_H5_MISSING','ARUCO_MISSING','STAGE_SKEW','SILENT_PARTIAL','DEAD_SYMLINK','TRUNCATED_ARTIFACT']);
const HAZ_WARN=new Set(['NAME_DATE_MISMATCH','CAM_COUNT_OFF','NO_SESS_FILE','CHUNK_UNVERIFIABLE']);

function chip(label,colorKey){
  if(label==='')return '<span class="muted">–</span>';
  const c=C[colorKey]||C.muted;
  return '<span class="chip" style="background:'+hexa(c,.14)+';border-color:'+hexa(c,.45)+'">'
       +'<span class="dot" style="background:'+c+'"></span>'+esc(label)+'</span>';
}
function hazCell(v){
  if(!v)return '<span class="muted">–</span>';
  return v.split('|').map(f=>{
    const c=HAZ_CRIT.has(f)?C.crit:(HAZ_WARN.has(f)?C.warn:C.muted);
    return '<span class="haz" style="background:'+hexa(c,.12)+';border-color:'+hexa(c,.4)+';color:var(--ink)">'+esc(f)+'</span>';
  }).join('');
}
function pctCell(r){
  const p=r.completeness_pct, st=r.completeness_state;
  if(p==='')return '<span class="muted">'+esc(st||'')+'</span>';
  const w=Math.round(parseFloat(p)*100);
  return '<span class="bar"><i style="width:'+w+'%"></i></span><span class="mono">'+w+'%</span>'
       +' <span class="muted">'+esc(st)+'</span>';
}
function boolCell(v){if(v==='true')return chip('yes','good');if(v==='false')return chip('no','muted');return '<span class="muted">–</span>';}
function recoverCell(r){
  if(!r.recover_type)return '<span class="muted">–</span>';
  const cheap=(r.recover_type==='slp2h5'||r.recover_type==='upload');
  const c=cheap?C.info:C.warn;
  const payload=esc(JSON.stringify({t:r.recover_type,m:r.recover_missing,c:r.recover_cmd,s:r.recover_steps}));
  return '<button class="recbtn" data-rec="'+payload+'" style="border-color:'+hexa(c,.5)+';background:'+hexa(c,.13)+'">▸ '+esc(r.recover_type)+' '+esc(r.recover_missing)+'</button>';
}
function openRec(rec){
  const m=document.getElementById('modal');
  m.querySelector('.mtitle').textContent='recover — '+rec.t+'   ('+rec.m+')';
  m.querySelector('pre').textContent=rec.s;
  m.querySelector('.copy').onclick=()=>{try{navigator.clipboard.writeText(rec.c);}catch(e){}};
  m.style.display='flex';
}

const R={
  txt:(v)=>v===''?'<span class="muted">–</span>':esc(v),
  num:(v)=>v===''?'<span class="muted">–</span>':'<span class="mono">'+esc(v)+'</span>',
};
const VIEWS={
  catalog:{rows:DATA.catalog, cols:[
    ['session_id','session',R.txt],['block','block',R.txt],['session_kind','kind',R.txt],
    ['layout','layout',R.txt],['date_start','date',R.txt],['labels','labels',R.txt],
    ['is_stim','stim',(v)=>boolCell(v)],['n_trials_observed','trials',R.num],
    ['n_colony_videos','vids',R.num],['has_sidecars','sidecars',(v)=>boolCell(v)],
    ['health_flag','health',(v)=>chip(v,HEALTH[v]||'muted')],
    ['pipeline_status','pipeline',(v)=>chip(v,STATUS[v]||'muted')],
    ['completeness_pct','complete %',(v,r)=>pctCell(r)],
    ['n_slp','slp',R.num],['n_aruco_det','aruco',R.num],['n_sleap_data','sleap_h5',R.num],
    ['sleap_models','models',R.txt],['saion_partition','partition',R.txt],
    ['stage_reached','stage',R.txt],['hazard_flags','hazards',(v)=>hazCell(v)],
    ['recover_type','recover',(v,r)=>recoverCell(r)],
  ]},
  videos:{rows:DATA.videos, cols:[
    ['session_id','session',R.txt],['block','block',R.txt],['vname','video',R.txt],
    ['cam_global','cam',R.num],['ext','ext',R.txt],['has_sidecar','sidecar',(v)=>boolCell(v)],
    ['fps','fps',R.num],['frame_count','frames',R.num],['duration_sec','dur s',R.num],
    ['missed_frames','missed',R.num],['frame_drop','drop',R.num],
    ['video_health','health',(v)=>chip(v,HEALTH[v]||'muted')],
    ['assigned_pc','pc',R.txt],['assigned_drive','drive',R.txt],
  ]},
  trials:{rows:DATA.trials, cols:[
    ['session_id','session',R.txt],['block','block',R.txt],['trial','trial',R.num],
    ['iso_time','time',R.txt],['duty','duty',R.num],['dur_s','dur s',R.num],
    ['interval_s','int s',R.num],['cam_frame_start','frame start',R.num],
    ['cam_frame_end','frame end',R.num],['gyro_rms_dps','gyro rms',R.num],
    ['acc_rms_g','acc rms',R.num],['temp_mean_C','temp C',R.num],
    ['imu_ok','imu',(v)=>boolCell(v)],
  ]},
};
const NUMKEYS=new Set(['n_trials_observed','n_colony_videos','n_slp','n_aruco_det','n_sleap_data',
  'completeness_pct','cam_global','fps','frame_count','duration_sec','missed_frames','frame_drop',
  'trial','duty','dur_s','interval_s','cam_frame_start','cam_frame_end','gyro_rms_dps','acc_rms_g','temp_mean_C']);

let view='catalog', sortKey='session_id', sortDir=1, q='', facet={};

function kpis(){
  const c=DATA.catalog;
  const cnt=(f)=>c.filter(f).length;
  const tiles=[
    ['blocks/rows',c.length],
    ['complete',cnt(r=>r.pipeline_status==='complete')],
    ['partial',cnt(r=>r.pipeline_status==='partial')],
    ['not started',cnt(r=>r.pipeline_status==='not_started')],
    ['stim',cnt(r=>r.is_stim==='true')],
    ['flagged',cnt(r=>r.hazard_flags!=='')],
  ];
  document.getElementById('kpis').innerHTML=tiles.map(t=>
    '<div class="kpi"><div class="v">'+t[1]+'</div><div class="l">'+t[0]+'</div></div>').join('');
}
function tabs(){
  const names=[['catalog','catalog'],['videos','videos'],['trials','trials']];
  document.getElementById('tabs').innerHTML=names.map(n=>
    '<button class="tab'+(n[0]===view?' active':'')+'" data-v="'+n[0]+'">'+n[1]+' ('+VIEWS[n[0]].rows.length+')</button>').join('');
  document.querySelectorAll('.tab').forEach(b=>b.onclick=()=>{view=b.dataset.v;q='';facet={};
    sortKey=VIEWS[view].cols[0][0];sortDir=1;render();});
}
function toolbar(){
  let h='<input type="search" id="q" placeholder="filter…" value="'+esc(q)+'">';
  if(view==='catalog'){
    h+=facetSel('session_kind','kind')+facetSel('pipeline_status','pipeline')+facetSel('is_stim','stim');
    h+='<label><input type="checkbox" id="hz"'+(facet.__hz?' checked':'')+'> flagged only</label>';
  }else if(view==='videos'){h+=facetSel('video_health','health');}
  h+='<span class="spacer"></span><span class="count" id="cnt"></span>';
  const tb=document.getElementById('toolbar');tb.innerHTML=h;
  document.getElementById('q').oninput=(e)=>{q=e.target.value;renderBody();};
  tb.querySelectorAll('select[data-k]').forEach(s=>s.onchange=()=>{facet[s.dataset.k]=s.value;renderBody();});
  const hz=document.getElementById('hz');if(hz)hz.onchange=()=>{facet.__hz=hz.checked;renderBody();};
}
function facetSel(key,label){
  const vals=[...new Set(VIEWS[view].rows.map(r=>r[key]).filter(v=>v!==''&&v!=null))].sort();
  return '<select data-k="'+key+'"><option value="">'+label+': all</option>'+
    vals.map(v=>'<option'+(facet[key]===v?' selected':'')+'>'+esc(v)+'</option>').join('')+'</select>';
}
function filtered(){
  let rows=VIEWS[view].rows.slice();
  const cols=VIEWS[view].cols.map(c=>c[0]);
  if(q){const s=q.toLowerCase();rows=rows.filter(r=>cols.some(k=>String(r[k]).toLowerCase().includes(s)));}
  for(const k in facet){if(k==='__hz')continue;if(facet[k]!=='')rows=rows.filter(r=>r[k]===facet[k]);}
  if(facet.__hz)rows=rows.filter(r=>r.hazard_flags!=='');
  const num=NUMKEYS.has(sortKey);
  rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];
    if(num){x=x===''?-Infinity:parseFloat(x);y=y===''?-Infinity:parseFloat(y);return (x-y)*sortDir;}
    return String(x).localeCompare(String(y))*sortDir;});
  return rows;
}
function renderHead(){
  const cols=VIEWS[view].cols;
  document.querySelector('#tbl thead').innerHTML='<tr>'+cols.map(c=>{
    const a=c[0]===sortKey?(' <span class="arrow">'+(sortDir>0?'▲':'▼')+'</span>'):'';
    return '<th data-k="'+c[0]+'">'+esc(c[1])+a+'</th>';}).join('')+'</tr>';
  document.querySelectorAll('#tbl th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k;if(k===sortKey)sortDir=-sortDir;else{sortKey=k;sortDir=1;}render();});
}
function renderBody(){
  const cols=VIEWS[view].cols, rows=filtered();
  const body=rows.map(r=>'<tr>'+cols.map(c=>{
    const num=NUMKEYS.has(c[0])&&c[2]===R.num;
    return '<td'+(num?' class="num"':'')+'>'+c[2](r[c[0]],r)+'</td>';}).join('')+'</tr>').join('');
  document.querySelector('#tbl tbody').innerHTML=body||'<tr><td>no matching rows</td></tr>';
  const cnt=document.getElementById('cnt');if(cnt)cnt.textContent=rows.length+' / '+VIEWS[view].rows.length+' rows';
}
function render(){tabs();toolbar();renderHead();renderBody();}
function applyTheme(t){document.documentElement.setAttribute('data-theme',t);
  Object.assign(C,{good:getVar('--good'),warn:getVar('--warn'),serious:getVar('--serious'),
    crit:getVar('--crit'),info:getVar('--info'),muted:getVar('--muted')});renderBody();}
(function(){
  const dark=window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.setAttribute('data-theme',dark?'dark':'light');
  document.getElementById('theme').onclick=()=>applyTheme(
    document.documentElement.getAttribute('data-theme')==='dark'?'light':'dark');
  const modal=document.getElementById('modal');
  modal.addEventListener('click',e=>{if(e.target===modal||e.target.classList.contains('close'))modal.style.display='none';});
  document.addEventListener('keydown',e=>{if(e.key==='Escape')modal.style.display='none';});
  document.addEventListener('click',e=>{const b=e.target.closest&&e.target.closest('.recbtn');if(b){try{openRec(JSON.parse(b.dataset.rec));}catch(err){}}});
  kpis();render();
})();
</script>
</body>
</html>
"""


def _keep(rows, columns):
    """Slim rows to the columns the viewer uses (keeps the HTML small)."""
    return [{k: r.get(k, "") for k in columns} for r in rows]


_CATALOG_KEYS = ["session_id", "block", "session_kind", "layout", "date_start", "labels",
                 "is_stim", "n_trials_observed", "n_colony_videos", "has_sidecars",
                 "health_flag", "pipeline_status", "completeness_pct", "completeness_state",
                 "n_slp", "n_aruco_det", "n_sleap_data", "sleap_models", "saion_partition",
                 "stage_reached", "hazard_flags", "recover_type", "recover_missing",
                 "recover_cmd", "recover_steps"]
_VIDEO_KEYS = ["session_id", "block", "vname", "cam_global", "ext", "has_sidecar", "fps",
               "frame_count", "duration_sec", "missed_frames", "frame_drop", "video_health",
               "assigned_pc", "assigned_drive"]
_TRIAL_KEYS = ["session_id", "block", "trial", "iso_time", "duty", "dur_s", "interval_s",
               "cam_frame_start", "cam_frame_end", "gyro_rms_dps", "acc_rms_g",
               "temp_mean_C", "imu_ok"]


def _embed(rows, columns):
    # Escape '<' so a stray '</script>' in data cannot break out of the tag.
    return json.dumps(_keep(rows, columns)).replace("<", "\\u003c")


def write_html(path, catalog_rows, video_rows, trial_rows, scanned_at, root):
    """Write a self-contained catalog.html with the rows embedded."""
    html = _TEMPLATE
    html = html.replace("__ROOT__", root)
    html = html.replace("__SCANNED_AT__", scanned_at)
    html = html.replace("__CATALOG__", _embed(catalog_rows, _CATALOG_KEYS))
    html = html.replace("__VIDEOS__", _embed(video_rows, _VIDEO_KEYS))
    html = html.replace("__TRIALS__", _embed(trial_rows, _TRIAL_KEYS))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
