"""A self-contained browser for a bounded sample of search findings."""

from __future__ import annotations

import json
from pathlib import Path


def write_html_report(path: str | Path, *, metadata: dict, records: list[dict]) -> None:
    payload = json.dumps(
        {"metadata": metadata, "records": records},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    document = _DOCUMENT.replace("__CLUSTERGREP_DATA__", payload)
    Path(path).write_text(document, encoding="utf-8")


_DOCUMENT = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>clustergrep findings</title>
<style>
:root{--ink:#183028;--muted:#64746d;--paper:#f3f0e7;--card:#fffdf7;--line:#c8c4b8;--accent:#d84b2a;--cool:#176b67;--hit:#ffe36b}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 85% 0,#d9e7df 0,transparent 34rem),var(--paper);font-family:Georgia,"Times New Roman",serif}
header{padding:2.6rem clamp(1.2rem,4vw,4rem) 1.5rem;border-bottom:1px solid var(--line)}h1{font-size:clamp(2rem,5vw,4.4rem);font-weight:400;line-height:.9;margin:0}.kicker,.meta,button,input,label,.term{font-family:"Avenir Next","Gill Sans",sans-serif}.kicker{text-transform:uppercase;letter-spacing:.17em;color:var(--accent);font-size:.76rem}.meta{color:var(--muted);margin-top:1rem}
main{display:grid;grid-template-columns:minmax(15rem,22rem) minmax(0,1fr);gap:2rem;padding:2rem clamp(1.2rem,4vw,4rem)}aside{position:sticky;top:1rem;align-self:start;max-height:calc(100vh - 2rem);overflow-y:auto;overscroll-behavior:contain;scrollbar-gutter:stable;padding-right:.35rem}.panel{background:#ffffffa8;border:1px solid var(--line);padding:1rem;margin-bottom:1rem}.panel h2{font-size:1rem;margin:.1rem 0 .8rem}.search{width:100%;padding:.7rem;border:1px solid var(--line);background:var(--card);font-size:1rem}.terms{max-height:42vh;overflow:auto}.term{display:flex;gap:.55rem;align-items:center;margin:.45rem 0;font-size:.88rem}.term span:last-child{margin-left:auto;color:var(--muted)}.hint{font:.8rem/1.35 "Avenir Next","Gill Sans",sans-serif;color:var(--muted)}.context-term{font:600 .72rem "Avenir Next","Gill Sans",sans-serif;text-transform:uppercase;letter-spacing:.08em;margin:.9rem 0 .35rem;color:var(--cool)}button{border:1px solid var(--ink);background:var(--ink);color:white;padding:.65rem .8rem;cursor:pointer;margin:.2rem .2rem .2rem 0}button.secondary{background:transparent;color:var(--ink)}
#results{min-width:0}.status{font-family:"Avenir Next","Gill Sans",sans-serif;color:var(--muted);margin-bottom:1rem}.record{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--cool);padding:1rem 1.15rem;margin-bottom:.8rem;box-shadow:0 8px 22px #26352d0b}.source{font:.78rem "Avenir Next","Gill Sans",sans-serif;color:var(--muted);margin-bottom:.55rem}.text{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}.text.json{font:.86rem/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#edf1eb;border:1px solid #d4d9d2;border-radius:.25rem;padding:.8rem;max-height:34rem;overflow:auto;tab-size:2}.text mark{background:var(--hit);color:#17241f;border-radius:.12rem;padding:.04em .08em;box-shadow:0 0 0 1px #d4aa20}.badges{margin-top:.75rem}.badge{display:inline-block;font:.75rem "Avenir Next","Gill Sans",sans-serif;border:1px solid var(--line);border-radius:1rem;padding:.2rem .48rem;margin:.15rem .2rem .1rem 0}.badge b{color:var(--accent)}
@media(max-width:760px){main{grid-template-columns:1fr}aside{position:static;max-height:none;overflow:visible;padding-right:0}.terms{max-height:14rem}}
</style>
</head>
<body>
<header><div class="kicker">clustergrep / findings explorer</div><h1 id="title"></h1><div class="meta" id="meta"></div></header>
<main><aside>
  <div class="panel"><h2>Zoom into context</h2><input id="context" class="search" placeholder="text or /regular expression/i"></div>
  <div class="panel"><h2>Remove context</h2><input id="remove" class="search" placeholder="text or /regular expression/i"><label class="term"><input id="hideMarkup" type="checkbox"> Show visible HTML text only</label></div>
  <div class="panel" id="displayPanel"><h2>Display</h2><label class="term"><input id="prettyJson" type="checkbox" checked> Pretty-print JSON records</label></div>
  <div class="panel"><h2>Matched terms</h2><div id="terms" class="terms"></div></div>
  <div class="panel" id="contextPanel"><h2>Contexts within terms</h2><div id="termContexts" class="terms"><div class="hint">Select a matched term to inspect its lexical context groups.</div></div></div>
  <div class="panel" id="facetPanel"><h2>Semantic branches</h2><p class="hint">WordNet senses that led from your query to matched terms. Select one to show findings reached through that sense.</p><div id="facets" class="terms"></div></div>
  <div class="panel"><button id="export">Download filter JSON</button><button id="reset" class="secondary">Reset</button></div>
</aside><section id="results"><div class="status" id="status"></div><div id="cards"></div></section></main>
<script>const DATA=__CLUSTERGREP_DATA__;
const $=id=>document.getElementById(id),selected=new Set(),selectedFacets=new Set(),selectedContexts=new Set(),excluded=new Set();
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const strip=s=>{const d=document.createElement('div');d.innerHTML=s;return d.textContent||''};
const regexEsc=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
function highlight(text,matches){const words=[...new Set(matches.map(m=>m.matched).filter(Boolean))].sort((a,b)=>b.length-a.length);if(!words.length)return esc(text);let re;try{re=new RegExp(words.map(regexEsc).join('|'),'giu')}catch(e){re=new RegExp(words.map(regexEsc).join('|'),'gi')}let out='',last=0;for(const hit of text.matchAll(re)){out+=esc(text.slice(last,hit.index))+`<mark>${esc(hit[0])}</mark>`;last=hit.index+hit[0].length}return out+esc(text.slice(last))}
const counts=new Map(Object.entries(DATA.metadata.term_counts||{}));if(!counts.size)for(const r of DATA.records)for(const m of r.matches)counts.set(m.term,(counts.get(m.term)||0)+1);
const facetCounts=new Map();for(const r of DATA.records)for(const f of r.facets||[])facetCounts.set(f,(facetCounts.get(f)||0)+1);
$('title').textContent=DATA.metadata.query;$('meta').textContent=`${DATA.metadata.lines_matched} matching lines · ${DATA.records.length} embedded · ${DATA.metadata.backend} · threshold ${DATA.metadata.threshold}`;
for(const [term,count] of [...counts].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]))){const l=document.createElement('label');l.className='term';l.innerHTML=`<input type="checkbox" value="${esc(term)}"> ${esc(term)} <span>${count}</span>`;l.querySelector('input').onchange=e=>{if(e.target.checked)selected.add(term);else{selected.delete(term);for(const id of [...selectedContexts])if(id.startsWith(term+':'))selectedContexts.delete(id)}renderContexts();render()};$('terms').appendChild(l)}
if(!facetCounts.size)$('facetPanel').hidden=true;for(const [facet,count] of [...facetCounts].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]))){const l=document.createElement('label');l.className='term';l.innerHTML=`<input type="checkbox" value="${esc(facet)}"> ${esc(facet)} <span>${count}</span>`;l.querySelector('input').onchange=e=>{e.target.checked?selectedFacets.add(facet):selectedFacets.delete(facet);render()};$('facets').appendChild(l)}
const hasJson=DATA.records.some(r=>r.json!==undefined);$('displayPanel').hidden=!hasJson;
const contextMembers=new Map();for(const groups of Object.values(DATA.metadata.contexts||{}))for(const group of groups)contextMembers.set(group.id,new Set(group.members));
function renderContexts(){const box=$('termContexts');box.innerHTML='';const terms=[...selected].filter(t=>(DATA.metadata.contexts||{})[t]);if(!terms.length){box.innerHTML='<div class="hint">Select a matched term to inspect its lexical context groups.</div>';return}for(const term of terms){const heading=document.createElement('div');heading.className='context-term';heading.textContent=term;box.appendChild(heading);for(const group of DATA.metadata.contexts[term]){const l=document.createElement('label');l.className='term';const sampled=group.members.length;l.title=sampled<group.count?`${sampled} embedded example(s)`:'';l.innerHTML=`<input type="checkbox" value="${esc(group.id)}"${selectedContexts.has(group.id)?' checked':''}${sampled?'':' disabled'}> <span>${esc(group.label)}</span> <span>${group.count}</span>`;l.querySelector('input').onchange=e=>{e.target.checked?selectedContexts.add(group.id):selectedContexts.delete(group.id);render()};box.appendChild(l)}}}
function predicate(id){const q=$(id).value.trim();if(!q)return()=>false;if(q.startsWith('/')&&q.lastIndexOf('/')>0){const end=q.lastIndexOf('/');try{const re=new RegExp(q.slice(1,end),q.slice(end+1));return r=>re.test(r.text)}catch(e){}}const low=q.toLowerCase();return r=>r.text.toLowerCase().includes(low)}
function inSelectedContext(r){if(!selectedContexts.size)return true;for(const id of selectedContexts)if(contextMembers.get(id)?.has(r.id))return true;return false}
function render(){const include=$('context').value.trim()?predicate('context'):()=>true,remove=predicate('remove'),rows=DATA.records.filter(r=>include(r)&&!remove(r)&&(!selected.size||r.matches.some(m=>selected.has(m.term)))&&(!selectedFacets.size||(r.facets||[]).some(f=>selectedFacets.has(f)))&&inSelectedContext(r)&&!r.matches.every(m=>excluded.has(m.term)));$('cards').innerHTML='';$('status').textContent=`Showing ${rows.length} of ${DATA.records.length} embedded findings`;for(const r of rows){const card=document.createElement('article');card.className='record';const asJson=$('prettyJson').checked&&r.json!==undefined;const shown=asJson?JSON.stringify(r.json,null,2):($('hideMarkup').checked?strip(r.text):r.text);const excerpt=!asJson&&r.source_length>r.text.length?' · excerpt from '+r.source_length+' characters':'';card.innerHTML=`<div class="source">${esc(r.file)}:${r.line}${excerpt}${asJson?' · JSON':''}</div><div class="text${asJson?' json':''}">${highlight(shown,r.matches)}</div><div class="badges">${(r.facets||[]).map(f=>`<span class="badge">${esc(f)}</span>`).join('')}${r.matches.map(m=>`<button class="badge" data-term="${esc(m.term)}"><b>${m.distance.toFixed(2)}</b> ${esc(m.term)} ×</button>`).join('')}</div>`;for(const b of card.querySelectorAll('button.badge'))b.onclick=()=>{excluded.add(b.dataset.term);render()};$('cards').appendChild(card)}}
for(const id of ['context','remove'])$(id).oninput=render;for(const id of ['hideMarkup','prettyJson'])$(id).onchange=render;$('reset').onclick=()=>{selected.clear();selectedFacets.clear();selectedContexts.clear();excluded.clear();$('context').value='';$('remove').value='';$('hideMarkup').checked=false;$('prettyJson').checked=true;for(const x of document.querySelectorAll('.terms input'))x.checked=false;renderContexts();render()};
function expression(q){if(!q)return null;if(q.startsWith('/')&&q.lastIndexOf('/')>0){const end=q.lastIndexOf('/'),flags=q.slice(end+1).replace(/[^ims]/g,'');return(flags?'(?'+flags+')':'')+q.slice(1,end)}return q.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}
$('export').onclick=()=>{const base=structuredClone(DATA.metadata.filter||{version:1});base.version=1;base.exclude=base.exclude||{};base.exclude.terms=[...new Set([...(base.exclude.terms||[]),...excluded])].sort();const remove=expression($('remove').value.trim());if(remove)base.exclude.context_regex=[...(base.exclude.context_regex||[]),remove];const include=expression($('context').value.trim());if(include){base.include=base.include||{};base.include.context_regex=[include]}if($('hideMarkup').checked){base.input=base.input||{};base.input.html='visible_text'}const blob=new Blob([JSON.stringify(base,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='clustergrep-filter.json';a.click();URL.revokeObjectURL(a.href)};renderContexts();render();
</script></body></html>'''
