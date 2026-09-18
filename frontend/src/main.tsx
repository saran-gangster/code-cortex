import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { AlertTriangle, Archive, ArrowDownToLine, Check, ChevronDown, CircleHelp, Clock3, CloudOff, Crosshair, Database, Download, Eye, FileJson, Gauge, GitCompare, Info, Layers3, Menu, Radio, RefreshCw, Search, Server, ShieldCheck, SlidersHorizontal, Sparkles, TimerReset, X } from 'lucide-react'
import './styles.css'

type Provenance = 'FIXTURE' | 'CACHED' | 'COMPUTED'
type Mode = 'baseline' | 'fusion'
type Intervention = 'paired' | 'missing' | 'invalid' | 'delayed'
type Frame = { id: string; time: string; sec: number; gap?: boolean; boxes: { label: string; score: number; x: number; y: number; w: number; h: number }[]; altitude: string; velocity: string; attitude: string; state: 'PAIRED' | 'MISSING' | 'INVALID' | 'DELAYED' }

const fixtureFrames: Frame[] = [
  { id: 'frame-0184', time: '00:36.8', sec: 36.8, boxes: [{ label: 'person', score: .91, x: 22, y: 30, w: 12, h: 29 }, { label: 'car', score: .83, x: 59, y: 50, w: 20, h: 16 }], altitude: '38.4 m', velocity: '+1.8 / −0.4 m/s', attitude: '−3.2° pitch · 1.1° roll', state: 'PAIRED' },
  { id: 'frame-0185', time: '00:37.0', sec: 37, boxes: [{ label: 'person', score: .88, x: 23, y: 31, w: 11, h: 28 }, { label: 'car', score: .81, x: 60, y: 50, w: 20, h: 16 }], altitude: '38.3 m', velocity: '+1.7 / −0.5 m/s', attitude: '−3.1° pitch · 1.0° roll', state: 'PAIRED' },
  { id: 'frame-0186', time: '00:37.2', sec: 37.2, boxes: [{ label: 'person', score: .84, x: 24, y: 32, w: 11, h: 27 }, { label: 'car', score: .79, x: 61, y: 51, w: 20, h: 16 }], altitude: '38.2 m', velocity: '+1.6 / −0.6 m/s', attitude: '−3.0° pitch · 1.0° roll', state: 'PAIRED' },
  { id: 'frame-0187', time: '00:37.4', sec: 37.4, gap: true, boxes: [], altitude: '—', velocity: '—', attitude: '—', state: 'MISSING' },
  { id: 'frame-0188', time: '00:38.2', sec: 38.2, boxes: [{ label: 'person', score: .71, x: 27, y: 35, w: 10, h: 24 }, { label: 'car', score: .67, x: 62, y: 52, w: 19, h: 15 }], altitude: '37.8 m', velocity: '+1.2 / −0.8 m/s', attitude: '−2.7° pitch · 1.2° roll', state: 'PAIRED' },
  { id: 'frame-0189', time: '00:38.4', sec: 38.4, boxes: [{ label: 'person', score: .69, x: 27, y: 35, w: 10, h: 24 }, { label: 'car', score: .64, x: 63, y: 52, w: 19, h: 15 }], altitude: '37.7 m', velocity: '+1.1 / −0.8 m/s', attitude: '−2.8° pitch · 1.2° roll', state: 'PAIRED' },
]

const staticMetrics = [
  { name: 'Dual-GPU Lightning gate', status: 'MEASURED', note: 'Shared initialization + eight-frame schedule · not a benchmark', rows: [['E1 / T4 #0', '40 updates'], ['E2 / T4 #1', '40 updates'], ['FiLM projection', '0 → 120.735']] },
  { name: 'AU-AIR matched holdout', status: 'UNAVAILABLE', note: 'No frozen baseline/fusion benchmark has been executed', rows: [['mAP50', '—'], ['Recall', '—'], ['Latency p95', '—']] },
  { name: 'External RGB / VisDrone', status: 'PLANNED', note: 'Custom mapped-class protocol is specified, not executed', rows: [['mAP50', '—'], ['Classes', '7 mapped'], ['State', 'unavailable']] },
]

type EvaluationReport = {
  artifact_kind?: unknown
  partition?: unknown
  model_id?: unknown
  checkpoint_id?: unknown
  config?: Record<string, unknown>
  metrics?: Record<string, unknown>
}
type ReportsResponse = { reports?: unknown }
const API_BASE = ((import.meta.env.VITE_API_URL as string | undefined) ?? 'http://127.0.0.1:8000').replace(/\/$/, '')
function textValue(value: unknown, fallback = '—'): string { return typeof value === 'string' && value.trim() ? value : fallback }
function numberValue(value: unknown, digits = 3): string { return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—' }
function firstValue(source: EvaluationReport, paths: string[]): unknown {
  for (const path of paths) {
    let current: unknown = source
    for (const part of path.split('.')) {
      if (!current || typeof current !== 'object') { current = undefined; break }
      current = (current as Record<string, unknown>)[part]
    }
    if (current !== undefined && current !== null) return current
  }
  return undefined
}
function shortHash(value: unknown): string { const text = textValue(value); return text === '—' ? text : text.slice(0, 12) }
function reportMetric(report: EvaluationReport, paths: string[], digits = 3): string { return numberValue(firstValue(report, paths), digits) }

function Badge({ children, tone = 'neutral' }: { children: React.ReactNode; tone?: string }) { return <span className={`badge badge-${tone}`}>{children}</span> }
function SectionLabel({ children, icon: Icon }: { children: React.ReactNode; icon?: React.ElementType }) { return <div className="section-label">{Icon && <Icon size={13} />}{children}</div> }

function App() {
  const [page, setPage] = useState<'console' | 'results' | 'architecture'>('console')
  const [mode, setMode] = useState<Mode>('fusion')
  const [intervention, setIntervention] = useState<Intervention>('paired')
  const [selectedId, setSelectedId] = useState('frame-0185')
  const [drawer, setDrawer] = useState(true)
  const [showChooser, setShowChooser] = useState(false)
  const [recording, setRecording] = useState('AU-AIR / harbour loop 07')
  const [streamState, setStreamState] = useState<'ready' | 'loading' | 'error' | 'empty'>('ready')
  const [decision, setDecision] = useState('')
  const [comment, setComment] = useState('')
  const [saved, setSaved] = useState(false)
  const [apiNote, setApiNote] = useState('')

  useEffect(() => {
    fetch(`${API_BASE}/health`).then(r => { if (!r.ok) throw new Error('health') }).then(() => setApiNote('API connected')).catch(() => setApiNote('API unavailable · fixture retained'))
  }, [])

  const selected = useMemo(() => fixtureFrames.find(f => f.id === selectedId) ?? fixtureFrames[1], [selectedId])
  const effectiveState = intervention === 'paired' ? selected.state : intervention === 'missing' ? 'MISSING' : intervention === 'invalid' ? 'INVALID' : 'DELAYED'
  const provenance: Provenance = mode === 'fusion' && intervention === 'paired' ? 'CACHED' : 'FIXTURE'

  function choose(name: string, state: typeof streamState) { setRecording(name); setStreamState(state); setShowChooser(false); setSaved(false) }
  function exportReview() {
    const payload = { recording, frameId: selected.id, model: mode, intervention, decision: decision || 'UNDECIDED', comment, provenance, exportedAt: new Date().toISOString() }
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = `aeroguard-review-${selected.id}.json`; a.click(); URL.revokeObjectURL(url); setSaved(true)
  }

  return <div className="app-shell">
    <header className="topbar"><div className="brand"><div className="brand-mark"><Crosshair size={19} /></div><div><strong>AEROGUARD</strong><span>FLIGHT-AWARE PERCEPTION</span></div></div><div className="top-actions"><Badge tone={apiNote === 'API connected' ? 'green' : 'amber'}>{apiNote || 'OFFLINE DEMO'}</Badge><button className="icon-btn" aria-label="Open architecture help" title="Open architecture help" onClick={() => setPage('architecture')}><CircleHelp size={18} /></button><div className="avatar" aria-label="Workspace owner">SG</div></div></header>
    <div className="layout"><aside className="sidebar"><div className="side-kicker">REVIEW CONSOLE</div><nav aria-label="Primary navigation"><button className={page === 'console' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('console')}><Eye size={17} />Live review <span className="nav-hot">01</span></button><button className={page === 'results' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('results')}><Gauge size={17} />Results & evidence</button><button className={page === 'architecture' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('architecture')}><Layers3 size={17} />Architecture</button></nav><div className="side-foot"><div className="online-dot" />Local workspace<br /><small>Protocol v0.1 · no safety certification</small></div></aside>
      <main className="main">{page === 'console' ? <><div className="page-head"><div><div className="eyebrow">FLIGHT REVIEW / 01</div><h1>Observe before you decide.</h1><p>Inspect the same aerial observation with and without flight-state conditioning.</p></div><button className="button secondary" onClick={() => setShowChooser(true)}><Archive size={15} />Change recording <ChevronDown size={15} /></button></div>
        <div className="record-strip"><div className="record-icon"><Radio size={17} /></div><div><strong>{recording}</strong><span>Replay · 5 Hz source · 6 frames in view</span></div><Badge tone={provenance === 'CACHED' ? 'cyan' : 'amber'}>{provenance}</Badge><button className="strip-action" onClick={() => setShowChooser(true)} aria-label="Open recording chooser"><Menu size={18} /></button></div>
        {streamState === 'error' && <div className="state-banner error"><AlertTriangle size={18} /><div><strong>Replay could not be loaded</strong><span>Fixture recovery is available; choose the demo recording again.</span></div><button onClick={() => choose('AU-AIR / harbour loop 07', 'ready')}><RefreshCw size={15} />Recover fixture</button></div>}
        {streamState === 'empty' && <div className="empty-state"><Archive size={27} /><h2>No frames in this recording</h2><p>There is no evidence to review in the selected stream.</p><button className="button secondary" onClick={() => choose('AU-AIR / harbour loop 07', 'ready')}>Load demo recording</button></div>}
        {streamState === 'loading' && <div className="loading-card"><RefreshCw className="spin" size={22} />Loading replay manifest…</div>}
        {streamState === 'ready' && <><div className="compare-toolbar"><div className="toggle-label"><GitCompare size={16} /><strong>Comparison lens</strong><span>same frame / same protocol</span></div><div className="segmented" role="tablist" aria-label="Model comparison"><button className={mode === 'baseline' ? 'selected' : ''} onClick={() => setMode('baseline')} role="tab" aria-selected={mode === 'baseline'}>Vision-only baseline</button><button className={mode === 'fusion' ? 'selected' : ''} onClick={() => setMode('fusion')} role="tab" aria-selected={mode === 'fusion'}>Flight-state fusion</button></div><Badge tone={provenance === 'CACHED' ? 'cyan' : 'amber'}>{provenance}</Badge></div>
          <div className="workspace-grid"><section className="frame-card"><div className="card-top"><SectionLabel icon={Crosshair}>AERIAL FRAME · {selected.id}</SectionLabel><div className="card-top-right"><span className="frame-time"><Clock3 size={13} />{selected.time}</span><Badge tone={effectiveState === 'PAIRED' ? 'green' : 'red'}>{effectiveState}</Badge></div></div><div className="frame-stage"><svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`Aerial frame ${selected.id} with ${selected.boxes.length} fixture detections`}><defs><linearGradient id="sky" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#d5c5a7" /><stop offset=".4" stopColor="#7a836f" /><stop offset="1" stopColor="#27382f" /></linearGradient><pattern id="grid" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M 6 0 L 0 0 0 6" fill="none" stroke="#e4e9db" strokeOpacity=".13" strokeWidth=".2" /></pattern></defs><rect width="100" height="100" fill="url(#sky)" /><path d="M0,72 L23,53 L43,69 L61,44 L100,57 L100,100 L0,100Z" fill="#324839" opacity=".78" /><path d="M0 76 L100 48 M0 88 L100 62 M31 0 L24 100 M68 0 L73 100" stroke="#e5d8bd" strokeOpacity=".32" strokeWidth="1.2" /><rect width="100" height="100" fill="url(#grid)" />{selected.boxes.map((box, i) => <g key={box.label}><rect x={box.x} y={box.y} width={box.w} height={box.h} fill="none" stroke={box.label === 'person' ? '#ffbf69' : '#6ee7d8'} strokeWidth=".9" /><rect x={box.x} y={box.y - 4.8} width={Math.max(13, box.label.length * 3.1)} height="4.8" fill={box.label === 'person' ? '#ffbf69' : '#6ee7d8'} /><text x={box.x + 1} y={box.y - 1.4} fontSize="3.1" fontWeight="700" fill="#10201e">{box.label} {Math.round(box.score * 100)}%</text></g>)}</svg><div className="frame-overlay"><span>1920 × 1080</span><span>north-up unavailable</span></div>{selected.gap && <div className="frame-gap"><TimerReset size={23} /><strong>0.8s source gap</strong><span>No frame emitted; no boxes fabricated.</span></div>}</div><div className="legend"><span><i className="dot person" />person</span><span><i className="dot car" />car</span><span className="legend-note"><Info size={13} />fixture scores validate workflow only; they are not model evidence</span></div></section>
            <aside className="state-card"><div className="card-top"><SectionLabel icon={Gauge}>FLIGHT STATE</SectionLabel><span className="live-dot">LIVE REPLAY</span></div><div className="state-hero"><span>INPUT STATUS</span><strong className={effectiveState === 'PAIRED' ? 'state-ok' : 'state-bad'}>{effectiveState === 'PAIRED' ? 'PAIRED' : 'INPUT DEGRADED'}</strong><small>{effectiveState === 'PAIRED' ? 'timestamp-aligned state available' : 'model keeps review open; no safe claim made'}</small></div><div className="metric-list"><div><span>Altitude</span><strong>{effectiveState === 'PAIRED' ? selected.altitude : 'Unavailable'}</strong></div><div><span>Velocity XY</span><strong>{effectiveState === 'PAIRED' ? selected.velocity : 'Unavailable'}</strong></div><div><span>Attitude</span><strong>{effectiveState === 'PAIRED' ? selected.attitude : 'Unavailable'}</strong></div></div><div className="model-line"><span>Displayed contract</span><strong>{mode === 'fusion' ? 'UI fixture · E2' : 'UI fixture · E1'}</strong><Badge tone={provenance === 'CACHED' ? 'cyan' : 'amber'}>{provenance}</Badge></div></aside>
          </div>
          <section className="timeline-card"><div className="card-top"><SectionLabel icon={Clock3}>REPLAY TIMELINE</SectionLabel><span className="timeline-hint">Select a frame to inspect evidence</span></div><div className="timeline"><div className="track" />{fixtureFrames.map(f => <button key={f.id} className={`timeline-node ${f.id === selected.id ? 'selected' : ''} ${f.gap ? 'gap' : ''}`} style={{ left: `${(f.sec - 36.8) / 1.8 * 86 + 7}%` }} onClick={() => setSelectedId(f.id)} aria-label={`Select ${f.id} at ${f.time}`}>{f.gap ? <TimerReset size={14} /> : <span />}</button>)}</div><div className="time-labels"><span>00:36.8</span><span className="event-label"><AlertTriangle size={13} /> 00:37.4 · metadata gap</span><span>00:38.4</span></div></section>
          <section className="intervention-card"><div><SectionLabel icon={SlidersHorizontal}>INPUT INTERVENTION</SectionLabel><h2>Stress the pairing, keep the claim honest.</h2><p>These controls alter the displayed input state and provenance. They do not create a new metric.</p></div><div className="intervention-grid">{([['paired', 'Paired state', 'Use timestamp-matched telemetry'], ['missing', 'Missing state', 'Mask the state vector'], ['invalid', 'Invalid state', 'Inject non-finite metadata'], ['delayed', 'Injected delay', 'Shift state by 800 ms']] as const).map(([key, title, sub]) => <button key={key} className={`intervention ${intervention === key ? 'chosen' : ''}`} onClick={() => setIntervention(key)}><span className="radio-dot" /><div><strong>{title}</strong><small>{sub}</small></div></button>)}</div></section>
        </>}
      </> : page === 'results' ? <Results /> : <Architecture />}</main></div>
    {drawer && page === 'console' && streamState === 'ready' && <EvidenceDrawer selected={selected} mode={mode} provenance={provenance} decision={decision} setDecision={setDecision} comment={comment} setComment={setComment} saved={saved} onExport={exportReview} onClose={() => setDrawer(false)} />}
    {!drawer && page === 'console' && <button className="drawer-peek" onClick={() => setDrawer(true)}><FileJson size={16} />Evidence</button>}
    {showChooser && <div className="modal-backdrop" role="presentation" onClick={() => setShowChooser(false)}><div className="chooser-modal" role="dialog" aria-modal="true" aria-labelledby="chooser-title" onClick={e => e.stopPropagation()}><div className="modal-head"><div><div className="eyebrow">REPLAY SOURCE</div><h2 id="chooser-title">Choose a recording</h2></div><button className="icon-btn" onClick={() => setShowChooser(false)} aria-label="Close"><X size={18} /></button></div><button className="record-option" onClick={() => choose('AU-AIR / harbour loop 07', 'ready')}><div className="option-mark green"><Check size={16} /></div><div><strong>AU-AIR / harbour loop 07</strong><span>Fixture · 6 frames · paired metadata + known gap</span></div><Badge tone="amber">FIXTURE</Badge></button><button className="record-option" onClick={() => { setShowChooser(false); setStreamState('loading'); setTimeout(() => setStreamState('empty'), 650) }}><div className="option-mark"><Archive size={16} /></div><div><strong>Empty stream sentinel</strong><span>State handling demo · 0 frames</span></div><Badge>EMPTY</Badge></button><button className="record-option" onClick={() => { setShowChooser(false); setStreamState('loading'); setTimeout(() => setStreamState('error'), 650) }}><div className="option-mark red"><CloudOff size={16} /></div><div><strong>Unavailable API replay</strong><span>Failure recovery demo · no fixture substitution</span></div><Badge tone="red">ERROR</Badge></button></div></div>}
  </div>
}

function EvidenceDrawer({ selected, mode, provenance, decision, setDecision, comment, setComment, saved, onExport, onClose }: { selected: Frame; mode: Mode; provenance: Provenance; decision: string; setDecision: (v: string) => void; comment: string; setComment: (v: string) => void; saved: boolean; onExport: () => void; onClose: () => void }) { return <aside className="evidence-drawer"><div className="drawer-head"><div><div className="eyebrow">EVIDENCE DRAWER</div><h2>Frame evidence</h2></div><button className="icon-btn" onClick={onClose} aria-label="Close evidence drawer"><X size={18} /></button></div><div className="drawer-scroll"><div className="provenance-box"><div><span>Provenance</span><strong>{provenance}</strong></div><p>{provenance === 'CACHED' ? 'Recorded output from an approved replay bundle.' : 'Fixture state for UI and workflow validation.'}</p></div><dl className="evidence-list"><div><dt>Frame ID</dt><dd>{selected.id}</dd></div><div><dt>Source time</dt><dd>{selected.time}</dd></div><div><dt>Model / protocol</dt><dd>{mode === 'fusion' ? 'FCOS + FiLM · E2' : 'FCOS · E1'}<small>not a trained-result claim</small></dd></div><div><dt>Image space</dt><dd>1920 × 1080 · XYXY boxes</dd></div><div><dt>Quality flags</dt><dd>{selected.gap ? 'timestamp gap' : 'none in fixture'}</dd></div></dl><div className="drawer-divider" /><SectionLabel icon={ShieldCheck}>OPERATOR REVIEW</SectionLabel><label className="field-label" htmlFor="decision">Decision</label><select id="decision" value={decision} onChange={e => setDecision(e.target.value)}><option value="">Choose a decision…</option><option value="accept">Accept evidence</option><option value="needs-review">Needs further review</option><option value="reject">Reject evidence</option></select><label className="field-label" htmlFor="comment">Comment <span>optional</span></label><textarea id="comment" value={comment} onChange={e => setComment(e.target.value)} placeholder="Describe what the next reviewer should inspect…" rows={4} /><button className="button primary full" onClick={onExport}><Download size={16} />{saved ? 'Review exported' : 'Save & export review'} </button><p className="microcopy"><Info size={12} />Exports only the declared evidence and your review. No risk label is inferred.</p></div></aside> }

function DevelopmentReportCard({ report }: { report: EvaluationReport }) {
  const humanRecall = reportMetric(report, ['metrics.human_recall_at_fixed_operating_point', 'metrics.pooled.per_class.1.recall', 'metrics.pooled.per_class.Human.recall', 'metrics.human_recall', 'metrics.fixed_operating_point.human_recall'])
  const latencyP50 = reportMetric(report, ['metrics.latency.p50_ms', 'metrics.latency_p50_ms', 'metrics.latency.p50', 'config.latency_p50_ms'])
  const latencyP95 = reportMetric(report, ['metrics.latency.p95_ms', 'metrics.latency_p95_ms', 'metrics.latency.p95', 'config.latency_p95_ms'])
  return <section className="result-panel development-report"><div className="result-head"><div><SectionLabel icon={Database}>DEVELOPMENT EVIDENCE</SectionLabel><h2>{textValue(report.model_id)}</h2></div><Badge tone="amber">NON-FINAL / DEVELOPMENT</Badge></div><p>{textValue(report.artifact_kind, 'Evaluation report')} · {textValue(report.partition, 'development')} · benchmark claim disabled</p><div className="metric-rows"><div><span>Partition</span><strong>{textValue(report.partition)}</strong></div><div><span>AP50</span><strong>{reportMetric(report, ['metrics.pooled.ap50', 'metrics.ap50'])}</strong></div><div><span>AP50:95</span><strong>{reportMetric(report, ['metrics.pooled.ap50_95', 'metrics.pooled.ap50:95'])}</strong></div><div><span>Fixed-point recall</span><strong>{reportMetric(report, ['metrics.pooled.recall', 'metrics.fixed_operating_point.recall', 'metrics.fixed_point_recall'])}</strong></div><div><span>Human recall</span><strong>{humanRecall}</strong></div><div><span>Latency p50 / p95</span><strong>{latencyP50} / {latencyP95} ms</strong></div><div><span>Checkpoint</span><strong title={textValue(report.checkpoint_id)}>{shortHash(report.checkpoint_id)}</strong></div><div><span>Initial weight origin</span><strong>{textValue(firstValue(report, ['config.initial_weights_origin', 'config.initial_weight_origin', 'config.weights_origin']))}</strong></div></div></section>
}

function PipelineFixtureCard({ report }: { report: EvaluationReport }) { return <section className="result-panel"><div className="result-head"><div><SectionLabel icon={Database}>PIPELINE FIXTURE</SectionLabel><h2>{textValue(report.model_id, 'Synthetic evaluator')}</h2></div><Badge tone="amber">PIPELINE FIXTURE</Badge></div><p>Synthetic boxes validate evaluator behavior only. They are never shown as model performance.</p><div className="metric-rows"><div><span>Partition</span><strong>synthetic_fixture</strong></div><div><span>Performance claim</span><strong>none</strong></div></div></section> }

function Results() {
  const [reports, setReports] = useState<EvaluationReport[]>([])
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  useEffect(() => {
    let active = true
    fetch(`${API_BASE}/reports`).then(response => { if (!response.ok) throw new Error(`reports:${response.status}`); return response.json() as Promise<ReportsResponse> }).then(payload => {
      if (!active) return
      const rows = Array.isArray(payload.reports) ? payload.reports.filter((item): item is EvaluationReport => Boolean(item && typeof item === 'object')) : []
      setReports(rows); setStatus('ready')
    }).catch(() => { if (active) setStatus('error') })
    return () => { active = false }
  }, [])
  const developmentReports = reports.filter(report => report.partition === 'development' && typeof report.model_id === 'string')
  const fixtureReports = reports.filter(report => report.partition === 'synthetic_fixture')
  const showStatic = status !== 'ready' || developmentReports.length === 0
  return <div className="content-page"><div className="page-head"><div><div className="eyebrow">RESULTS / EVIDENCE REGISTER</div><h1>Measured, missing, planned.</h1><p>Every result is labelled by what exists in this workspace today.</p></div><Badge tone="amber">NO FABRICATED METRICS</Badge></div>
    {status === 'loading' && <div className="notice"><RefreshCw className="spin" size={18} /><div><strong>Loading evaluation reports</strong><span>Checking the connected API for schema-validated development evidence.</span></div></div>}
    {status === 'error' && <div className="notice"><AlertTriangle size={18} /><div><strong>Development reports unavailable</strong><span>The API could not be reached. Showing the documented unavailable/planned cards; no metrics were inferred.</span></div></div>}
    {status === 'ready' && developmentReports.length === 0 && <div className="notice"><Info size={18} /><div><strong>No development reports are present</strong><span>Showing the documented unavailable/planned cards. Synthetic pipeline fixtures are never treated as model performance.</span></div></div>}
    {developmentReports.length > 0 && <div className="notice"><Info size={18} /><div><strong>Development evidence found</strong><span>These are real schema-validated development reports. They are visible for review and are explicitly non-final.</span></div></div>}
    {developmentReports.length > 0 && <div className="results-grid">{developmentReports.map((report, index) => <DevelopmentReportCard key={`${textValue(report.model_id)}-${index}`} report={report} />)}</div>}
    {fixtureReports.length > 0 && <div className="results-grid" style={{ marginTop: 14 }}>{fixtureReports.map((report, index) => <PipelineFixtureCard key={`fixture-${index}`} report={report} />)}</div>}
    {showStatic && <div className="results-grid">{staticMetrics.map(m => <section className="result-panel" key={m.name}><div className="result-head"><div><SectionLabel icon={Database}>{m.status}</SectionLabel><h2>{m.name}</h2></div><Badge tone={m.status === 'MEASURED' ? 'green' : m.status === 'UNAVAILABLE' ? 'red' : 'amber'}>{m.status}</Badge></div><p>{m.note}</p><div className="metric-rows">{m.rows.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div></section>)}</div>}
    <section className="limitations"><SectionLabel icon={AlertTriangle}>LIMITATION REGISTER</SectionLabel><div className="limit-row"><span className="number">01</span><div><strong>Fusion benefit is a hypothesis</strong><p>A positive result requires matched session-held-out evaluation. The UI does not imply improvement from a replay comparison.</p></div></div><div className="limit-row"><span className="number">02</span><div><strong>Supervised review only</strong><p>No autonomous-flight decision, collision distance, or safety certification is displayed or inferred.</p></div></div></section></div>
}

function Architecture() { const nodes = [['01', 'Source recordings', 'AU-AIR / VisDrone', 'current'], ['02', 'Validation + pairing', 'schema · timestamps · splits', 'current'], ['03', 'Model gate + FiLM', 'FCOS smoke · tested conditioning', 'current'], ['04', 'Inference API', 'FastAPI · bounded /infer · replay', 'current'], ['05', 'Review console', 'this frontend · fixture replay', 'current'], ['06', 'Storage + monitoring', 'object store · Postgres · drift', 'planned']] as const; return <div className="content-page"><div className="page-head"><div><div className="eyebrow">SYSTEM / ARCHITECTURE</div><h1>One lane, two horizons.</h1><p>What a judge can use now, and what the production path still needs.</p></div><div className="architecture-legend"><span><i className="line current" />Current / evidenced</span><span><i className="line planned" />Planned / not implemented</span></div></div><div className="architecture-map">{nodes.map(([num, title, sub, kind], i) => <div className="arch-step" key={num}><div className={`arch-node ${kind}`}><span>{num}</span><strong>{title}</strong><small>{sub}</small><Badge tone={kind === 'current' ? 'green' : 'amber'}>{kind}</Badge></div>{i < nodes.length - 1 && <div className="arch-arrow">↓</div>}</div>)}</div><div className="architecture-foot"><div><SectionLabel icon={Server}>SCALE PATH</SectionLabel><p>At production scale, the plan separates CPU web service from a bounded GPU inference worker, stores replay media in object storage, and keeps review records in PostgreSQL. Those pieces remain planned until their artifacts exist.</p></div><div className="stack-tags"><Badge>React + Vite</Badge><Badge>FastAPI current</Badge><Badge>PyTorch current</Badge><Badge>Offline-first</Badge></div></div></div> }

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
