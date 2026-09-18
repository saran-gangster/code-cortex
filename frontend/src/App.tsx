import {
  Activity, AlertTriangle, Archive, ArrowRight, Box, Check, ChevronDown, CircleHelp, Clock3, CloudOff,
  Download, Eye, FileCheck2, Gauge, GitCompare, HardDrive, Info, Layers3, LoaderCircle, Menu, Radio,
  RefreshCw, Save, Server, ShieldCheck, SlidersHorizontal, X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, API_BASE, errorMessage } from './api'
import { documentedReports, fixtureDetections, OFFLINE_RUN_ID, offlineFrames, offlineModels, offlineRun } from './fixtures'
import type {
  CapabilityResponse, Detection, EvaluationReport, HealthResponse, InferenceRecord, Intervention, ModelSummary,
  ModelView, Page, ReadinessResponse, Review, ReviewDecision, RunSummary, ServiceMode,
} from './types'

const pageFromHash = (): Page => {
  const value = window.location.hash.replace(/^#\/?/, '')
  return value === 'results' || value === 'architecture' ? value : 'live'
}

const modelKind = (id: string): 'e1' | 'e2' | 'other' => {
  const value = id.toLowerCase()
  if (/(^|[-_])e1([-_]|$)|rgb[-_]?masked|vision/.test(value)) return 'e1'
  if (/(^|[-_])e2([-_]|$)|film|fusion/.test(value)) return 'e2'
  return 'other'
}

const sourceLabel = (source: InferenceRecord['prediction_source']) => source === 'computed' ? 'COMPUTED' : source === 'cached' ? 'CACHED' : 'FIXTURE'
const sourceTone = (source: InferenceRecord['prediction_source']) => source === 'computed' ? 'green' : source === 'cached' ? 'cyan' : 'amber'
const shortHash = (value: string) => value === '0'.repeat(64) ? 'not applicable' : `${value.slice(0, 12)}…`
const formatTime = (milliseconds: number | null) => milliseconds == null ? 'Not supplied' : `${Math.floor(milliseconds / 60_000).toString().padStart(2, '0')}:${Math.floor((milliseconds % 60_000) / 1000).toString().padStart(2, '0')}.${Math.floor(milliseconds % 1000 / 100)}`
const formatMetric = (value: number | null, digits = 4) => value == null ? '—' : value.toFixed(digits)

function pathNumber(source: Record<string, unknown>, path: string): number | null {
  let current: unknown = source
  for (const key of path.split('.')) {
    if (!current || typeof current !== 'object') return null
    current = (current as Record<string, unknown>)[key]
  }
  return typeof current === 'number' && Number.isFinite(current) ? current : null
}

function reportMetrics(report: EvaluationReport) {
  return {
    ap50: pathNumber(report.metrics, 'pooled.ap50'),
    recall: pathNumber(report.metrics, 'operating_point_sweep.best_f1_point.recall'),
    detectionAccuracy: pathNumber(report.metrics, 'operating_point_sweep.best_f1_point.detection_accuracy'),
    frames: pathNumber(report.metrics, 'evaluated_frame_count'),
    latency: pathNumber(report.metrics, 'latency.p95_ms'),
  }
}

function Badge({ children, tone = 'neutral' }: { children: React.ReactNode; tone?: 'neutral' | 'green' | 'cyan' | 'amber' | 'red' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

function SectionLabel({ children, icon: Icon }: { children: React.ReactNode; icon?: React.ElementType }) {
  return <div className="section-label">{Icon && <Icon aria-hidden size={13} />}{children}</div>
}

function ServiceBadge({ mode, health }: { mode: ServiceMode; health: HealthResponse | null }) {
  if (mode === 'loading') return <Badge><LoaderCircle className="spin" size={11} /> CONNECTING</Badge>
  if (mode === 'offline') return <Badge tone="amber"><CloudOff size={11} /> OFFLINE DEMO</Badge>
  if (mode === 'degraded') return <Badge tone="amber"><AlertTriangle size={11} /> API DEGRADED</Badge>
  return <Badge tone="green"><span className="status-dot" /> API {health?.version ?? 'CONNECTED'}</Badge>
}

type Snapshot = {
  health: HealthResponse | null
  readiness: ReadinessResponse | null
  capabilities: CapabilityResponse | null
  runs: RunSummary[]
  models: ModelSummary[]
  reports: EvaluationReport[]
  reviews: Review[]
  reportsFallback: boolean
}

const initialSnapshot: Snapshot = {
  health: null, readiness: null, capabilities: null, runs: [], models: [], reports: [], reviews: [], reportsFallback: false,
}

export default function App() {
  const [page, setPage] = useState<Page>(pageFromHash)
  const [serviceMode, setServiceMode] = useState<ServiceMode>('loading')
  const [snapshot, setSnapshot] = useState<Snapshot>(initialSnapshot)
  const [workspaceError, setWorkspaceError] = useState('')
  const [activeRunId, setActiveRunId] = useState('')
  const [frames, setFrames] = useState<InferenceRecord[]>([])
  const [framesState, setFramesState] = useState<'idle' | 'loading' | 'ready' | 'empty' | 'error'>('idle')
  const [framesError, setFramesError] = useState('')
  const [selectedFrameId, setSelectedFrameId] = useState('')
  const [selectedFrame, setSelectedFrame] = useState<InferenceRecord | null>(null)
  const [frameDetailWarning, setFrameDetailWarning] = useState('')
  const [modelView, setModelView] = useState<ModelView>('split')
  const [e1ModelId, setE1ModelId] = useState('')
  const [e2ModelId, setE2ModelId] = useState('')
  const [modelDetailNote, setModelDetailNote] = useState('')
  const [intervention, setIntervention] = useState<Intervention>('recorded')
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [chooserOpen, setChooserOpen] = useState(false)
  const [decision, setDecision] = useState<ReviewDecision | ''>('')
  const [comment, setComment] = useState('')
  const [reviewState, setReviewState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [reviewMessage, setReviewMessage] = useState('')
  const [selectedReview, setSelectedReview] = useState<Review | null>(null)
  const [selectedReport, setSelectedReport] = useState<EvaluationReport | null>(null)
  const [reportDetailState, setReportDetailState] = useState<'idle' | 'loading' | 'error'>('idle')
  const [reportDetailError, setReportDetailError] = useState('')
  const chooserCloseRef = useRef<HTMLButtonElement>(null)

  const navigate = useCallback((next: Page) => {
    window.location.hash = `/${next}`
    setPage(next)
    setDrawerOpen(false)
  }, [])

  useEffect(() => {
    const onHashChange = () => setPage(pageFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const loadWorkspace = useCallback(async () => {
    setServiceMode('loading')
    setWorkspaceError('')
    const [health, readiness, capabilities, runs, models, reports, reviews] = await Promise.allSettled([
      api.health(), api.readiness(), api.capabilities(), api.runs(), api.models(), api.reports(), api.reviews(),
    ])
    if (health.status === 'rejected') {
      setServiceMode('offline')
      setWorkspaceError(errorMessage(health.reason))
      setSnapshot({ health: null, readiness: null, capabilities: null, runs: [offlineRun], models: offlineModels, reports: documentedReports, reviews: [], reportsFallback: true })
      setActiveRunId(OFFLINE_RUN_ID)
      return
    }
    const ready = readiness.status === 'fulfilled' ? readiness.value : null
    const failedResources = [runs, models, reports, reviews, capabilities, readiness].filter((result) => result.status === 'rejected').length
    setServiceMode(failedResources || ready?.status === 'degraded' ? 'degraded' : 'online')
    const apiRuns = runs.status === 'fulfilled' ? runs.value : []
    const apiModels = models.status === 'fulfilled' ? models.value : []
    const apiReports = reports.status === 'fulfilled' ? reports.value : []
    setSnapshot({
      health: health.value, readiness: ready,
      capabilities: capabilities.status === 'fulfilled' ? capabilities.value : null,
      runs: apiRuns, models: apiModels, reports: apiReports.length ? apiReports : documentedReports,
      reviews: reviews.status === 'fulfilled' ? reviews.value : [], reportsFallback: reports.status === 'rejected' || !apiReports.length,
    })
    setWorkspaceError(failedResources ? `${failedResources} workspace resource${failedResources === 1 ? '' : 's'} could not be loaded. Available API data is shown.` : '')
    setActiveRunId((current) => apiRuns.some((run) => run.run_id === current) ? current : (apiRuns[0]?.run_id ?? ''))
  }, [])

  useEffect(() => { void loadWorkspace() }, [loadWorkspace])

  useEffect(() => {
    if (!snapshot.models.length) return
    setE1ModelId((current) => current || snapshot.models.find((model) => modelKind(model.model_id) === 'e1')?.model_id || snapshot.models[0].model_id)
    setE2ModelId((current) => current || snapshot.models.find((model) => modelKind(model.model_id) === 'e2')?.model_id || snapshot.models[1]?.model_id || snapshot.models[0].model_id)
  }, [snapshot.models])

  useEffect(() => {
    let cancelled = false
    setSelectedFrame(null)
    setSelectedFrameId('')
    setFrameDetailWarning('')
    setReviewState('idle')
    setReviewMessage('')
    if (!activeRunId) { setFrames([]); setFramesState('empty'); return }
    if (activeRunId === OFFLINE_RUN_ID) {
      setFrames(offlineFrames); setFramesState('ready'); setSelectedFrameId(offlineFrames[offlineFrames.length - 1]?.frame_id ?? ''); return
    }
    setFramesState('loading')
    setFramesError('')
    api.frames(activeRunId).then((items) => {
      if (cancelled) return
      setFrames(items); setFramesState(items.length ? 'ready' : 'empty'); setSelectedFrameId(items[items.length - 1]?.frame_id ?? '')
    }).catch((error) => {
      if (cancelled) return
      setFrames([]); setFramesState('error'); setFramesError(errorMessage(error))
    })
    return () => { cancelled = true }
  }, [activeRunId])

  useEffect(() => {
    let cancelled = false
    if (!selectedFrameId) { setSelectedFrame(null); return }
    const listRecord = frames.find((frame) => frame.frame_id === selectedFrameId) ?? null
    if (activeRunId === OFFLINE_RUN_ID) { setSelectedFrame(listRecord); return }
    api.frame(activeRunId, selectedFrameId).then((record) => {
      if (!cancelled) { setSelectedFrame(record); setFrameDetailWarning('') }
    }).catch((error) => {
      if (!cancelled) { setSelectedFrame(listRecord); setFrameDetailWarning(`Frame-detail check failed: ${errorMessage(error)}. Using the run manifest record.`) }
    })
    return () => { cancelled = true }
  }, [activeRunId, frames, selectedFrameId])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const editing = target?.matches('input, textarea, select, button')
      if (event.key === 'Escape') { setDrawerOpen(false); setChooserOpen(false); return }
      if (page !== 'live' || editing) return
      if (event.key.toLowerCase() === 'e') { setDrawerOpen((open) => !open); return }
      const index = frames.findIndex((frame) => frame.frame_id === selectedFrameId)
      if (event.key === 'ArrowLeft' && index > 0) setSelectedFrameId(frames[index - 1].frame_id)
      if (event.key === 'ArrowRight' && index >= 0 && index < frames.length - 1) setSelectedFrameId(frames[index + 1].frame_id)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [frames, page, selectedFrameId])

  useEffect(() => { if (chooserOpen) window.setTimeout(() => chooserCloseRef.current?.focus(), 0) }, [chooserOpen])

  const activeRun = useMemo(() => snapshot.runs.find((run) => run.run_id === activeRunId) ?? (activeRunId === OFFLINE_RUN_ID ? offlineRun : null), [activeRunId, snapshot.runs])
  const frameReviews = useMemo(() => snapshot.reviews.filter((review) => review.run_id === activeRunId && review.frame_id === selectedFrameId), [activeRunId, selectedFrameId, snapshot.reviews])
  const availableRuns = useMemo(() => snapshot.runs.some((run) => run.run_id === OFFLINE_RUN_ID) ? snapshot.runs : [...snapshot.runs, offlineRun], [snapshot.runs])

  const inspectModel = async (kind: 'e1' | 'e2', modelId: string) => {
    if (kind === 'e1') setE1ModelId(modelId); else setE2ModelId(modelId)
    setModelDetailNote('')
    if (serviceMode === 'offline') return
    try { const detail = await api.model(modelId); setModelDetailNote(`${detail.model_id} verified against the model detail endpoint.`) }
    catch (error) { setModelDetailNote(`Model detail unavailable: ${errorMessage(error)}`) }
  }

  const inspectReport = async (report: EvaluationReport) => {
    setSelectedReport(report); setReportDetailError('')
    if (report.documentedFallback || serviceMode === 'offline') { setReportDetailState('idle'); return }
    setReportDetailState('loading')
    try { setSelectedReport(await api.report(report.report_id)); setReportDetailState('idle') }
    catch (error) { setReportDetailState('error'); setReportDetailError(errorMessage(error)) }
  }

  const inspectReview = async (review: Review) => {
    if (serviceMode === 'offline') { setSelectedReview(review); return }
    try { setSelectedReview(await api.review(review.review_id)) }
    catch (error) { setReviewMessage(`Review detail unavailable: ${errorMessage(error)}`) }
  }

  const saveReview = async () => {
    if (!selectedFrame || !activeRun || !decision) { setReviewState('error'); setReviewMessage('Choose a decision before saving.'); return }
    if (activeRunId === OFFLINE_RUN_ID) { setReviewState('error'); setReviewMessage('Offline demo decisions cannot be written to the service. Export a local JSON copy instead.'); return }
    setReviewState('saving'); setReviewMessage('Saving to the AeroGuard review store…')
    try {
      const stored = await api.saveReview({ run_id: activeRunId, frame_id: selectedFrame.frame_id, decision, comment })
      const verified = await api.review(stored.review_id)
      setSnapshot((current) => ({ ...current, reviews: [verified, ...current.reviews.filter((item) => item.review_id !== verified.review_id)] }))
      setSelectedReview(verified); setReviewState('saved'); setReviewMessage(`Saved and verified as ${verified.review_id.slice(0, 12)}…`)
    } catch (error) { setReviewState('error'); setReviewMessage(`Review was not saved: ${errorMessage(error)}`) }
  }

  const exportReview = () => {
    if (!selectedFrame || !activeRun) return
    const payload = {
      run_id: activeRun.run_id, frame_id: selectedFrame.frame_id, decision: decision || 'undecided', comment,
      frame_provenance: selectedFrame.prediction_source, model_id: selectedFrame.model_id, exported_at: new Date().toISOString(),
      note: activeRunId === OFFLINE_RUN_ID ? 'Offline fixture workflow export; not measured model evidence.' : 'Local copy of an operator review.',
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `aeroguard-review-${selectedFrame.frame_id}.json`; anchor.click(); URL.revokeObjectURL(url)
  }

  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => navigate('live')} aria-label="AeroGuard live review"><img src="/assets/aeroguard-mark.svg" alt="" /><span><strong>AeroGuard</strong><small>Flight-Aware Perception</small></span></button>
      <div className="top-actions"><span className="protocol-copy">Protocol v0.3-dev<small>No safety certification</small></span><ServiceBadge mode={serviceMode} health={snapshot.health} /><button className="icon-button" onClick={() => navigate('architecture')} aria-label="Open architecture"><CircleHelp size={17} /></button><span className="avatar" aria-label="Workspace owner SG">SG</span></div>
    </header>
    <div className={`app-layout ${drawerOpen ? 'drawer-is-open' : ''}`}>
      <aside className="sidebar"><p className="sidebar-label">Review Console</p><nav aria-label="Primary navigation"><NavButton active={page === 'live'} icon={Eye} onClick={() => navigate('live')}>Live review</NavButton><NavButton active={page === 'results'} icon={Gauge} onClick={() => navigate('results')}>Results & evidence</NavButton><NavButton active={page === 'architecture'} icon={Layers3} onClick={() => navigate('architecture')}>Architecture</NavButton></nav><div className="sidebar-foot"><span><i className={serviceMode === 'offline' ? 'amber-dot' : 'green-dot'} />{serviceMode === 'offline' ? 'Offline fixture workspace' : 'Local API workspace'}</span><small>{API_BASE}<br />No safety certification</small></div></aside>
      <main className="main-content" id="main-content">
        {page === 'live' && <LiveReview serviceMode={serviceMode} workspaceError={workspaceError} activeRun={activeRun} activeRunId={activeRunId} frames={frames} framesState={framesState} framesError={framesError} selectedFrame={selectedFrame} selectedFrameId={selectedFrameId} setSelectedFrameId={setSelectedFrameId} frameDetailWarning={frameDetailWarning} modelView={modelView} setModelView={setModelView} models={snapshot.models} e1ModelId={e1ModelId} e2ModelId={e2ModelId} inspectModel={inspectModel} modelDetailNote={modelDetailNote} intervention={intervention} setIntervention={setIntervention} onOpenChooser={() => setChooserOpen(true)} onOpenEvidence={() => setDrawerOpen(true)} onUseOffline={() => setActiveRunId(OFFLINE_RUN_ID)} onRetry={() => void loadWorkspace()} />}
        {page === 'results' && <ResultsEvidence mode={serviceMode} reports={snapshot.reports} reportsFallback={snapshot.reportsFallback} models={snapshot.models} reviews={snapshot.reviews} error={workspaceError} selectedReport={selectedReport} reportDetailState={reportDetailState} reportDetailError={reportDetailError} onInspectReport={(report) => void inspectReport(report)} onCloseReport={() => setSelectedReport(null)} onRetry={() => void loadWorkspace()} />}
        {page === 'architecture' && <Architecture capabilities={snapshot.capabilities} readiness={snapshot.readiness} mode={serviceMode} />}
      </main>
    </div>
    {page === 'live' && !drawerOpen && selectedFrame && <button className="evidence-peek" onClick={() => setDrawerOpen(true)}><FileCheck2 size={15} /> Evidence <kbd>E</kbd></button>}
    {drawerOpen && <EvidenceDrawer run={activeRun} frame={selectedFrame} intervention={intervention} reviews={frameReviews} decision={decision} setDecision={setDecision} comment={comment} setComment={setComment} reviewState={reviewState} reviewMessage={reviewMessage} selectedReview={selectedReview} onInspectReview={(review) => void inspectReview(review)} onSave={() => void saveReview()} onExport={exportReview} onClose={() => setDrawerOpen(false)} />}
    {chooserOpen && <RecordingChooser runs={availableRuns} selected={activeRunId} closeRef={chooserCloseRef} onChoose={(runId) => { setActiveRunId(runId); setChooserOpen(false) }} onClose={() => setChooserOpen(false)} />}
  </div>
}

function NavButton({ active, icon: Icon, onClick, children }: { active: boolean; icon: React.ElementType; onClick: () => void; children: React.ReactNode }) {
  return <button className={`nav-button ${active ? 'active' : ''}`} onClick={onClick} aria-current={active ? 'page' : undefined}><Icon size={16} /><span>{children}</span></button>
}

type LiveReviewProps = {
  serviceMode: ServiceMode; workspaceError: string; activeRun: RunSummary | null; activeRunId: string; frames: InferenceRecord[]
  framesState: 'idle' | 'loading' | 'ready' | 'empty' | 'error'; framesError: string; selectedFrame: InferenceRecord | null
  selectedFrameId: string; setSelectedFrameId: (id: string) => void; frameDetailWarning: string; modelView: ModelView
  setModelView: (view: ModelView) => void; models: ModelSummary[]; e1ModelId: string; e2ModelId: string
  inspectModel: (kind: 'e1' | 'e2', id: string) => void; modelDetailNote: string; intervention: Intervention
  setIntervention: (value: Intervention) => void; onOpenChooser: () => void; onOpenEvidence: () => void
  onUseOffline: () => void; onRetry: () => void
}

function LiveReview(props: LiveReviewProps) {
  const { serviceMode, workspaceError, activeRun, activeRunId, frames, framesState, framesError, selectedFrame, selectedFrameId, setSelectedFrameId, frameDetailWarning, modelView, setModelView, models, e1ModelId, e2ModelId, inspectModel, modelDetailNote, intervention, setIntervention, onOpenChooser, onOpenEvidence, onUseOffline, onRetry } = props
  const e1Models = models.filter((model) => modelKind(model.model_id) === 'e1')
  const e2Models = models.filter((model) => modelKind(model.model_id) === 'e2')
  const source = selectedFrame?.prediction_source ?? activeRun?.prediction_source ?? 'fixture'
  return <div className="page live-page">
    <header className="page-header compact"><div><p className="eyebrow">Flight Review / 01</p><h1>Live aerial review</h1><p>Compare the same frame and protocol without hiding missing evidence.</p></div><div className="page-actions"><button className="button secondary" onClick={onOpenChooser}><Archive size={15} /> Change recording <ChevronDown size={14} /></button><button className="button evidence-button" onClick={onOpenEvidence}><FileCheck2 size={15} /> Evidence <kbd>E</kbd></button></div></header>
    <div className="record-strip"><span className="record-symbol"><Radio size={16} /></span><div><strong>{activeRun?.run_id ?? 'No recording selected'}</strong><small>{activeRun ? `${activeRun.frame_count} frame${activeRun.frame_count === 1 ? '' : 's'} · ${activeRun.prediction_source} replay` : 'Choose a replay source to begin'}</small></div><Badge tone={sourceTone(source)}>{sourceLabel(source)}</Badge><button className="icon-button" onClick={onOpenChooser} aria-label="Choose recording"><Menu size={17} /></button></div>
    {workspaceError && <div className="inline-notice amber"><AlertTriangle size={17} /><div><strong>{serviceMode === 'offline' ? 'API unavailable — offline demonstration active' : 'Partial workspace response'}</strong><span>{workspaceError}</span></div><button onClick={onRetry}><RefreshCw size={14} /> Retry</button></div>}
    {framesState === 'loading' && <LoadingState label="Loading replay manifest…" />}
    {framesState === 'error' && <ErrorState title="Replay could not be loaded" message={framesError} onRetry={onRetry} secondary={{ label: 'Use offline demo', action: onUseOffline }} />}
    {framesState === 'empty' && <EmptyState onUseOffline={onUseOffline} />}
    {framesState === 'ready' && selectedFrame && <>
      <section className="review-panel"><div className="compare-head"><div><p className="eyebrow">Comparison lens — same frame / same protocol</p><h2>Frame {selectedFrame.frame_id}</h2></div><div className="segmented" role="tablist" aria-label="Model comparison view">{([['e1', 'E1 RGB'], ['e2', 'E2 + state'], ['split', 'Split']] as const).map(([id, label]) => <button key={id} role="tab" aria-selected={modelView === id} className={modelView === id ? 'selected' : ''} onClick={() => setModelView(id)}>{label}</button>)}</div></div>
        <div className="model-selectors" aria-label="Comparison models"><label>E1 model<select value={e1ModelId} onChange={(event) => void inspectModel('e1', event.target.value)}>{(e1Models.length ? e1Models : models).map((model) => <option key={model.model_id}>{model.model_id}</option>)}</select></label><label>E2 model<select value={e2ModelId} onChange={(event) => void inspectModel('e2', event.target.value)}>{(e2Models.length ? e2Models : models).map((model) => <option key={model.model_id}>{model.model_id}</option>)}</select></label>{modelDetailNote && <span className="model-note">{modelDetailNote}</span>}</div>
        <div className={`feed-layout view-${modelView}`}><div className="feeds">{modelView !== 'e2' && <Feed kind="e1" modelId={e1ModelId} frame={selectedFrame} offline={activeRunId === OFFLINE_RUN_ID} intervention={intervention} />}{modelView !== 'e1' && <Feed kind="e2" modelId={e2ModelId} frame={selectedFrame} offline={activeRunId === OFFLINE_RUN_ID} intervention={intervention} />}</div><DetectionDetail frame={selectedFrame} modelView={modelView} intervention={intervention} /></div>
      </section>
      {frameDetailWarning && <div className="micro-warning"><AlertTriangle size={13} />{frameDetailWarning}</div>}
      <Timeline frames={frames} selectedFrameId={selectedFrameId} onSelect={setSelectedFrameId} />
      <section className="history-panel"><div className="panel-heading"><div><p className="eyebrow">History capture</p><h2>Selected timeline frame</h2></div><Badge tone={selectedFrame.quality_flags.length ? 'amber' : 'green'}>{selectedFrame.quality_flags.length ? 'FLAGGED' : 'VALID'}</Badge></div><div className="history-grid"><FrameVisual frame={selectedFrame} detections={selectedFrame.detections} label="Recorded replay evidence" /><div className="history-detail"><SectionLabel icon={Clock3}>History frame detection detail</SectionLabel><h3>{selectedFrame.frame_id} · {formatTime(selectedFrame.source_time_ms)}</h3><dl><div><dt>Model</dt><dd>{selectedFrame.model_id}</dd></div><div><dt>Detections</dt><dd>{selectedFrame.detections.length}</dd></div><div><dt>Input mode</dt><dd>{selectedFrame.input_mode.replace(/_/g, ' ')}</dd></div><div><dt>Alignment</dt><dd>{selectedFrame.metadata_alignment.replace(/_/g, ' ')}</dd></div><div><dt>Latency</dt><dd>{selectedFrame.latency.end_to_end_ms == null ? 'Not recorded' : `${selectedFrame.latency.end_to_end_ms.toFixed(1)} ms`}</dd></div></dl></div></div></section>
      <InterventionPanel value={intervention} onChange={setIntervention} />
    </>}
  </div>
}

function Feed({ kind, modelId, frame, offline, intervention }: { kind: 'e1' | 'e2'; modelId: string; frame: InferenceRecord; offline: boolean; intervention: Intervention }) {
  const matches = modelKind(frame.model_id) === kind || modelKind(frame.model_id) === 'other'
  const available = offline || matches
  const detections = offline ? fixtureDetections(frame, kind) : available ? frame.detections : []
  return <article className="feed-card"><header><div><p>{kind === 'e1' ? 'E1 RGB stream' : 'E2 + state stream'}</p><h3>{kind === 'e1' ? 'E1 · FCOS — ResNet-50 FPN' : 'E2 · FCOS + gated FiLM'}</h3></div><Badge tone={available ? sourceTone(frame.prediction_source) : 'neutral'}>{available ? sourceLabel(frame.prediction_source) : 'NO OUTPUT'}</Badge></header><div className="feed-visual"><FrameVisual frame={frame} detections={detections} label={`${kind.toUpperCase()} frame output`} compact />{!available && <div className="unavailable-overlay"><Box size={20} /><strong>No replay output</strong><span>{modelId || `${kind.toUpperCase()} model`} has no record for this frame.</span></div>}</div><footer><span><b>{detections.length}</b> detections</span><span>{kind === 'e2' ? `${intervention === 'recorded' ? 'Recorded state' : `Intervention: ${intervention}`}` : 'RGB inference'}</span></footer></article>
}

function FrameVisual({ frame, detections, label, compact = false }: { frame: InferenceRecord; detections: Detection[]; label: string; compact?: boolean }) {
  const image = frame.image_data_url || frame.image_url
  const { width, height } = frame.original_size
  return <div className={`frame-visual ${compact ? 'compact' : ''}`} role="img" aria-label={`${label}: ${detections.length} detections from ${sourceLabel(frame.prediction_source).toLowerCase()} evidence`}>
    {image ? <img src={image} alt="Replay frame" /> : <div className="protocol-canvas" aria-hidden><span className="road road-a" /><span className="road road-b" /><span className="terrain terrain-a" /><span className="terrain terrain-b" /></div>}
    {!image && <span className="canvas-label">Protocol canvas · source image not provided by API</span>}
    {detections.map((detection, index) => { const [x1, y1, x2, y2] = detection.box_xyxy; const color = detection.class_name === 'Human' ? 'lime' : index % 2 ? 'cyan' : 'amber'; return <span key={`${detection.class_name}-${index}`} className={`detection-box ${color}`} style={{ left: `${x1 / width * 100}%`, top: `${y1 / height * 100}%`, width: `${(x2 - x1) / width * 100}%`, height: `${(y2 - y1) / height * 100}%` }}><i>{detection.class_name} {detection.raw_score.toFixed(2)}</i></span> })}
    <span className="frame-id-label">{frame.frame_id}</span>
  </div>
}

function DetectionDetail({ frame, modelView, intervention }: { frame: InferenceRecord; modelView: ModelView; intervention: Intervention }) {
  const effectiveMode = intervention === 'missing' ? 'state_masked' : frame.input_mode
  const effectiveAlignment = intervention === 'delayed' ? 'injected_delay' : intervention === 'invalid' ? 'unavailable' : frame.metadata_alignment
  return <aside className="detection-detail"><div className="detail-head"><div><p>Live frame detection detail</p><h3>{frame.frame_id}</h3></div><span>{formatTime(frame.source_time_ms)}</span></div><div className="detail-list">{frame.detections.length ? frame.detections.map((item, index) => <div key={`${item.class_name}-${index}`}><span>{item.class_name}</span><strong>{item.raw_score.toFixed(3)}</strong></div>) : <p className="empty-copy">No detections emitted.</p>}</div><div className="state-summary"><p>Input state</p><strong>{intervention === 'recorded' ? 'RECORDED' : `SIMULATED ${intervention.toUpperCase()}`}</strong><span>{effectiveMode.replace(/_/g, ' ')} · {effectiveAlignment.replace(/_/g, ' ')}</span></div><dl className="mini-metadata"><div><dt>View</dt><dd>{modelView === 'split' ? 'E1 / E2 split' : modelView.toUpperCase()}</dd></div><div><dt>Image space</dt><dd>{frame.original_size.width} × {frame.original_size.height}</dd></div><div><dt>Flags</dt><dd>{frame.quality_flags.length ? frame.quality_flags.join(', ') : 'none'}</dd></div></dl></aside>
}

function Timeline({ frames, selectedFrameId, onSelect }: { frames: InferenceRecord[]; selectedFrameId: string; onSelect: (id: string) => void }) {
  return <section className="timeline-panel"><div className="panel-heading"><SectionLabel icon={Clock3}>Frame timeline</SectionLabel><span>Arrow keys move through frames</span></div><div className="timeline-items">{frames.map((frame) => <button key={frame.frame_id} className={selectedFrameId === frame.frame_id ? 'selected' : ''} onClick={() => onSelect(frame.frame_id)} aria-pressed={selectedFrameId === frame.frame_id}><span>{frame.frame_id.replace(/^.*?([0-9]+)$/, '$1')}</span><i className={frame.quality_flags.length ? 'flagged' : ''} /><small>{formatTime(frame.source_time_ms)}</small></button>)}</div></section>
}

function InterventionPanel({ value, onChange }: { value: Intervention; onChange: (value: Intervention) => void }) {
  const items: Array<[Intervention, string, string]> = [['recorded', 'Recorded state', 'Use the replay contract unchanged'], ['missing', 'Missing state', 'Mask the displayed state vector'], ['invalid', 'Invalid state', 'Show a sanitized invalid-state path'], ['delayed', 'Injected delay', 'Show delayed metadata alignment']]
  return <section className="intervention-panel"><div><SectionLabel icon={SlidersHorizontal}>Input intervention</SectionLabel><h2>Stress the pairing, keep the claim honest.</h2><p>These controls change the review lens only. They do not run inference or create a new metric.</p></div><div className="intervention-options">{items.map(([id, title, detail]) => <button key={id} className={value === id ? 'selected' : ''} onClick={() => onChange(id)} aria-pressed={value === id}><i /><span><strong>{title}</strong><small>{detail}</small></span></button>)}</div></section>
}

function ResultsEvidence({ mode, reports, reportsFallback, models, reviews, error, selectedReport, reportDetailState, reportDetailError, onInspectReport, onCloseReport, onRetry }: { mode: ServiceMode; reports: EvaluationReport[]; reportsFallback: boolean; models: ModelSummary[]; reviews: Review[]; error: string; selectedReport: EvaluationReport | null; reportDetailState: 'idle' | 'loading' | 'error'; reportDetailError: string; onInspectReport: (report: EvaluationReport) => void; onCloseReport: () => void; onRetry: () => void }) {
  const sorted = [...reports].sort((a, b) => { const rank = (report: EvaluationReport) => /full-pass/.test(report.model_id) ? 0 : /masked|shuffled/.test(report.model_id) ? 1 : 2; return rank(a) - rank(b) || a.model_id.localeCompare(b.model_id) })
  const e1 = sorted.find((report) => report.model_id === 'imagenet-full-pass-e1-rgb-masked')
  const e2 = sorted.find((report) => report.model_id === 'imagenet-full-pass-e2-paired-film')
  return <div className="page results-page"><header className="page-header"><div><p className="eyebrow">Results / Evidence register</p><h1>Measured evidence, with its limits attached.</h1><p>Every result is labelled by what exists in the workspace today.</p></div><Badge tone="amber">NO FABRICATED METRICS</Badge></header>
    {(error || reportsFallback) && <div className="inline-notice amber"><AlertTriangle size={18} /><div><strong>{mode === 'offline' ? 'Live reports unavailable' : 'Using documented report fallback'}</strong><span>{error || 'The reports endpoint returned no saved reports. Values below are the documented Review 2 snapshot, not a live API response.'}</span></div><button onClick={onRetry}><RefreshCw size={14} /> Retry</button></div>}
    <section className="split-evidence panel"><div className="panel-heading"><div><SectionLabel icon={Server}>Dataset evidence</SectionLabel><h2>AU-AIR split register</h2></div><Badge tone="green">32,823 TOTAL FRAMES</Badge></div><div className="split-bar" aria-label="Training 56.43 percent, development 17.47 percent, sealed final test 26.10 percent"><span className="train" style={{ width: '56.43%' }} /><span className="dev" style={{ width: '17.47%' }} /><span className="test" style={{ width: '26.10%' }} /></div><div className="split-stats"><div><i className="train" /><span>Training<strong>18,523</strong><small>56.43%</small></span></div><div><i className="dev" /><span>Development<strong>5,734</strong><small>17.47%</small></span></div><div><i className="test" /><span>Final test<strong>8,566</strong><small>26.10% · sealed</small></span></div></div></section>
    {e1 && e2 && <section className="headline-comparison panel"><div className="panel-heading"><div><SectionLabel icon={GitCompare}>Review 2 comparison</SectionLabel><h2>Development operating evidence</h2></div><Badge tone="amber">FINAL TEST SEALED</Badge></div><div className="comparison-table"><div className="table-head"><span>Model</span><span>AP50</span><span>Recall</span><span>Detection accuracy</span></div>{[e1, e2].map((report) => { const metrics = reportMetrics(report); return <button key={report.model_id} onClick={() => onInspectReport(report)}><span><strong>{modelKind(report.model_id).toUpperCase()}</strong><small>{report.model_id}</small></span><b>{formatMetric(metrics.ap50)}</b><b>{formatMetric(metrics.recall)}</b><b>{formatMetric(metrics.detectionAccuracy)}</b></button> })}</div><p className="definition-note"><Info size={13} /> Detection accuracy is TP/(TP+FP+FN). It is not image-classification accuracy. AP50 is pooled development evidence; recall and detection accuracy are shown at the development-selected best-F1 operating point.</p></section>}
    <div className="register-heading"><div><p className="eyebrow">Saved model reports</p><h2>Repository-discovered register</h2></div><span>{sorted.length} report{sorted.length === 1 ? '' : 's'} · future adapters appear automatically</span></div>
    {sorted.length ? <div className="report-grid">{sorted.map((report) => <ReportCard key={report.report_id} report={report} onInspect={() => onInspectReport(report)} />)}</div> : <div className="empty-inline"><Archive size={22} /><strong>No evaluation reports discovered</strong><span>The register will populate when the backend discovers schema-valid reports.</span></div>}
    <section className="catalog-panel panel"><div className="panel-heading"><div><SectionLabel icon={Server}>Model catalogue</SectionLabel><h2>API model availability</h2></div><span>{models.length} models</span></div><div className="catalog-list">{models.length ? models.map((model) => <div key={model.model_id}><span><strong>{model.model_id}</strong><small>{model.partition ?? model.prediction_source ?? 'catalog entry'}</small></span><Badge tone={model.available_for_inference ? 'green' : 'neutral'}>{model.available_for_inference ? 'INFERENCE READY' : 'REPORT / REPLAY ONLY'}</Badge></div>) : <p>No model catalogue entries returned.</p>}</div></section>
    <section className="review-register panel"><div className="panel-heading"><div><SectionLabel icon={ShieldCheck}>Human decisions</SectionLabel><h2>Saved operator reviews</h2></div><Badge>{reviews.length}</Badge></div><p>{reviews.length ? 'Reviews are persisted by the backend and linked to run and frame evidence.' : 'No operator reviews have been saved yet.'}</p></section>
    {selectedReport && <ReportDetail report={selectedReport} state={reportDetailState} error={reportDetailError} onClose={onCloseReport} />}
  </div>
}

function ReportCard({ report, onInspect }: { report: EvaluationReport; onInspect: () => void }) {
  const metrics = reportMetrics(report)
  return <article className="report-card"><div className="report-card-top"><span><Activity size={14} /> {report.partition}</span><Badge tone={report.documentedFallback ? 'amber' : 'green'}>{report.documentedFallback ? 'DOCUMENTED SNAPSHOT' : 'MEASURED'}</Badge></div><h3>{report.model_id}</h3><p>{report.artifact_kind.replace(/_/g, ' ')} · checkpoint {shortHash(report.checkpoint_id)}</p><dl><div><dt>AP50</dt><dd>{formatMetric(metrics.ap50)}</dd></div><div><dt>Recall</dt><dd>{formatMetric(metrics.recall)}</dd></div><div><dt>Detection accuracy</dt><dd>{formatMetric(metrics.detectionAccuracy)}</dd></div><div><dt>Evaluated frames</dt><dd>{metrics.frames?.toLocaleString() ?? '—'}</dd></div></dl><button onClick={onInspect}>Inspect report <ArrowRight size={14} /></button></article>
}

function ReportDetail({ report, state, error, onClose }: { report: EvaluationReport; state: 'idle' | 'loading' | 'error'; error: string; onClose: () => void }) {
  const metrics = reportMetrics(report)
  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}><section className="report-dialog" role="dialog" aria-modal="true" aria-labelledby="report-title"><header><div><p className="eyebrow">Evaluation report detail</p><h2 id="report-title">{report.model_id}</h2></div><button className="icon-button" onClick={onClose} aria-label="Close report detail"><X size={18} /></button></header>{state === 'loading' ? <LoadingState label="Verifying report detail…" /> : <div className="report-dialog-body">{state === 'error' && <div className="micro-warning"><AlertTriangle size={13} />{error}</div>}<div className="dialog-metrics"><div><span>AP50</span><strong>{formatMetric(metrics.ap50)}</strong></div><div><span>Recall</span><strong>{formatMetric(metrics.recall)}</strong></div><div><span>Detection accuracy</span><strong>{formatMetric(metrics.detectionAccuracy)}</strong></div><div><span>Latency p95</span><strong>{metrics.latency == null ? '—' : `${metrics.latency.toFixed(2)} ms`}</strong></div></div><dl className="report-metadata"><div><dt>Partition</dt><dd>{report.partition}</dd></div><div><dt>Checkpoint</dt><dd>{report.checkpoint_id}</dd></div><div><dt>Protocol</dt><dd>{shortHash(report.protocol_sha256)}</dd></div><div><dt>Final test</dt><dd>{report.final_test_unsealed ? 'Unsealed by report' : 'Sealed'}</dd></div></dl><h3>Limitations</h3><ul>{report.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul></div>}</section></div>
}

function Architecture({ capabilities, readiness, mode }: { capabilities: CapabilityResponse | null; readiness: ReadinessResponse | null; mode: ServiceMode }) {
  const pipeline = [['01', 'Source recordings', 'AU-AIR / VisDrone', 'current'], ['02', 'Validation + pairing', 'schema · timestamps · splits', 'current'], ['03', 'Model gate + FiLM', 'FCOS conditioned features', 'current'], ['04', 'Inference API', 'FastAPI · replay · reviews', 'current'], ['05', 'Review console', 'this frontend · evidence', 'current'], ['06', 'Storage + monitoring', 'object store · Postgres · drift', 'planned']] as const
  const details = [
    ['01 — Source recordings', 'AU-AIR: 32,823 matched RGB frames at 1920×1080 with 5 Hz telemetry. Training uses 18,523 frames; development uses 5,734; 8,566 final-test frames remain sealed.', ['AU-AIR dataset', 'VisDrone RGB', '1920×1080', '5 Hz source']],
    ['02 — Validation + pairing', 'Schema validation, timestamp alignment, and declared split construction preserve frame/state provenance before any model call.', ['Schema validation', 'Timestamp alignment', 'Split construction', 'Sealed test roots']],
    ['03 — Model gate + FiLM', 'E1 is an image-only FCOS baseline. E2 adds gated FiLM conditioning from flight state; invalid state is sanitized before encoding.', ['FCOS · ResNet-50 FPN', 'Gated FiLM', '8-feature state vector', 'Invalid-state sanitization']],
    ['04 — Inference API', `FastAPI exposes health, readiness, replay, model reports, inference, and idempotent human review. Computed inference is ${readiness?.computed_inference_ready ? 'ready' : 'currently unavailable'}; replay remains ${readiness?.replay_ready === false ? 'unavailable' : 'ready'}.`, ['Typed JSON contracts', 'Bounded inference', 'Replay fallback', 'Review persistence']],
    ['05 — Review console', 'The website loads API runs, frames, models, reports, and reviews. Offline fixture replay is kept separate from measured development reports.', ['React + Vite', 'Typed API client', 'Fixture labelling', 'JSON export']],
    ['06 — Storage + monitoring', 'Planned production path: immutable object storage, PostgreSQL audit records, queued GPU workers, and drift monitoring.', ['S3-compatible store', 'PostgreSQL', 'Queued workers', 'Drift monitoring']],
  ]
  return <div className="page architecture-page"><header className="page-header"><div><p className="eyebrow">System / Architecture</p><h1>Evidence moves in one direction.</h1><p>Current components and planned scale paths are intentionally separated.</p></div><div className="architecture-legend"><Badge tone="green"><i className="green-dot" /> CURRENT / EVIDENCED</Badge><Badge tone="amber"><i className="amber-dot" /> PLANNED / NOT IMPLEMENTED</Badge></div></header>
    <section className="pipeline panel"><div className="panel-heading"><SectionLabel icon={Layers3}>Primary pipeline</SectionLabel><Badge tone={mode === 'offline' ? 'amber' : 'green'}>{mode === 'offline' ? 'OFFLINE VIEW' : `API ${capabilities?.api_version ?? 'CONNECTED'}`}</Badge></div><div className="pipeline-row">{pipeline.map(([number, title, subtitle, status], index) => <div className="pipeline-step" key={number}><article className={status}><span>{number}</span><strong>{title}</strong><small>{subtitle}</small><Badge tone={status === 'current' ? 'green' : 'amber'}>{status.toUpperCase()}</Badge></article>{index < pipeline.length - 1 && <ArrowRight aria-hidden size={15} />}</div>)}</div></section>
    <div className="architecture-details">{details.map(([title, copy, tags], index) => <article className={`architecture-card ${index === 5 ? 'planned' : ''}`} key={title as string}><div><span>{title}</span><Badge tone={index === 5 ? 'amber' : 'green'}>{index === 5 ? 'PLANNED' : 'CURRENT'}</Badge></div><p>{copy as string}</p><footer>{(tags as string[]).map((tag) => <span key={tag}>{tag}</span>)}</footer></article>)}</div>
    <section className="scale-path panel"><div><SectionLabel icon={HardDrive}>Scale path</SectionLabel><h2>Planned production architecture</h2><p>Object storage, PostgreSQL, queued GPU workers, and drift monitoring are design specifications. They are not represented as operational services.</p></div><Badge tone="amber">NOT IMPLEMENTED</Badge></section>
    <section className="model-architecture"><article><div className="model-letter">E1</div><div><p className="eyebrow">RGB baseline</p><h2>FCOS · ResNet-50 FPN</h2><p>Image-only object detection. Flight state is not ingested. Output is bounding boxes, class labels, and confidence scores.</p><dl><div><dt>Backbone</dt><dd>ResNet-50 FPN</dd></div><div><dt>Head</dt><dd>FCOS · anchor-free</dd></div><div><dt>State</dt><dd>Not used</dd></div></dl></div></article><article><div className="model-letter cyan">E2</div><div><p className="eyebrow">Flight-aware</p><h2>FCOS + gated FiLM</h2><p>Flight state conditions feature channels. Missing or invalid state is sanitized and the gate can bypass adjustment.</p><dl><div><dt>Backbone</dt><dd>ResNet-50 FPN</dd></div><div><dt>Head</dt><dd>FCOS + gated FiLM</dd></div><div><dt>State</dt><dd>alt, vXYZ, roll, pitch, sin/cos(yaw)</dd></div></dl></div></article></section>
  </div>
}

function EvidenceDrawer({ run, frame, intervention, reviews, decision, setDecision, comment, setComment, reviewState, reviewMessage, selectedReview, onInspectReview, onSave, onExport, onClose }: { run: RunSummary | null; frame: InferenceRecord | null; intervention: Intervention; reviews: Review[]; decision: ReviewDecision | ''; setDecision: (decision: ReviewDecision | '') => void; comment: string; setComment: (comment: string) => void; reviewState: 'idle' | 'saving' | 'saved' | 'error'; reviewMessage: string; selectedReview: Review | null; onInspectReview: (review: Review) => void; onSave: () => void; onExport: () => void; onClose: () => void }) {
  return <aside className="evidence-drawer" role="dialog" aria-modal="true" aria-labelledby="evidence-title"><header><div><p className="eyebrow">Evidence Drawer</p><h2 id="evidence-title">Frame evidence</h2></div><button className="icon-button" onClick={onClose} aria-label="Close evidence drawer"><X size={18} /></button></header><div className="drawer-body">{frame && run ? <>
    <section className="provenance-card"><div><SectionLabel>Provenance</SectionLabel><Badge tone={sourceTone(frame.prediction_source)}>{sourceLabel(frame.prediction_source)}</Badge></div><p>{frame.prediction_source === 'fixture' ? 'Fixture evidence validates the interface and review workflow only. It is not a measured model result.' : frame.prediction_source === 'cached' ? 'Recorded output returned by the replay API.' : 'Computed output returned by the inference runtime.'}</p></section>
    <SectionLabel>Frame metadata</SectionLabel><dl className="evidence-metadata"><div><dt>Frame ID</dt><dd>{frame.frame_id}</dd></div><div><dt>Source time</dt><dd>{formatTime(frame.source_time_ms)}</dd></div><div><dt>Run</dt><dd>{run.run_id}</dd></div><div><dt>Model / protocol</dt><dd>{frame.model_id}<small>{shortHash(frame.protocol_sha256)}</small></dd></div><div><dt>Image space</dt><dd>{frame.original_size.width} × {frame.original_size.height}</dd></div><div><dt>State source</dt><dd>{frame.metadata_alignment.replace(/_/g, ' ')}</dd></div><div><dt>Review lens</dt><dd>{intervention}</dd></div><div><dt>Quality flags</dt><dd>{frame.quality_flags.length ? frame.quality_flags.join(', ') : 'none'}</dd></div></dl>
    {reviews.length > 0 && <section className="prior-reviews"><SectionLabel icon={ShieldCheck}>Prior reviews</SectionLabel>{reviews.map((review) => <button key={review.review_id} onClick={() => onInspectReview(review)}><span>{review.decision.replace('_', ' ')}</span><small>{new Date(review.created_at).toLocaleString()}</small></button>)}{selectedReview && <p>Verified review: {selectedReview.review_id.slice(0, 16)}…</p>}</section>}
    <div className="drawer-divider" /><SectionLabel icon={ShieldCheck}>Operator review</SectionLabel><label className="field-label" htmlFor="review-decision">Decision</label><select id="review-decision" value={decision} onChange={(event) => setDecision(event.target.value as ReviewDecision | '')}><option value="">— Select decision —</option><option value="accept">Accept evidence</option><option value="needs_review">Needs further review</option><option value="reject">Reject evidence</option></select><label className="field-label" htmlFor="review-comment">Comment <span>optional</span></label><textarea id="review-comment" rows={4} maxLength={4000} value={comment} onChange={(event) => setComment(event.target.value)} placeholder="Add observation or context…" />
    {reviewMessage && <div className={`save-message ${reviewState}`} role="status">{reviewState === 'saved' ? <Check size={14} /> : reviewState === 'error' ? <AlertTriangle size={14} /> : <LoaderCircle className="spin" size={14} />}{reviewMessage}</div>}
    <button className="button primary full" disabled={reviewState === 'saving'} onClick={onSave}><Save size={15} />{reviewState === 'saving' ? 'Saving…' : 'Save decision to API'}</button><button className="button secondary full" onClick={onExport}><Download size={15} />Export local JSON copy</button><p className="drawer-note"><Info size={12} />Saving writes the declared human decision only. No risk label is inferred.</p>
  </> : <div className="drawer-empty"><Archive size={22} /><p>Select a frame to inspect its evidence.</p></div>}</div></aside>
}

function RecordingChooser({ runs, selected, closeRef, onChoose, onClose }: { runs: RunSummary[]; selected: string; closeRef: React.RefObject<HTMLButtonElement | null>; onChoose: (id: string) => void; onClose: () => void }) {
  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}><section className="chooser-dialog" role="dialog" aria-modal="true" aria-labelledby="chooser-title"><header><div><p className="eyebrow">Replay source</p><h2 id="chooser-title">Choose a recording</h2></div><button ref={closeRef} className="icon-button" onClick={onClose} aria-label="Close recording chooser"><X size={18} /></button></header><div className="chooser-list">{runs.map((run) => <button key={run.run_id} className={run.run_id === selected ? 'selected' : ''} onClick={() => onChoose(run.run_id)}><span className="option-icon">{run.run_id === selected ? <Check size={15} /> : <Archive size={15} />}</span><span><strong>{run.run_id}</strong><small>{run.offline ? 'Built-in offline workflow fixture · never measured evidence' : `${run.frame_count} API frame${run.frame_count === 1 ? '' : 's'}`}</small></span><Badge tone={sourceTone(run.prediction_source)}>{run.offline ? 'OFFLINE FIXTURE' : sourceLabel(run.prediction_source)}</Badge></button>)}</div></section></div>
}

function LoadingState({ label }: { label: string }) { return <div className="loading-state" role="status"><LoaderCircle className="spin" size={22} /><span>{label}</span></div> }
function ErrorState({ title, message, onRetry, secondary }: { title: string; message: string; onRetry: () => void; secondary?: { label: string; action: () => void } }) { return <div className="state-card error-state"><AlertTriangle size={24} /><h2>{title}</h2><p>{message}</p><div><button className="button primary" onClick={onRetry}><RefreshCw size={14} />Retry</button>{secondary && <button className="button secondary" onClick={secondary.action}>{secondary.label}</button>}</div></div> }
function EmptyState({ onUseOffline }: { onUseOffline: () => void }) { return <div className="state-card"><Archive size={24} /><h2>No frames in this recording</h2><p>The API returned an empty replay. No evidence has been synthesized.</p><button className="button secondary" onClick={onUseOffline}>Open labelled offline demo</button></div> }
