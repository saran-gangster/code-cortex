import {
  Activity, AlertTriangle, Archive, ArrowRight, Box, Check, ChevronDown, CircleHelp, Clock3, CloudOff,
  Download, Eye, FileCheck2, Gauge, GitCompare, HardDrive, Info, Layers3, LoaderCircle, Menu, Radio,
  Pause, Play, RefreshCw, Save, Server, ShieldCheck, SlidersHorizontal, X,
} from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, API_BASE, errorMessage } from './api'
import { containImage, replayImageUrl } from './replay-image'
import MovingDemo from './MovingDemo'
import { documentedReports, mergeDocumentedReports, fixtureDetections, OFFLINE_RUN_ID, offlineFrames, offlineModels, offlineRun } from './fixtures'
import type {
  CapabilityResponse, Detection, EvaluationReport, HealthResponse, InferenceRecord, Intervention, ModelSummary,
  ModelView, Page, ReadinessResponse, Review, ReviewDecision, RunSummary, ServiceMode,
} from './types'

const pageFromHash = (): Page => {
  const value = window.location.hash.replace(/^#\/?/, '')
  return value === 'results' || value === 'architecture' ? value : 'live'
}

const modelKind = (id: string): 'e2' | 'e1' | 'other' => {
  const value = id.toLowerCase()
  if (/(^|[-_])e1([-_]|$)|rgb[-_]?masked|vision/.test(value)) return 'e1'
  if (/(^|[-_])e2([-_]|$)|film|fusion/.test(value)) return 'e2'
  return 'other'
}

const sourceLabel = (source: InferenceRecord['prediction_source']) => source === 'annotation' ? 'ANNOTATIONS' : source === 'computed' ? 'COMPUTED' : source === 'cached' ? 'CACHED' : 'FIXTURE'
const sourceTone = (source: InferenceRecord['prediction_source']) => source === 'computed' ? 'green' : source === 'cached' || source === 'annotation' ? 'cyan' : 'amber'
const scoreLabel = (score: number | null) => score == null ? 'annotation' : score.toFixed(2)
const detectionLabel = (detection: Detection) => `${detection.class_name} ${scoreLabel(detection.raw_score)}`
const shortHash = (value: string) => value === '0'.repeat(64) ? 'not applicable' : `${value.slice(0, 12)}…`
const formatTime = (milliseconds: number | null) => milliseconds == null ? 'Not supplied' : `${Math.floor(milliseconds / 60_000).toString().padStart(2, '0')}:${Math.floor((milliseconds % 60_000) / 1000).toString().padStart(2, '0')}.${Math.floor(milliseconds % 1000 / 100)}`
const formatMetric = (value: number | null, digits = 4) => value == null ? '—' : value.toFixed(digits)
const frameNumber = (frame: InferenceRecord) => frame.frame_id.replace(/^.*?([0-9]+)$/, '$1').replace(/^0+(?=\d{3})/, '')
const stateModelLabel = (frame: InferenceRecord | null) => frame?.state_model_id?.match(/(?:^|-)e(\d+)(?:-|$)/)?.[1] ? `E${frame.state_model_id.match(/(?:^|-)e(\d+)(?:-|$)/)![1]}` : 'E2'
const frameStatus = (frame: InferenceRecord) => frame.quality_flags.some((flag) => /invalid/i.test(flag)) ? 'invalid' : frame.quality_flags.length ? 'delayed' : 'valid'

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
    f1: pathNumber(report.metrics, 'operating_point_sweep.best_f1_point.f1'),
    threshold: pathNumber(report.metrics, 'operating_point_sweep.best_f1_point.score_threshold'),
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

function BrandMark() {
  return <span className="brand-mark" aria-hidden="true"><img src="/assets/aeroguard-logo.png" alt="" width="74" height="73" /></span>
}

function useModalFocus(onClose: () => void) {
  const ref = useRef<HTMLElement>(null)
  const close = useRef(onClose)
  close.current = onClose
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    const dialog = ref.current
    dialog?.querySelector<HTMLButtonElement>('button')?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); close.current(); return }
      if (event.key !== 'Tab' || !dialog) return
      const items = [...dialog.querySelectorAll<HTMLElement>('button:not(:disabled), select, textarea, [tabindex="0"]')]
      const first = items[0], last = items[items.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
    }
    dialog?.addEventListener('keydown', onKey)
    return () => {
      dialog?.removeEventListener('keydown', onKey)
      window.requestAnimationFrame(() => {
        if (previous?.isConnected && previous !== document.body) previous.focus()
        else document.querySelector<HTMLButtonElement>('.run-context')?.focus()
      })
    }
  }, [])
  return ref
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
  const [selectedFrameDetail, setSelectedFrame] = useState<InferenceRecord | null>(null)
  const [playing, setPlaying] = useState(true)
  const [frameDetailWarning, setFrameDetailWarning] = useState('')
  const [modelView, setModelView] = useState<ModelView>('split')
  const [e2ModelId, setE2ModelId] = useState('')
  const [e1ModelId, setE1ModelId] = useState('')
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
  const reportRequestRef = useRef(0)
  const chooserCloseRef = useRef<HTMLButtonElement>(null)
  const selectedFrame = selectedFrameDetail?.frame_id === selectedFrameId ? selectedFrameDetail : frames.find((frame) => frame.frame_id === selectedFrameId) ?? null
  const seekFrame = useCallback((id: string) => { setPlaying(false); setSelectedFrameId(id) }, [])

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
      runs: apiRuns, models: apiModels, reports: mergeDocumentedReports(apiReports),
      reviews: reviews.status === 'fulfilled' ? reviews.value : [], reportsFallback: mergeDocumentedReports(apiReports).some((report) => report.documentedFallback),
    })
    setWorkspaceError(failedResources ? `${failedResources} workspace resource${failedResources === 1 ? '' : 's'} could not be loaded. Available API data is shown.` : '')
    setActiveRunId((current) => {
      if (current === OFFLINE_RUN_ID || apiRuns.some((run) => run.run_id === current)) return current
      return apiRuns.find((run) => (run.prediction_source === 'computed' || run.prediction_source === 'cached') && run.frame_count > 1)?.run_id ?? OFFLINE_RUN_ID
    })
  }, [])

  useEffect(() => { void loadWorkspace() }, [loadWorkspace])

  useEffect(() => {
    if (!snapshot.models.length) return
    setE2ModelId((current) => current || snapshot.models.find((model) => modelKind(model.model_id) === 'e2')?.model_id || snapshot.models[0].model_id)
    setE1ModelId((current) => current || snapshot.models.find((model) => modelKind(model.model_id) === 'e1')?.model_id || snapshot.models[1]?.model_id || snapshot.models[0].model_id)
  }, [snapshot.models])

  useEffect(() => {
    let cancelled = false
    setSelectedFrame(null)
    setSelectedFrameId('')
    setFrameDetailWarning('')
    setReviewState('idle')
    setReviewMessage('')
    setPlaying(true)
    if (!activeRunId) { setFrames([]); setFramesState('empty'); return }
    if (activeRunId === OFFLINE_RUN_ID) {
      setFrames(offlineFrames); setFramesState('ready'); setSelectedFrameId(offlineFrames[0]?.frame_id ?? ''); return
    }
    setFramesState('loading')
    setFramesError('')
    api.frames(activeRunId).then((items) => {
      if (cancelled) return
      setFrames(items); setFramesState(items.length ? 'ready' : 'empty'); setSelectedFrameId(items[0]?.frame_id ?? '')
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

  // Advance only through the manifest: never manufacture frames or interpolate predictions.
  useEffect(() => {
    if (!playing || page !== 'live' || drawerOpen || chooserOpen || framesState !== 'ready' || frames.length < 2) return
    const timer = window.setTimeout(() => {
      setSelectedFrameId((current) => frames[(frames.findIndex((frame) => frame.frame_id === current) + 1) % frames.length].frame_id)
    }, 1200)
    return () => window.clearTimeout(timer)
  }, [playing, page, drawerOpen, chooserOpen, frames, framesState, selectedFrameId])

  // Pick up newly available API frames without resetting the operator's position.
  useEffect(() => {
    if (!activeRunId || activeRunId === OFFLINE_RUN_ID || page !== 'live') return
    let cancelled = false
    let timer: number
    const refresh = async () => {
      try {
        const items = await api.frames(activeRunId)
        if (cancelled) return
        setFrames((current) => JSON.stringify(current) === JSON.stringify(items) ? current : items)
        setFramesState(items.length ? 'ready' : 'empty')
        setSelectedFrameId((current) => items.some((frame) => frame.frame_id === current) ? current : items[0]?.frame_id ?? '')
        setFramesError('')
      } catch (error) { if (!cancelled) setFramesError(`Replay refresh unavailable: ${errorMessage(error)}`) }
      if (!cancelled) timer = window.setTimeout(refresh, 5000)
    }
    timer = window.setTimeout(refresh, 5000)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [activeRunId, page])

  useEffect(() => {
    setDecision(''); setComment(''); setReviewState('idle'); setReviewMessage(''); setSelectedReview(null)
  }, [activeRunId, selectedFrameId])

  useEffect(() => { if (drawerOpen || chooserOpen) setPlaying(false) }, [drawerOpen, chooserOpen])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const editing = target?.matches('input, textarea, select, button')
      if (event.key === 'Escape') { setDrawerOpen(false); setChooserOpen(false); return }
      if (page !== 'live' || editing || chooserOpen) return
      if (event.key.toLowerCase() === 'e') { setDrawerOpen((open) => !open); return }
      if (drawerOpen) return
      if (event.key === ' ') { event.preventDefault(); setPlaying((current) => !current); return }
      const index = frames.findIndex((frame) => frame.frame_id === selectedFrameId)
      if (event.key === 'ArrowLeft' && index > 0) { event.preventDefault(); seekFrame(frames[index - 1].frame_id) }
      if (event.key === 'ArrowRight' && index >= 0 && index < frames.length - 1) { event.preventDefault(); seekFrame(frames[index + 1].frame_id) }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [frames, page, selectedFrameId, drawerOpen, chooserOpen, seekFrame])

  useEffect(() => { if (chooserOpen) window.setTimeout(() => chooserCloseRef.current?.focus(), 0) }, [chooserOpen])

  const activeRun = useMemo(() => snapshot.runs.find((run) => run.run_id === activeRunId) ?? (activeRunId === OFFLINE_RUN_ID ? offlineRun : null), [activeRunId, snapshot.runs])
  const frameReviews = useMemo(() => snapshot.reviews.filter((review) => review.run_id === activeRunId && review.frame_id === selectedFrameId), [activeRunId, selectedFrameId, snapshot.reviews])
  const availableRuns = useMemo(() => snapshot.runs.some((run) => run.run_id === OFFLINE_RUN_ID) ? snapshot.runs : [...snapshot.runs, offlineRun], [snapshot.runs])

  const inspectModel = async (kind: 'e2' | 'e1', modelId: string) => {
    if (kind === 'e2') setE2ModelId(modelId); else setE1ModelId(modelId)
    setModelDetailNote('')
    if (serviceMode === 'offline') return
    try { const detail = await api.model(modelId); setModelDetailNote(`${detail.model_id} verified against the model detail endpoint.`) }
    catch (error) { setModelDetailNote(`Model detail unavailable: ${errorMessage(error)}`) }
  }

  const inspectReport = async (report: EvaluationReport) => {
    const requestId = ++reportRequestRef.current
    setSelectedReport(report); setReportDetailError('')
    if (report.documentedFallback || serviceMode === 'offline') { setReportDetailState('idle'); return }
    setReportDetailState('loading')
    try {
      const detail = await api.report(report.report_id)
      if (requestId !== reportRequestRef.current) return
      setSelectedReport(detail); setReportDetailState('idle')
    } catch (error) {
      if (requestId !== reportRequestRef.current) return
      setReportDetailState('error'); setReportDetailError(errorMessage(error))
    }
  }

  const inspectReview = async (review: Review) => {
    if (serviceMode === 'offline') { setSelectedReview(review); return }
    try { setSelectedReview(await api.review(review.review_id)) }
    catch (error) { setReviewMessage(`Review detail unavailable: ${errorMessage(error)}`) }
  }

  const saveReview = async () => {
    if (!selectedFrame || !activeRun || !decision) { setReviewState('error'); setReviewMessage('Choose a decision before saving.'); return }
    if (activeRunId === OFFLINE_RUN_ID) { setReviewState('error'); setReviewMessage('This bundled replay is not registered in the API review store. Export a local JSON copy or select an API recording to save a review.'); return }
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
      note: activeRunId === OFFLINE_RUN_ID ? 'AU-AIR development replay; cached GPU-computed E2 and E4 detections. Not a benchmark claim.' : 'Local copy of an operator review.',
    }
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `aeroguard-review-${selectedFrame.frame_id}.json`; anchor.click(); URL.revokeObjectURL(url)
  }

  return <div className={`app-shell ${drawerOpen ? 'drawer-is-open' : ''}`}>
    <header className="topbar" inert={drawerOpen || chooserOpen}>
      <div className="header-context"><button className="brand" onClick={() => navigate('live')} aria-label="AeroGuard live review"><BrandMark /><span><strong>AeroGuard</strong><small>Flight-Aware Perception</small></span></button><button className="run-context" onClick={() => setChooserOpen(true)} aria-label="Choose recording"><span>Flight Review / 01</span><small>{activeRunId === OFFLINE_RUN_ID ? 'AU-AIR development replay' : activeRun?.run_id ?? 'Choose a recording'} · {activeRun?.prediction_source ?? 'connecting'}</small></button></div>
      <div className="top-actions"><button className={`header-evidence ${drawerOpen ? 'active' : ''}`} aria-expanded={drawerOpen} aria-controls="evidence-drawer" onClick={() => setDrawerOpen((open) => !open)}>⌁ Evidence [E]</button><span className="avatar" aria-label="Workspace owner S">S</span></div>
    </header>
    <div className="app-layout" inert={drawerOpen || chooserOpen}>
      <main className="main-content" id="main-content">
        {page === 'live' && <LiveReview playing={playing && frames.length > 1} onTogglePlayback={() => setPlaying((current) => !current)} serviceMode={serviceMode} workspaceError={workspaceError} activeRun={activeRun} activeRunId={activeRunId} frames={frames} framesState={framesState} framesError={framesError} selectedFrame={selectedFrame} selectedFrameId={selectedFrameId} setSelectedFrameId={seekFrame} frameDetailWarning={frameDetailWarning} modelView={modelView} setModelView={setModelView} models={snapshot.models} e2ModelId={e2ModelId} e1ModelId={e1ModelId} inspectModel={inspectModel} modelDetailNote={modelDetailNote} intervention={intervention} setIntervention={setIntervention} onOpenChooser={() => setChooserOpen(true)} onOpenEvidence={() => setDrawerOpen(true)} onUseOffline={() => setActiveRunId(OFFLINE_RUN_ID)} onRetry={() => void loadWorkspace()} />}
        {page === 'results' && <ResultsEvidence mode={serviceMode} reports={snapshot.reports} reportsFallback={snapshot.reportsFallback} models={snapshot.models} reviews={snapshot.reviews} error={workspaceError} selectedReport={selectedReport} reportDetailState={reportDetailState} reportDetailError={reportDetailError} onInspectReport={(report) => void inspectReport(report)} onCloseReport={() => { reportRequestRef.current += 1; setSelectedReport(null) }} onRetry={() => void loadWorkspace()} />}
        {page === 'architecture' && <Architecture capabilities={snapshot.capabilities} readiness={snapshot.readiness} mode={serviceMode} />}
      </main>
    </div>
    <nav className="bottom-nav" inert={drawerOpen || chooserOpen} aria-label="Primary navigation"><NavButton active={page === 'live'} icon={Eye} onClick={() => navigate('live')}>Live review</NavButton><NavButton active={page === 'results'} icon={Gauge} onClick={() => navigate('results')}>Results & evidence</NavButton><NavButton active={page === 'architecture'} icon={Layers3} onClick={() => navigate('architecture')}>Architecture</NavButton></nav>
    {drawerOpen && <EvidenceDrawer run={activeRun} frame={selectedFrame} intervention={intervention} reviews={frameReviews} decision={decision} setDecision={setDecision} comment={comment} setComment={setComment} reviewState={reviewState} reviewMessage={reviewMessage} selectedReview={selectedReview} onInspectReview={(review) => void inspectReview(review)} onSave={() => void saveReview()} onExport={exportReview} onClose={() => setDrawerOpen(false)} />}
    {chooserOpen && <RecordingChooser runs={availableRuns} selected={activeRunId} closeRef={chooserCloseRef} onChoose={(runId) => { setActiveRunId(runId); setChooserOpen(false) }} onClose={() => setChooserOpen(false)} />}
  </div>
}

function NavButton({ active, icon: Icon, onClick, children }: { active: boolean; icon: React.ElementType; onClick: () => void; children: React.ReactNode }) {
  return <button className={`nav-button ${active ? 'active' : ''}`} onClick={onClick} aria-current={active ? 'page' : undefined}><Icon size={16} /><span>{children}</span></button>
}

type LiveReviewProps = {
  playing: boolean; onTogglePlayback: () => void
  serviceMode: ServiceMode; workspaceError: string; activeRun: RunSummary | null; activeRunId: string; frames: InferenceRecord[]
  framesState: 'idle' | 'loading' | 'ready' | 'empty' | 'error'; framesError: string; selectedFrame: InferenceRecord | null
  selectedFrameId: string; setSelectedFrameId: (id: string) => void; frameDetailWarning: string; modelView: ModelView
  setModelView: (view: ModelView) => void; models: ModelSummary[]; e2ModelId: string; e1ModelId: string
  inspectModel: (kind: 'e2' | 'e1', id: string) => void; modelDetailNote: string; intervention: Intervention
  setIntervention: (value: Intervention) => void; onOpenChooser: () => void; onOpenEvidence: () => void
  onUseOffline: () => void; onRetry: () => void
}

function LiveReview(props: LiveReviewProps) {
  const [demoView, setDemoView] = useState<'static' | 'moving'>('static')
  const { serviceMode, workspaceError, activeRun, activeRunId, frames, framesState, framesError, selectedFrame, selectedFrameId, setSelectedFrameId, frameDetailWarning, modelView, setModelView, models, e2ModelId, e1ModelId, inspectModel, modelDetailNote, intervention, setIntervention, onOpenChooser, onOpenEvidence, onUseOffline, onRetry } = props
  const e2Models = models.filter((model) => modelKind(model.model_id) === 'e2')
  const e1Models = models.filter((model) => modelKind(model.model_id) === 'e1')
  const liveFrame = selectedFrame
  return <div className="page live-page">
    {framesState === 'loading' && <LoadingState label="Loading replay manifest…" />}
    {framesState === 'error' && <ErrorState title="Replay could not be loaded" message={framesError} onRetry={onRetry} secondary={{ label: 'Open AU-AIR replay', action: onUseOffline }} />}
    {framesState === 'empty' && <EmptyState onUseOffline={onUseOffline} />}
    {framesState === 'ready' && selectedFrame && <>
      <div className="segmented demo-view-switch" role="group" aria-label="Demo view">
        <button className={demoView === 'static' ? 'selected' : ''} aria-pressed={demoView === 'static'} onClick={() => setDemoView('static')}>Static</button>
        <button className={demoView === 'moving' ? 'selected' : ''} aria-pressed={demoView === 'moving'} onClick={() => {
          if (props.playing) props.onTogglePlayback()
          setDemoView('moving')
        }}>Moving</button>
      </div>
      {demoView === 'moving' ? <MovingDemo /> : <>
      <section className="review-panel"><div className="compare-head"><div><p className="eyebrow">Comparison lens — same frame / same protocol</p><h1>Live aerial review</h1></div><div className="segmented" role="group" aria-label="Model comparison view">{([['e1', 'E1 RGB'], ['e2', `${stateModelLabel(selectedFrame)} + STATE`], ['split', 'Split']] as const).map(([id, label]) => <button key={id} aria-pressed={modelView === id} className={modelView === id ? 'selected' : ''} onClick={() => setModelView(id)}>{label}</button>)}</div></div>
        {liveFrame && <div key={modelView} className={`feed-layout view-${modelView}`}><div className="feeds">{modelView !== 'e2' && <Feed kind="e1" modelId={e1ModelId} frame={liveFrame} offline={activeRunId === OFFLINE_RUN_ID} intervention={intervention} />}{modelView !== 'e1' && <Feed kind="e2" modelId={e2ModelId} frame={liveFrame} offline={activeRunId === OFFLINE_RUN_ID} intervention={intervention} />}</div><DetectionDetail frame={liveFrame} modelView={modelView} intervention={intervention} /></div>}
      </section>
      {frameDetailWarning && <div className="micro-warning"><AlertTriangle size={13} />{frameDetailWarning}</div>}
      <Timeline frames={frames} selectedFrameId={selectedFrameId} onSelect={setSelectedFrameId} playing={props.playing} onTogglePlayback={props.onTogglePlayback} />
      {framesError && <div className="micro-warning" role="status">{framesError}</div>}
      <section className="history-panel"><div className="panel-heading"><div><p className="eyebrow">History capture</p><h2>Selected timeline frame</h2></div><span key={`${selectedFrame.frame_id}-status`} className={`quality-badge ${frameStatus(selectedFrame)}`}>{frameStatus(selectedFrame) === 'invalid' ? 'Invalid' : frameStatus(selectedFrame) === 'delayed' ? 'Flagged' : 'Valid'}</span></div><div key={selectedFrame.frame_id} className="history-grid"><FrameVisual frame={selectedFrame} detections={selectedFrame.detections} label="Recorded replay evidence" /><HistoryDetail frame={selectedFrame} /></div></section>
      <details className="review-settings"><summary>Review controls · {serviceMode === 'offline' ? 'AU-AIR development replay · cached evidence' : 'Recording and models'}</summary>
        {workspaceError && <div className="inline-notice amber"><AlertTriangle size={17} /><div><strong>{serviceMode === 'offline' ? 'API unavailable — bundled AU-AIR development replay' : 'Partial workspace response'}</strong><span>{workspaceError}</span></div><button onClick={onRetry}><RefreshCw size={14} /> Retry</button></div>}
        <div className="model-selectors" aria-label="Comparison models"><label>E2 model<select value={e2ModelId} onChange={(event) => void inspectModel('e2', event.target.value)}>{(e2Models.length ? e2Models : models).map((model) => <option key={model.model_id}>{model.model_id}</option>)}</select></label><label>E1 model<select value={e1ModelId} onChange={(event) => void inspectModel('e1', event.target.value)}>{(e1Models.length ? e1Models : models).map((model) => <option key={model.model_id}>{model.model_id}</option>)}</select></label>{modelDetailNote && <span className="model-note">{modelDetailNote}</span>}</div>
        <InterventionPanel value={intervention} onChange={setIntervention} />
      </details>
      </>}
    </>}
  </div>
}

function Feed({ kind, modelId, frame, offline, intervention }: { kind: 'e2' | 'e1'; modelId: string; frame: InferenceRecord; offline: boolean; intervention: Intervention }) {
  const paired = Boolean(frame.rgb_detections && frame.state_detections)
  const matches = paired || modelKind(frame.model_id) === kind || modelKind(frame.model_id) === 'other'
  const available = offline || matches
  const detections = offline || paired ? fixtureDetections(frame, kind) : available ? frame.detections : []
  return <article className={`feed-card feed-${kind}`}><header><div><p>{kind === 'e1' ? 'E1 RGB stream' : `${stateModelLabel(frame)} + state stream`}</p><h3>{kind === 'e1' ? 'E1 · FCOS — ResNet-50 FPN' : `${stateModelLabel(frame)} · FCOS + gated FiLM`}</h3></div><Badge tone={available ? sourceTone(frame.prediction_source) : 'neutral'}>{available ? sourceLabel(frame.prediction_source) : 'NO OUTPUT'}</Badge></header><div className="feed-visual"><FrameVisual frame={frame} detections={detections} label={`${kind === 'e1' ? 'E1' : stateModelLabel(frame)} frame output`} compact />{!available && <div className="unavailable-overlay"><Box size={20} /><strong>No replay output</strong><span>{modelId || `${kind.toUpperCase()} model`} has no record for this frame.</span></div>}</div><footer><span><b>{detections.length}</b> detections</span><span>{kind === 'e2' ? `${intervention === 'recorded' ? 'Recorded state' : `Intervention: ${intervention}`}` : 'RGB inference'}{frame.latency.end_to_end_ms != null ? ` · ${frame.latency.end_to_end_ms.toFixed(0)} ms` : ''}</span></footer></article>
}

function FrameVisual({ frame, detections, label, compact = false }: { frame: InferenceRecord; detections: Detection[]; label: string; compact?: boolean }) {
  const image = replayImageUrl(frame)
  const [failedImage, setFailedImage] = useState('')
  const [loadedImage, setLoadedImage] = useState('')
  const [bounds, setBounds] = useState({ width: 0, height: 0 })
  const visualRef = useRef<HTMLDivElement>(null)
  const { width, height } = frame.original_size
  const hasImage = Boolean(image && failedImage !== image)
  useEffect(() => {
    if (!visualRef.current || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(([entry]) => setBounds({ width: entry.contentRect.width, height: entry.contentRect.height }))
    observer.observe(visualRef.current)
    return () => observer.disconnect()
  }, [])
  const fitted = hasImage && bounds.width > 0 ? containImage(bounds.width, bounds.height, width, height) : { inset: 0 }
  return <div ref={visualRef} className={`frame-visual ${compact ? 'compact' : ''}`} role="img" aria-label={`${label}: ${detections.length} detections from ${sourceLabel(frame.prediction_source).toLowerCase()} evidence`}>
    {!hasImage && <div className="protocol-canvas" aria-hidden><span className="terrain terrain-a" /></div>}
    <div className="frame-image-space" style={fitted}>
    {hasImage && <img key={image} src={image} alt={`AU-AIR replay frame ${frame.frame_id}`} onLoad={() => setLoadedImage(image!)} onError={() => setFailedImage(image!)} style={{ opacity: loadedImage === image ? 1 : 0 }} />}
    {(!hasImage || loadedImage === image) && detections.map((detection, index) => { const [x1, y1, x2, y2] = detection.box_xyxy; const color = detection.class_name === 'Human' ? 'lime' : index % 2 ? 'cyan' : 'amber'; return <span key={`${frame.frame_id}-${detection.class_name}-${index}`} className={`detection-box ${color}`} style={{ left: `${x1 / width * 100}%`, top: `${y1 / height * 100}%`, width: `${(x2 - x1) / width * 100}%`, height: `${(y2 - y1) / height * 100}%` }}><i>{detectionLabel(detection)}</i></span> })}
    </div>
    {hasImage && loadedImage !== image && <span className="image-loading"><LoaderCircle className="spin" size={16} />Loading source frame</span>}
    {!compact && <span className="capture-label">{frame.prediction_source === 'fixture' ? 'FIXTURE FRAME' : 'RECORDED FRAME'}</span>}
    {!hasImage && <span className="canvas-label">{frame.prediction_source === 'fixture' ? 'Illustrative fixture · not measured output' : 'Illustration · source image unavailable'}</span>}
    <span className="frame-id-label">FRAME {frameNumber(frame)}</span>
    {!compact && <span className="capture-time">CAPTURE {formatTime(frame.source_time_ms)}</span>}
  </div>
}

function DetectionDetail({ frame, modelView, intervention }: { frame: InferenceRecord; modelView: ModelView; intervention: Intervention }) {
  const effectiveMode = intervention === 'missing' ? 'state_masked' : frame.input_mode
  const effectiveAlignment = intervention === 'delayed' ? 'injected_delay' : intervention === 'invalid' ? 'unavailable' : frame.metadata_alignment
  return <aside className="detection-detail"><div className="detail-head"><div><p>Live frame detection detail</p><h3>Frame {frameNumber(frame)}</h3></div><span>{formatTime(frame.source_time_ms)}</span></div>{(['e2', 'e1'] as const).filter((kind) => modelView === 'split' || modelView === kind).map((kind) => {
    const offline = Boolean(frame.rgb_detections && frame.state_detections)
    const available = offline || modelKind(frame.model_id) === kind || modelKind(frame.model_id) === 'other'
    const detections = offline ? fixtureDetections(frame, kind) : available ? frame.detections : []
    return <section className={`detail-model detail-${kind}`} key={kind}><div className="detail-model-heading"><span>{kind === 'e1' ? 'E1 / RGB' : `${stateModelLabel(frame)} / STATE`}</span><span>{available ? sourceLabel(frame.prediction_source).toLowerCase() : 'no output'}</span></div><div className="detail-list">{detections.map((item, index) => <div key={index}><span>{item.class_name}</span><strong>{scoreLabel(item.raw_score)}</strong></div>)}{!available && <p className="empty-copy">No replay output for this model.</p>}</div></section>
  })}<div className="state-summary"><p>Scene state</p><strong>{intervention === 'recorded' ? effectiveMode.replace(/_/g, ' ') : `SIMULATED ${intervention.toUpperCase()}`}</strong><span>{frame.state ? `AU-AIR · altitude ${frame.state[0].toFixed(2)} m` : effectiveAlignment.replace(/_/g, ' ')} · {frame.original_size.width} × {frame.original_size.height}</span></div></aside>
}

function Timeline({ frames, selectedFrameId, onSelect, playing, onTogglePlayback }: { frames: InferenceRecord[]; selectedFrameId: string; onSelect: (id: string) => void; playing: boolean; onTogglePlayback: () => void }) {
  const selectedIndex = Math.max(0, frames.findIndex((frame) => frame.frame_id === selectedFrameId))
  return <section className="timeline-panel"><div className="panel-heading"><SectionLabel>Frame timeline</SectionLabel><div className="playback-controls"><span>{frames.length < 2 ? '1 frame · waiting for API frames' : `${selectedIndex + 1} / ${frames.length} · ${playing ? 'Replay · 1.2s / frame' : 'Paused'}`}</span><button onClick={onTogglePlayback} disabled={frames.length < 2} aria-label={playing ? 'Pause replay' : 'Play replay'} aria-pressed={playing}>{playing ? <Pause size={12} /> : <Play size={12} />}{playing ? 'Pause' : 'Play'}</button></div></div><div className="timeline-items"><div className="timeline-track" style={{ '--frame-count': frames.length, '--selected-index': selectedIndex } as React.CSSProperties}><div className="timeline-selection" aria-hidden />{frames.map((frame, index) => <button key={frame.frame_id} className={selectedFrameId === frame.frame_id ? 'selected' : ''} onClick={() => onSelect(frame.frame_id)} onKeyDown={(event) => {
    const next = event.key === 'ArrowLeft' ? index - 1 : event.key === 'ArrowRight' ? index + 1 : event.key === 'Home' ? 0 : event.key === 'End' ? frames.length - 1 : -1
    if (next >= 0 && next < frames.length) { event.preventDefault(); onSelect(frames[next].frame_id); (event.currentTarget.parentElement?.querySelectorAll('button')[next] as HTMLButtonElement)?.focus() }
  }} aria-label={`Frame ${frameNumber(frame)}, ${formatTime(frame.source_time_ms)}, ${frameStatus(frame)}`} aria-pressed={selectedFrameId === frame.frame_id}><span>{frameNumber(frame)}</span><i className={frameStatus(frame) === 'invalid' ? 'invalid' : frame.quality_flags.length ? 'flagged' : ''} /><small>{formatTime(frame.source_time_ms)}</small></button>)}</div></div></section>
}

function HistoryDetail({ frame }: { frame: InferenceRecord }) {
  const offline = Boolean(frame.rgb_detections && frame.state_detections)
  return <aside className="history-detail"><SectionLabel>History frame detection detail</SectionLabel><h3>Frame {frameNumber(frame)} · {formatTime(frame.source_time_ms)}</h3><div className="history-scene"><SectionLabel>Scene</SectionLabel><p>{frame.input_mode.replace(/_/g, ' ')}</p><p>{frame.state ? `Altitude ${frame.state[0].toFixed(2)} m` : frame.metadata_alignment.replace(/_/g, ' ')} · {frame.original_size.width} × {frame.original_size.height}</p></div>{(['e2', 'e1'] as const).map((kind) => {
    const available = offline || modelKind(frame.model_id) === kind || modelKind(frame.model_id) === 'other'
    const detections = offline ? fixtureDetections(frame, kind) : available ? frame.detections : []
    return <div key={kind} className={`history-model ${kind}`}><strong>{kind === 'e1' ? 'E1 RGB' : `${stateModelLabel(frame)} + STATE`} · {available ? `${detections.length} detections` : 'No output'}</strong><p>{detections.map(detectionLabel).join(' · ') || 'No replay evidence available'}</p>{kind === 'e2' && <p>{frame.prediction_source === 'fixture' ? 'Illustrative fixture — not measured evidence' : `${sourceLabel(frame.prediction_source)} · ${frame.latency.end_to_end_ms == null ? 'Latency not recorded' : `${frame.latency.end_to_end_ms.toFixed(1)} ms`}`}</p>}</div>
  })}</aside>
}

function InterventionPanel({ value, onChange }: { value: Intervention; onChange: (value: Intervention) => void }) {
  const items: Array<[Intervention, string, string]> = [['recorded', 'Recorded state', 'Use the replay contract unchanged'], ['missing', 'Missing state', 'Mask the displayed state vector'], ['invalid', 'Invalid state', 'Show a sanitized invalid-state path'], ['delayed', 'Injected delay', 'Show delayed metadata alignment']]
  return <section className="intervention-panel"><div><SectionLabel icon={SlidersHorizontal}>Input intervention</SectionLabel><h2>Stress the pairing, keep the claim honest.</h2><p>These controls change the review lens only. They do not run inference or create a new metric.</p></div><div className="intervention-options">{items.map(([id, title, detail]) => <button key={id} className={value === id ? 'selected' : ''} onClick={() => onChange(id)} aria-pressed={value === id}><i /><span><strong>{title}</strong><small>{detail}</small></span></button>)}</div></section>
}

function ResultsEvidence({ mode, reports, reportsFallback, models, reviews, error, selectedReport, reportDetailState, reportDetailError, onInspectReport, onCloseReport, onRetry }: { mode: ServiceMode; reports: EvaluationReport[]; reportsFallback: boolean; models: ModelSummary[]; reviews: Review[]; error: string; selectedReport: EvaluationReport | null; reportDetailState: 'idle' | 'loading' | 'error'; reportDetailError: string; onInspectReport: (report: EvaluationReport) => void; onCloseReport: () => void; onRetry: () => void }) {
  const sorted = [...reports].sort((a, b) => { const rank = (report: EvaluationReport) => /full-pass/.test(report.model_id) ? 0 : /masked|shuffled/.test(report.model_id) ? 1 : 2; return rank(a) - rank(b) || a.model_id.localeCompare(b.model_id) })
  const comparison = documentedReports.map((snapshot) => sorted.find((report) => report.model_id === snapshot.model_id && report.partition === 'development')).filter((report): report is EvaluationReport => Boolean(report))
  const snapshotCount = sorted.filter((report) => report.documentedFallback).length
  return <div className="page results-page"><header className="page-header"><div><p className="eyebrow">Results / Evidence register</p><h1>Measured evidence, with its limits attached.</h1><p>Every result is labelled by what exists in the workspace today.</p></div><Badge tone="amber">NO FABRICATED METRICS</Badge></header>
    {(error || reportsFallback) && <div className="inline-notice amber"><AlertTriangle size={18} /><div><strong>{mode === 'offline' ? 'Live reports unavailable' : 'Using documented report fallback'}</strong><span>{error || 'Missing API reports are supplemented by labelled snapshots of committed development evaluations. API-discovered reports remain preferred.'}</span></div><button onClick={onRetry}><RefreshCw size={14} /> Retry</button></div>}
    <section className="split-evidence panel"><div className="panel-heading"><div><SectionLabel icon={Server}>Dataset evidence</SectionLabel><h2>AU-AIR split register</h2></div><Badge tone="green">32,823 TOTAL FRAMES</Badge></div><div className="split-bar" aria-label="Training 56.43 percent, development 17.47 percent, sealed final test 26.10 percent"><span className="train" style={{ width: '56.43%' }} /><span className="dev" style={{ width: '17.47%' }} /><span className="test" style={{ width: '26.10%' }} /></div><div className="split-stats"><div><i className="train" /><span>Training<strong>18,523</strong><small>56.43%</small></span></div><div><i className="dev" /><span>Development<strong>5,734</strong><small>17.47%</small></span></div><div><i className="test" /><span>Final test<strong>8,566</strong><small>26.10% · sealed</small></span></div></div></section>
    {comparison.length > 0 && <section className="headline-comparison panel"><div className="panel-heading"><div><SectionLabel icon={GitCompare}>Review 2 comparison</SectionLabel><h2>Development operating evidence</h2></div><Badge tone="amber">FINAL TEST SEALED</Badge></div><div className="comparison-table"><div className="table-head"><span>Model</span><span>AP50</span><span>Recall</span><span>F1</span><span>Detection accuracy</span><span>Threshold</span></div>{comparison.map((report) => { const metrics = reportMetrics(report); return <button key={report.model_id} onClick={() => onInspectReport(report)}><span><strong>{report.model_id.match(/(?:^|-)e[1-4](?:-|$)/)?.[0].replace(/-/g, '').toUpperCase()}</strong><small>{report.model_id}</small><small>Development evidence · {report.documentedFallback ? 'documented snapshot' : 'API report'}</small></span><b>{formatMetric(metrics.ap50)}</b><b>{formatMetric(metrics.recall)}</b><b>{formatMetric(metrics.f1)}</b><b>{formatMetric(metrics.detectionAccuracy)}</b><b>{formatMetric(metrics.threshold, 2)}</b></button> })}</div><p className="definition-note"><Info size={13} /> Detection accuracy is TP/(TP+FP+FN). It is not image-classification accuracy. AP50 is pooled development evidence; recall, F1 and detection accuracy are shown at the development-selected best-F1 operating point.</p></section>}
    <div className="register-heading"><div><p className="eyebrow">Saved model reports</p><h2>Repository-discovered register</h2></div><span>{sorted.length} reports · {sorted.length - snapshotCount} API-discovered · {snapshotCount} documented snapshot{snapshotCount === 1 ? '' : 's'}</span></div>
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
  const modalRef = useModalFocus(onClose)
  const metrics = reportMetrics(report)
  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}><section ref={modalRef} className="report-dialog" role="dialog" aria-modal="true" aria-labelledby="report-title"><header><div><p className="eyebrow">Evaluation report detail</p><h2 id="report-title">{report.model_id}</h2></div><button className="icon-button" onClick={onClose} aria-label="Close report detail"><X size={18} /></button></header>{state === 'loading' ? <LoadingState label="Verifying report detail…" /> : <div className="report-dialog-body">{state === 'error' && <div className="micro-warning"><AlertTriangle size={13} />{error}</div>}<div className="dialog-metrics"><div><span>AP50</span><strong>{formatMetric(metrics.ap50)}</strong></div><div><span>Recall</span><strong>{formatMetric(metrics.recall)}</strong></div><div><span>Detection accuracy</span><strong>{formatMetric(metrics.detectionAccuracy)}</strong></div><div><span>Latency p95</span><strong>{metrics.latency == null ? '—' : `${metrics.latency.toFixed(2)} ms`}</strong></div></div><dl className="report-metadata"><div><dt>Partition</dt><dd>{report.partition}</dd></div><div><dt>Checkpoint</dt><dd>{report.checkpoint_id}</dd></div><div><dt>Protocol</dt><dd>{shortHash(report.protocol_sha256)}</dd></div>{report.documentedFallback && <div><dt>Snapshot source</dt><dd>{String(report.config.evidence_source ?? 'Committed development report')}</dd></div>}<div><dt>Final test</dt><dd>{report.final_test_unsealed ? 'Unsealed by report' : 'Sealed'}</dd></div></dl><h3>Limitations</h3><ul>{report.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul></div>}</section></div>
}

function Architecture({ capabilities, readiness, mode }: { capabilities: CapabilityResponse | null; readiness: ReadinessResponse | null; mode: ServiceMode }) {
  const pipeline = [['01', 'Source recordings', 'AU-AIR / VisDrone', 'current'], ['02', 'Validation + pairing', 'schema · timestamps · splits', 'current'], ['03', 'Model gate + FiLM', 'FCOS conditioned features', 'current'], ['04', 'Inference API', 'FastAPI · replay · reviews', 'current'], ['05', 'Review console', 'this frontend · evidence', 'current'], ['06', 'Storage + monitoring', 'object store · Postgres · drift', 'planned']] as const
  const details = [
    ['01 — Source recordings', 'AU-AIR: 32,823 matched RGB frames at 1920×1080 with 5 Hz telemetry. Training uses 18,523 frames; development uses 5,734; 8,566 final-test frames remain sealed.', ['AU-AIR dataset', 'VisDrone RGB', '1920×1080', '5 Hz source']],
    ['02 — Validation + pairing', 'Schema validation, timestamp alignment, and declared split construction preserve frame/state provenance before any model call.', ['Schema validation', 'Timestamp alignment', 'Split construction', 'Sealed test roots']],
    ['03 — Model gate + FiLM', 'E1 is the image-only FCOS baseline. E2 adds gated FiLM conditioning from flight state; invalid state is sanitized before encoding.', ['FCOS · ResNet-50 FPN', 'Gated FiLM', '8-feature state vector', 'Invalid-state sanitization']],
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
  const drawerRef = useRef<HTMLElement>(null)
  useEffect(() => {
    const drawer = drawerRef.current
    drawer?.querySelector<HTMLButtonElement>('button')?.focus()
    const trap = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || !drawer) return
      const items = [...drawer.querySelectorAll<HTMLElement>('button:not(:disabled), select, textarea, [tabindex="0"]')]
      const first = items[0], last = items[items.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
    }
    drawer?.addEventListener('keydown', trap)
    return () => { drawer?.removeEventListener('keydown', trap); window.requestAnimationFrame(() => document.querySelector<HTMLButtonElement>('.header-evidence')?.focus()) }
  }, [])
  return <aside ref={drawerRef} id="evidence-drawer" className="evidence-drawer" role="dialog" aria-modal="true" aria-labelledby="evidence-title"><header><div><p className="eyebrow">Evidence Drawer</p><h2 id="evidence-title">Frame evidence</h2></div><button className="icon-button" onClick={onClose} aria-label="Close evidence drawer"><X size={18} /></button></header><div className="drawer-body">{frame && run ? <>
    <section className="provenance-card"><div><SectionLabel>Provenance</SectionLabel><Badge tone={sourceTone(frame.prediction_source)}>{sourceLabel(frame.prediction_source)}</Badge></div><p>{frame.rgb_model_id ? 'AU-AIR development replay · cached GPU-computed RGB and state-model detections. Display threshold 0.45; not a benchmark claim.' : frame.prediction_source === 'fixture' ? 'Fixture evidence validates the interface and review workflow only. It is not a measured model result.' : frame.prediction_source === 'cached' ? 'Recorded output returned by the replay API.' : 'Computed output returned by the inference runtime.'}</p></section>
    <SectionLabel>Frame metadata</SectionLabel><dl className="evidence-metadata"><div><dt>Frame ID</dt><dd>{frame.frame_id}</dd></div><div><dt>Source time</dt><dd>{formatTime(frame.source_time_ms)}</dd></div><div><dt>Run</dt><dd>{run.run_id}</dd></div><div><dt>Model / protocol</dt><dd>{frame.model_id}<small>{shortHash(frame.protocol_sha256)}</small></dd></div><div><dt>Image space</dt><dd>{frame.original_size.width} × {frame.original_size.height}</dd></div><div><dt>State source</dt><dd>{frame.metadata_alignment.replace(/_/g, ' ')}</dd></div><div><dt>Review lens</dt><dd>{intervention}</dd></div><div><dt>Quality flags</dt><dd>{frame.quality_flags.length ? frame.quality_flags.join(', ') : 'none'}</dd></div></dl>
    {reviews.length > 0 && <section className="prior-reviews"><SectionLabel icon={ShieldCheck}>Prior reviews</SectionLabel>{reviews.map((review) => <button key={review.review_id} onClick={() => onInspectReview(review)}><span>{review.decision.replace('_', ' ')}</span><small>{new Date(review.created_at).toLocaleString()}</small></button>)}{selectedReview && <p>Verified review: {selectedReview.review_id.slice(0, 16)}…</p>}</section>}
    <div className="drawer-divider" /><SectionLabel icon={ShieldCheck}>Operator review</SectionLabel><label className="field-label" htmlFor="review-decision">Decision</label><select id="review-decision" value={decision} onChange={(event) => setDecision(event.target.value as ReviewDecision | '')}><option value="">— Select decision —</option><option value="accept">Accept evidence</option><option value="needs_review">Needs further review</option><option value="reject">Reject evidence</option></select><label className="field-label" htmlFor="review-comment">Comment <span>optional</span></label><textarea id="review-comment" rows={4} maxLength={4000} value={comment} onChange={(event) => setComment(event.target.value)} placeholder="Add observation or context…" />
    {reviewMessage && <div className={`save-message ${reviewState}`} role="status">{reviewState === 'saved' ? <Check size={14} /> : reviewState === 'error' ? <AlertTriangle size={14} /> : <LoaderCircle className="spin" size={14} />}{reviewMessage}</div>}
    <button className="button primary full" disabled={reviewState === 'saving'} onClick={onSave}><Save size={15} />{reviewState === 'saving' ? 'Saving…' : 'Save decision to API'}</button><button className="button secondary full" onClick={onExport}><Download size={15} />Export local JSON copy</button><p className="drawer-note"><Info size={12} />Saving writes the declared human decision only. No risk label is inferred.</p>
  </> : <div className="drawer-empty"><Archive size={22} /><p>Select a frame to inspect its evidence.</p></div>}</div></aside>
}

function RecordingChooser({ runs, selected, closeRef, onChoose, onClose }: { runs: RunSummary[]; selected: string; closeRef: React.RefObject<HTMLButtonElement | null>; onChoose: (id: string) => void; onClose: () => void }) {
  const modalRef = useModalFocus(onClose)
  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}><section ref={modalRef} className="chooser-dialog" role="dialog" aria-modal="true" aria-labelledby="chooser-title"><header><div><p className="eyebrow">Replay source</p><h2 id="chooser-title">Choose a recording</h2></div><button ref={closeRef} className="icon-button" onClick={onClose} aria-label="Close recording chooser"><X size={18} /></button></header><div className="chooser-list">{runs.map((run) => <button key={run.run_id} className={run.run_id === selected ? 'selected' : ''} onClick={() => onChoose(run.run_id)}><span className="option-icon">{run.run_id === selected ? <Check size={15} /> : <Archive size={15} />}</span><span><strong>{run.run_id}</strong><small>{run.offline ? 'AU-AIR development replay · cached GPU-computed evidence' : `${run.frame_count} API frame${run.frame_count === 1 ? '' : 's'}`}</small></span><Badge tone={sourceTone(run.prediction_source)}>{run.offline ? 'CACHED REPLAY' : sourceLabel(run.prediction_source)}</Badge></button>)}</div></section></div>
}

function LoadingState({ label }: { label: string }) { return <div className="loading-state" role="status"><BrandMark /><LoaderCircle className="spin" size={18} /><span>{label}</span></div> }
function ErrorState({ title, message, onRetry, secondary }: { title: string; message: string; onRetry: () => void; secondary?: { label: string; action: () => void } }) { return <div className="state-card error-state"><AlertTriangle size={24} /><h2>{title}</h2><p>{message}</p><div><button className="button primary" onClick={onRetry}><RefreshCw size={14} />Retry</button>{secondary && <button className="button secondary" onClick={secondary.action}>{secondary.label}</button>}</div></div> }
function EmptyState({ onUseOffline }: { onUseOffline: () => void }) { return <div className="state-card"><BrandMark /><Archive size={20} /><h2>No frames in this recording</h2><p>The API returned an empty replay. No evidence has been synthesized.</p><button className="button secondary" onClick={onUseOffline}>Open AU-AIR development replay</button></div> }
