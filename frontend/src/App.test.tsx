// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { api } from './api'
import { documentedReports, offlineFrames, offlineModels } from './fixtures'
import { containImage, replayImageUrl } from './replay-image'

const flush = async () => { await act(async () => { await Promise.resolve(); await Promise.resolve() }) }
const tick = async (ms = 1200) => { await act(async () => { await vi.advanceTimersByTimeAsync(ms) }) }

beforeEach(() => {
  vi.useFakeTimers()
  window.location.hash = ''
  vi.spyOn(api, 'health').mockResolvedValue({ status: 'ok', version: 'test', model_ready: false, runtime_mode: 'fixture_replay' })
  vi.spyOn(api, 'readiness').mockResolvedValue({ ready: true, status: 'ready', computed_inference_ready: false, replay_ready: true })
  vi.spyOn(api, 'capabilities').mockResolvedValue({ api_version: '1', computed_inference: false, fixture_replay: true, review_persistence: true, supported_prediction_sources: ['cached'], endpoints: {} })
  vi.spyOn(api, 'runs').mockResolvedValue([{ run_id: 'api-replay', frame_count: 6, prediction_source: 'cached' }])
  vi.spyOn(api, 'frames').mockResolvedValue(offlineFrames)
  vi.spyOn(api, 'frame').mockImplementation(async (_run, id) => offlineFrames.find((frame) => frame.frame_id === id)!)
  vi.spyOn(api, 'models').mockResolvedValue(offlineModels)
  vi.spyOn(api, 'reports').mockResolvedValue(documentedReports.map((report) => ({ ...report, documentedFallback: false })))
  vi.spyOn(api, 'report').mockImplementation(async (id) => documentedReports.find((report) => report.report_id === id)!)
  vi.spyOn(api, 'reviews').mockResolvedValue([])
})
afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks() })

describe('real replay workflow', () => {
  it('defaults to real AU-AIR replay when API offers only fixtures or single-frame runs, preserving the chooser', async () => {
    vi.mocked(api.runs).mockResolvedValue([
      { run_id: 'fixture-run', frame_count: 1, prediction_source: 'fixture' },
      { run_id: 'single-computed-frame', frame_count: 1, prediction_source: 'computed' },
    ])
    render(<App />); await flush()
    expect(screen.getByRole('button', { name: 'Choose recording' }).textContent).toContain('AU-AIR development replay')
    expect(screen.getByRole('button', { name: 'Pause replay' })).toBeTruthy()
    await tick()
    expect(screen.getByRole('heading', { name: 'Frame 130' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Choose recording' })); await flush()
    const chooser = screen.getByRole('dialog', { name: 'Choose a recording' })
    expect(within(chooser).getByText('fixture-run')).toBeTruthy()
    expect(within(chooser).getByText('single-computed-frame')).toBeTruthy()
    fireEvent.keyDown(document.activeElement!, { key: 'Tab', shiftKey: true })
    expect(document.activeElement?.textContent).toContain('auair-development-replay')
    fireEvent.keyDown(document.activeElement!, { key: 'Escape' }); await flush()
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('advances API frames and synchronizes timeline, detail, images and history; pauses and seeks', async () => {
    render(<App />); await flush()
    expect(screen.getByRole('heading', { name: 'Frame 129' })).toBeTruthy()
    expect(document.querySelector('.feed-e1 footer')?.textContent).toContain('2 detections')
    expect(document.querySelector('.feed-e2 footer')?.textContent).toContain('1 detections')
    await tick()
    expect(screen.getByRole('heading', { name: 'Frame 130' })).toBeTruthy()
    expect(document.querySelector('.history-detail h3')?.textContent).toContain('00:26.0')
    expect(document.querySelector('.feed-e1 img')?.getAttribute('src')).toContain('0000130.jpg')
    fireEvent.click(screen.getByRole('button', { name: 'Pause replay' }))
    await tick(2400)
    expect(screen.getByRole('heading', { name: 'Frame 130' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Frame 133, 00:26.6, valid' }))
    expect(document.querySelector('.feed-e2 footer')?.textContent).toContain('0 detections')
    expect(screen.getByRole('button', { name: 'Play replay' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Play replay' })); await tick(); await tick()
    expect(screen.getByRole('heading', { name: 'Frame 129' })).toBeTruthy()
  })

  it('pauses for evidence and submits the exact selected API frame through the review contract', async () => {
    const review = { review_id: 'review-test', created_at: new Date().toISOString(), run_id: 'api-replay', frame_id: offlineFrames[0].frame_id, decision: 'needs_review' as const, comment: 'Workflow regression check' }
    const save = vi.spyOn(api, 'saveReview').mockResolvedValue(review)
    vi.spyOn(api, 'review').mockResolvedValue(review)
    render(<App />); await flush()
    fireEvent.click(screen.getByRole('button', { name: '⌁ Evidence [E]' })); await flush(); await tick(2400)
    const dialog = screen.getByRole('dialog', { name: 'Frame evidence' })
    expect(within(dialog).getByText(offlineFrames[0].frame_id)).toBeTruthy()
    fireEvent.change(screen.getByLabelText('Decision'), { target: { value: 'needs_review' } })
    fireEvent.change(screen.getByLabelText(/Comment/), { target: { value: review.comment } })
    fireEvent.click(screen.getByRole('button', { name: 'Save decision to API' })); await flush()
    expect(save).toHaveBeenCalledWith({ run_id: review.run_id, frame_id: review.frame_id, decision: review.decision, comment: review.comment })
    expect(screen.getByRole('status').textContent).toContain('Saved and verified')
  })

  it('does not invent advancement in a single-frame API run', async () => {
    vi.mocked(api.frames).mockResolvedValue([offlineFrames[0]])
    render(<App />); await flush(); await tick(3600)
    expect(screen.getByRole('heading', { name: 'Frame 129' })).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Play replay' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('uses the honest bundled cached sequence when the API is unavailable', async () => {
    vi.mocked(api.health).mockRejectedValue(new Error('offline'))
    render(<App />); await flush()
    expect(screen.getByRole('button', { name: 'Choose recording' }).textContent).toContain('AU-AIR development replay')
    expect(document.querySelector('.feed-e2 h3')?.textContent).toContain('E4')
    expect(document.querySelector('.feed-e1 .badge')?.textContent).toBe('CACHED')
    await tick()
    expect(screen.getByRole('heading', { name: 'Frame 130' })).toBeTruthy()
  })

  it('preserves model view switching and dynamically discovered reports', async () => {
    render(<App />); await flush()
    fireEvent.click(screen.getByRole('button', { name: 'E1 RGB' }))
    expect(document.querySelectorAll('.feed-card').length).toBe(1)
    fireEvent.click(screen.getByRole('button', { name: 'E4 + STATE' }))
    expect(document.querySelector('.feed-e1')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Results & evidence' })); await flush()
    expect(document.querySelectorAll('.report-card').length).toBe(4)
    expect(screen.getByRole('heading', { name: 'Repository-discovered register' })).toBeTruthy()
    fireEvent.click(screen.getAllByRole('button', { name: 'Inspect report' })[0]); await flush()
    expect(screen.getByRole('dialog')).toBeTruthy()
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Close report detail')
    fireEvent.keyDown(document.activeElement!, { key: 'Escape' }); await flush()
    expect(screen.queryByRole('dialog')).toBeNull()
  })
})

it('fits original-coordinate boxes to letterboxed image bounds', () => {
  expect(containImage(860, 270, 1920, 1080)).toEqual({ width: 480, height: 270, left: 190, top: 0 })
  expect(replayImageUrl(offlineFrames[0])).toContain('/assets/auair-demo/frame_20190905091750_x_0000129.jpg')
  expect(replayImageUrl({ ...offlineFrames[0], image_url: '/runs/image.jpg' })).toBe('http://127.0.0.1:8000/runs/image.jpg')
})
