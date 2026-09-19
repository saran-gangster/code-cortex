// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import MovingDemo from './MovingDemo'
import { movingScenes } from './moving-scenes'

afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('autoplays all four images at 1.2s even without animation callbacks, loops, pauses and seeks', async () => {
  vi.useFakeTimers()
  vi.stubGlobal('Image', class {
    onload?: () => void
    set src(_value: string) { Promise.resolve().then(() => this.onload?.()) }
  })
  // A throttled canvas must not stall frame playback.
  vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1))
  vi.stubGlobal('cancelAnimationFrame', vi.fn())
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    clearRect: vi.fn(), save: vi.fn(), translate: vi.fn(), rotate: vi.fn(),
    scale: vi.fn(), drawImage: vi.fn(), strokeRect: vi.fn(), fillRect: vi.fn(),
    fillText: vi.fn(), restore: vi.fn(), measureText: () => ({ width: 100 }),
  } as unknown as CanvasRenderingContext2D)
  render(<MovingDemo />)
  await act(async () => { await Promise.resolve(); await Promise.resolve() })
  expect(new Set(movingScenes.map((scene) => scene.image_url)).size).toBe(4)
  expect(screen.getByRole('button', { name: 'Pause moving drone demo' })).toBeTruthy()
  const position = () => screen.getByLabelText('Moving frame position').textContent
  expect(position()).toBe('1 / 4')
  for (const expected of ['2 / 4', '3 / 4', '4 / 4', '1 / 4']) {
    await act(async () => { await vi.advanceTimersByTimeAsync(1200) })
    expect(position()).toBe(expected)
  }
  fireEvent.click(screen.getByRole('button', { name: 'Moving frame 3' }))
  expect(position()).toBe('3 / 4')
  await act(async () => { await vi.advanceTimersByTimeAsync(3600) })
  expect(position()).toBe('3 / 4')
  fireEvent.click(screen.getByRole('button', { name: 'Play moving drone demo' }))
  await act(async () => { await vi.advanceTimersByTimeAsync(1200) })
  expect(position()).toBe('4 / 4')
})
