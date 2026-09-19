import { useEffect, useRef, useState } from 'react'
import { Pause, Play } from 'lucide-react'
import { movingScenes } from './moving-scenes'
import './moving-demo.css'

const FRAME_DURATION_MS = 1200

// Camera motion belongs only to this secondary demonstration. Image and boxes
// share the same transform so camera movement cannot detach the annotations.
export default function MovingDemo() {
  const rgbCanvas = useRef<HTMLCanvasElement>(null)
  const stateCanvas = useRef<HTMLCanvasElement>(null)
  const images = useRef<HTMLImageElement[]>([])
  const elapsed = useRef(0)
  const [ready, setReady] = useState(false)
  const [playing, setPlaying] = useState(true)
  const [frameIndex, setFrameIndex] = useState(0)

  useEffect(() => {
    let cancelled = false
    const pending = movingScenes.map((frame) => new Promise<HTMLImageElement>((resolve, reject) => {
      const image = new Image()
      image.onload = () => resolve(image)
      image.onerror = reject
      image.src = frame.image_url!
    }))
    Promise.all(pending).then((loaded) => {
      if (!cancelled) { images.current = loaded; setReady(true) }
    }).catch(() => { /* Leave playback disabled if the source images cannot load. */ })
    return () => { cancelled = true }
  }, [])

  // Frame playback uses the same cadence as Static, independent of canvas FPS.
  useEffect(() => {
    if (!ready || !playing) return
    const timer = window.setTimeout(() => setFrameIndex((index) => (index + 1) % movingScenes.length), FRAME_DURATION_MS)
    return () => window.clearTimeout(timer)
  }, [ready, playing, frameIndex])

  useEffect(() => {
    if (!ready) return
    let request = 0
    let previous: number | undefined
    const paint = (now: number) => {
      if (playing && previous !== undefined) elapsed.current += Math.min(now - previous, 100)
      previous = now
      const milliseconds = elapsed.current
      const frame = movingScenes[frameIndex]
      const { width, height } = frame.original_size
      const phase = milliseconds / 1000
      const zoom = 1.22 + 0.04 * Math.sin(phase * 0.5)
      const x = Math.sin(phase * 0.55) * width * 0.045
      const y = Math.sin(phase * 0.4) * height * 0.025
      const roll = Math.sin(phase * 0.45) * 0.018
      for (const [canvas, kind] of [[rgbCanvas.current, 'e1'], [stateCanvas.current, 'e2']] as const) {
        const context = canvas?.getContext('2d')
        if (!canvas || !context) continue
        context.clearRect(0, 0, width, height)
        context.save()
        context.translate(width / 2 + x, height / 2 + y)
        context.rotate(roll)
        context.scale(zoom, zoom)
        context.translate(-width / 2, -height / 2)
        context.drawImage(images.current[frameIndex], 0, 0, width, height)
        for (const detection of frame.annotations.filter((item) => kind === 'e2' || item.class_name === 'Car')) {
          const [left, top, right, bottom] = detection.box_xyxy
          const color = detection.class_name === 'Van' ? '#80e4e0' : '#ffc571'
          const label = `${detection.class_name} ${detection.display_score.toFixed(2)}`
          context.strokeStyle = color
          context.lineWidth = 2 / zoom
          context.strokeRect(left, top, right - left, bottom - top)
          context.font = '22px sans-serif'
          context.fillStyle = color
          context.fillRect(left, top - 34, context.measureText(label).width + 16, 32)
          context.fillStyle = '#071017'
          context.fillText(label, left + 8, top - 11)
        }
        context.restore()
      }
      if (playing) request = requestAnimationFrame(paint)
    }
    paint(performance.now())
    return () => cancelAnimationFrame(request)
  }, [ready, playing, frameIndex])

  return <section className="moving-demo" aria-labelledby="moving-demo-title">
    <header className="panel-heading">
      <h2 id="moving-demo-title">Moving drone demo</h2>
      <div className="moving-demo-playback"><span aria-label="Moving frame position">{frameIndex + 1} / {movingScenes.length}</span><button className="button secondary" disabled={!ready} aria-label={playing ? 'Pause moving drone demo' : 'Play moving drone demo'} aria-pressed={playing} onClick={() => setPlaying((value) => !value)}>
        {playing ? <Pause size={14} /> : <Play size={14} />}{playing ? 'Pause' : 'Play'}
      </button></div>
    </header>
    <div className="moving-demo-feeds">
      <article><h3>E1 RGB</h3><canvas ref={rgbCanvas} width={1672} height={941} role="img" aria-label="E1 moving drone view" /></article>
      <article><h3>E4 + STATE</h3><canvas ref={stateCanvas} width={1672} height={941} role="img" aria-label="E4 moving drone view" /></article>
    </div>
    <div className="moving-demo-timeline" role="group" aria-label="Moving frame timeline">
      {movingScenes.map((scene, index) => <button key={scene.image_url} disabled={!ready} className={frameIndex === index ? 'selected' : ''} aria-label={`Moving frame ${index + 1}`} aria-pressed={frameIndex === index} onClick={() => { setPlaying(false); setFrameIndex(index) }}>
        <img src={scene.image_url} alt="" />
        <span>Frame {index + 1}<small>{(index * FRAME_DURATION_MS / 1000).toFixed(1)}s</small></span>
      </button>)}
    </div>
  </section>
}
