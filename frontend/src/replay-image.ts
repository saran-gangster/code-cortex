import { API_BASE } from './api'
import type { InferenceRecord } from './types'

export function replayImageUrl(frame: InferenceRecord): string | undefined {
  const source = frame.image_data_url || frame.image_url
  if (!source) return undefined
  if (source.startsWith('data:image/')) return source
  // Bundled development images are served by Vite; API-relative URLs use the API origin.
  const base = source.startsWith('/assets/auair-demo/') ? window.location.origin : API_BASE
  try {
    const url = new URL(source, `${base}/`)
    return ['http:', 'https:'].includes(url.protocol) ? url.href : undefined
  } catch { return undefined }
}

export function containImage(width: number, height: number, imageWidth: number, imageHeight: number) {
  const scale = Math.min(width / imageWidth, height / imageHeight)
  const fittedWidth = imageWidth * scale
  const fittedHeight = imageHeight * scale
  return { width: fittedWidth, height: fittedHeight, left: (width - fittedWidth) / 2, top: (height - fittedHeight) / 2 }
}
