import { useEffect, useMemo, useRef, useState } from 'react'

function clamp01(n) {
  if (Number.isNaN(n)) return 0
  if (n < 0) return 0
  if (n > 1) return 1
  return n
}

function nodeLabel(node) {
  if (!node) return 'N'
  if (node.kind === 'line') return String(node.label ?? node.id)
  return node.role === 'RAIDER' ? `R${node.id}` : `D${node.id}`
}

function edgeLabel(edge) {
  if (!edge) return ''
  if (edge.kind === 'line') return String(edge.type ?? 'LINE')
  if (typeof edge.distance === 'number') return `${edge.distance.toFixed(2)}m`
  return typeof edge.weight === 'number' ? `${Math.round(edge.weight * 100)}%` : ''
}

export default function Graph2D({ graph, height = 360 }) {
  const wrapRef = useRef(null)
  const canvasRef = useRef(null)
  const pointerRef = useRef({ x: -9999, y: -9999, inside: false })
  const dataRef = useRef({ nodes: [], edges: [], meta: null })

  const [hovered, setHovered] = useState(null)
  const [tooltip, setTooltip] = useState({ x: 0, y: 0 })

  const data = useMemo(() => {
    const nodes = Array.isArray(graph?.nodes) ? graph.nodes : []
    const edges = Array.isArray(graph?.edges) ? graph.edges : []
    const meta = graph?.meta ?? null
    return { nodes, edges, meta }
  }, [graph])

  useEffect(() => {
    dataRef.current = data
  }, [data])

  useEffect(() => {
    const wrap = wrapRef.current
    const canvas = canvasRef.current
    if (!wrap || !canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let raf = 0
    const dpr = () => Math.min(window.devicePixelRatio || 1, 2)

    const palette = (isDark) => {
      return {
        bg1: isDark ? 'rgba(2,6,23,0.55)' : 'rgba(255,255,255,0.88)',
        bg2: isDark ? 'rgba(15,23,42,0.35)' : 'rgba(250,250,249,0.9)',
        border: isDark ? 'rgba(51,65,85,0.75)' : 'rgba(226,232,240,0.95)',
        text: isDark ? 'rgba(226,232,240,0.92)' : 'rgba(15,23,42,0.9)',
        sub: isDark ? 'rgba(148,163,184,0.9)' : 'rgba(71,85,105,0.9)',
        edge: isDark ? 'rgba(148,163,184,0.55)' : 'rgba(100,116,139,0.55)',
        edgeStrong: isDark ? 'rgba(226,232,240,0.7)' : 'rgba(51,65,85,0.5)',
        node: isDark ? 'rgba(148,163,184,0.9)' : 'rgba(51,65,85,0.92)',
        node2: isDark ? 'rgba(226,232,240,0.95)' : 'rgba(15,23,42,0.95)',
        nodeLine: isDark ? 'rgba(100,116,139,0.95)' : 'rgba(100,116,139,0.95)',
        hover: isDark ? 'rgba(226,232,240,0.9)' : 'rgba(15,23,42,0.9)',
      }
    }

    const toCanvas = (pos, w, h, pad) => {
      // Court: x [0..10], y [0..6.5]
      const x = Number(pos?.[0] ?? 0)
      const y = Number(pos?.[1] ?? 0)
      const nx = x / 10
      const ny = y / 6.5
      const cx = pad + nx * (w - pad * 2)
      const cy = pad + (1 - ny) * (h - pad * 2)
      return { x: cx, y: cy }
    }

    const draw = (t) => {
      const rect = wrap.getBoundingClientRect()
      const W = Math.max(1, Math.floor(rect.width))
      const H = Math.max(1, Math.floor(rect.height))
      const ratio = dpr()

      if (canvas.width !== Math.floor(W * ratio) || canvas.height !== Math.floor(H * ratio)) {
        canvas.width = Math.floor(W * ratio)
        canvas.height = Math.floor(H * ratio)
        canvas.style.width = `${W}px`
        canvas.style.height = `${H}px`
      }

      ctx.setTransform(ratio, 0, 0, ratio, 0, 0)

      const isDark = document.documentElement.classList.contains('dark')
      const c = palette(isDark)
      const pad = 18

      // Background
      ctx.clearRect(0, 0, W, H)
      const grd = ctx.createLinearGradient(0, 0, 0, H)
      grd.addColorStop(0, c.bg1)
      grd.addColorStop(1, c.bg2)
      ctx.fillStyle = grd
      ctx.fillRect(0, 0, W, H)

      // Frame
      ctx.strokeStyle = c.border
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.roundRect(0.5, 0.5, W - 1, H - 1, 14)
      ctx.stroke()

      const nodes = dataRef.current.nodes
      const edges = dataRef.current.edges

      const points = new Map()
      for (const n of nodes) {
        if (!Array.isArray(n.pos)) continue
        points.set(String(n.id), toCanvas(n.pos, W, H, pad))
      }

      // Hover detection (nodes first).
      const p = pointerRef.current
      let hoveredNode = null
      let hoveredNodeDist = Infinity
      if (p.inside) {
        for (const n of nodes) {
          const id = String(n.id)
          const pt = points.get(id)
          if (!pt) continue
          const dx = p.x - pt.x
          const dy = p.y - pt.y
          const d2 = dx * dx + dy * dy
          const r = n.kind === 'line' ? 10 : 12
          if (d2 < r * r && d2 < hoveredNodeDist) {
            hoveredNodeDist = d2
            hoveredNode = n
          }
        }
      }

      setHovered((prev) => {
        const prevId = prev ? String(prev.id) : null
        const nextId = hoveredNode ? String(hoveredNode.id) : null
        if (prevId === nextId) return prev
        return hoveredNode
      })

      // Edges
      for (const e of edges) {
        const s = points.get(String(e.source))
        const d = points.get(String(e.target))
        if (!s || !d) continue
        const w = clamp01(Number(e.weight ?? 0.4))
        ctx.strokeStyle = w > 0.7 ? c.edgeStrong : c.edge
        ctx.lineWidth = 1 + w * 1.6
        ctx.beginPath()
        ctx.moveTo(s.x, s.y)
        ctx.lineTo(d.x, d.y)
        ctx.stroke()

        // Edge label at midpoint (small + faint).
        const label = edgeLabel(e)
        if (label) {
          const mx = (s.x + d.x) / 2
          const my = (s.y + d.y) / 2
          const wobble = Math.sin((t / 1000) * 1.2 + mx * 0.01) * 0.6
          ctx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
          ctx.fillStyle = isDark ? 'rgba(148,163,184,0.7)' : 'rgba(71,85,105,0.65)'
          ctx.fillText(label, mx + 6, my - 6 + wobble)
        }
      }

      // Nodes
      for (const n of nodes) {
        const id = String(n.id)
        const pt = points.get(id)
        if (!pt) continue
        const isLine = n.kind === 'line'
        const isHover = hoveredNode && String(hoveredNode.id) === id

        const baseR = isLine ? 7 : 9
        const pulse = 0.6 + 0.6 * Math.sin((t / 1000) * 1.6 + Number(n.id) * 0.9)
        const r = baseR + (isHover ? 3 : 0) + (isLine ? 0 : 0.8 * pulse)

        if (isLine) {
          ctx.fillStyle = c.nodeLine
          ctx.strokeStyle = isHover ? c.hover : c.border
          ctx.lineWidth = isHover ? 2 : 1
          ctx.beginPath()
          ctx.roundRect(pt.x - r, pt.y - r, r * 2, r * 2, 4)
          ctx.fill()
          ctx.stroke()
        } else {
          const fill = n.role === 'RAIDER' ? c.node2 : c.node
          ctx.fillStyle = fill
          ctx.strokeStyle = isHover ? c.hover : c.border
          ctx.lineWidth = isHover ? 2 : 1
          ctx.beginPath()
          ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2)
          ctx.fill()
          ctx.stroke()
        }

        // Node label
        const text = nodeLabel(n)
        ctx.font = '700 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
        ctx.fillStyle = c.text
        ctx.fillText(text, pt.x + r + 6, pt.y - 6)
      }

      // Meta badges (top-left)
      const meta = dataRef.current.meta
      if (meta) {
        ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
        ctx.fillStyle = c.sub
        ctx.fillText(`contact ${Number(meta.best_contact_score ?? 0).toFixed(2)}`, pad, pad - 4)
        ctx.fillText(
          `contain ${Number(meta.best_containment_score ?? 0).toFixed(2)}`,
          pad + 148,
          pad - 4,
        )
        ctx.fillText(`def ${meta.visible_defenders ?? 0}`, pad + 310, pad - 4)
      }

      raf = window.requestAnimationFrame(draw)
    }

    raf = window.requestAnimationFrame(draw)

    return () => window.cancelAnimationFrame(raf)
  }, [height])

  useEffect(() => {
    const wrap = wrapRef.current
    if (!wrap) return

    const onMove = (e) => {
      const rect = wrap.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      pointerRef.current = { x, y, inside: true }
      setTooltip({ x: e.clientX, y: e.clientY })
    }
    const onLeave = () => {
      pointerRef.current = { x: -9999, y: -9999, inside: false }
    }

    wrap.addEventListener('pointermove', onMove, { passive: true })
    wrap.addEventListener('pointerleave', onLeave, { passive: true })
    return () => {
      wrap.removeEventListener('pointermove', onMove)
      wrap.removeEventListener('pointerleave', onLeave)
    }
  }, [])

  return (
    <div
      ref={wrapRef}
      className="relative overflow-hidden rounded-xl border border-stone-200 bg-white/60 dark:border-stone-800 dark:bg-stone-950/30"
      style={{ height }}
    >
      <canvas ref={canvasRef} className="absolute inset-0" />

      {!data.nodes.length ? (
        <div className="absolute inset-0 grid place-items-center px-6 text-center text-xs text-stone-600 dark:text-stone-400">
          Waiting for graph nodes/edges.
        </div>
      ) : null}

      {hovered ? (
        <div
          className="pointer-events-none fixed z-50 w-[260px] -translate-x-1/2 rounded-xl border border-stone-200 bg-white/90 p-3 text-[11px] text-stone-800 shadow-lg backdrop-blur dark:border-stone-700 dark:bg-stone-950/85 dark:text-stone-100"
          style={{ left: tooltip.x, top: tooltip.y - 14 }}
        >
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0 truncate font-semibold">
              {hovered.kind === 'line'
                ? String(hovered.label ?? hovered.id)
                : `${hovered.role === 'RAIDER' ? 'Raider' : 'Defender'} ${hovered.id}`}
            </div>
            <div className="shrink-0 text-stone-500 dark:text-stone-400">
              {hovered.kind === 'line' ? 'line' : 'player'}
            </div>
          </div>

          {Array.isArray(hovered.pos) ? (
            <div className="mt-2 grid grid-cols-2 gap-x-2 gap-y-1 text-stone-700 dark:text-stone-200">
              <div className="text-stone-500 dark:text-stone-400">court x</div>
              <div className="text-right tabular-nums">
                {Number(hovered.pos[0]).toFixed(2)}
              </div>
              <div className="text-stone-500 dark:text-stone-400">court y</div>
              <div className="text-right tabular-nums">
                {Number(hovered.pos[1]).toFixed(2)}
              </div>
              {typeof hovered.track_confidence === 'number' ? (
                <>
                  <div className="text-stone-500 dark:text-stone-400">track</div>
                  <div className="text-right tabular-nums">
                    {Number(hovered.track_confidence).toFixed(2)}
                  </div>
                </>
              ) : null}
              {typeof hovered.visibility_confidence === 'number' ? (
                <>
                  <div className="text-stone-500 dark:text-stone-400">visible</div>
                  <div className="text-right tabular-nums">
                    {Number(hovered.visibility_confidence).toFixed(2)}
                  </div>
                </>
              ) : null}
            </div>
          ) : (
            <div className="mt-2 text-stone-600 dark:text-stone-400">
              No spatial position.
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}
