import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import Graph2D from './Graph2D.jsx'
import CourtMat2D from './CourtMat2D.jsx'
import RaidReplay3D from './RaidReplay3D.jsx'

const LS_BACKEND_HTTP = 'kabaddi.backendHttp'
const LS_THEME = 'kabaddi.theme'
const LS_LAST_OUTPUTS = 'kabaddi.lastOutputs'

function normalizeBaseUrl(input) {
  const trimmed = String(input ?? '').trim()
  if (!trimmed) return ''
  // Accept "localhost:8000" and normalize it.
  const withScheme =
    /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`
  return withScheme.replace(/\/+$/, '')
}

function formatEventType(eventType) {
  if (!eventType) return 'UNKNOWN'
  return String(eventType).replace(/^CONFIRMED_/, '').replaceAll('_', ' ')
}

function Badge({ tone = 'slate', children }) {
  const cls = {
    slate:
      'bg-slate-100 text-slate-700 ring-slate-200 dark:bg-slate-800/60 dark:text-slate-100 dark:ring-slate-700',
    amber:
      'bg-amber-50 text-amber-800 ring-amber-200 dark:bg-amber-900/25 dark:text-amber-100 dark:ring-amber-900/40',
    emerald:
      'bg-emerald-50 text-emerald-800 ring-emerald-200 dark:bg-emerald-900/25 dark:text-emerald-100 dark:ring-emerald-900/40',
    rose:
      'bg-rose-50 text-rose-800 ring-rose-200 dark:bg-rose-900/25 dark:text-rose-100 dark:ring-rose-900/40',
  }[tone]
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset ${cls}`}
    >
      {children}
    </span>
  )
}

function Panel({ title, right, children, density = 'normal' }) {
  const headerPad =
    density === 'compact' ? 'px-3 py-2' : 'px-4 py-3'
  const bodyPad = density === 'compact' ? 'p-3' : 'p-4'
  return (
    <section className="rounded-2xl border border-stone-200 bg-white shadow-sm dark:border-stone-800 dark:bg-stone-900/60 dark:shadow-none">
      <header className={`flex items-start justify-between gap-3 border-b border-stone-100 ${headerPad} dark:border-stone-800/70`}>
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold text-stone-900 dark:text-stone-50">
            {title}
          </h2>
        </div>
        {right ? <div className="shrink-0">{right}</div> : null}
      </header>
      <div className={bodyPad}>{children}</div>
    </section>
  )
}

function StreamView({ src, alt, height = 360 }) {
  if (!src) {
    return (
      <div
        className="grid place-items-center rounded-xl border border-dashed border-stone-200 bg-stone-50 text-xs text-stone-500 dark:border-stone-800 dark:bg-stone-950/30 dark:text-stone-400"
        style={{ height }}
      >
        Configure backend URL to start streaming.
      </div>
    )
  }

  return (
    <div className="stream-frame" style={{ height }}>
      <img
        src={src}
        alt={alt}
        className="h-full w-full rounded-xl border border-stone-200 bg-stone-50 object-contain dark:border-stone-800 dark:bg-stone-950/40"
        loading="eager"
        decoding="async"
        referrerPolicy="no-referrer"
      />
    </div>
  )
}

function VideoPreview({ mp4Src, mjpegSrc, title = 'Video', allowFullscreen = false }) {
  const [fallback, setFallback] = useState(false)
  const mjpegRef = useRef(null)

  useEffect(() => {
    // If the source changes, retry MP4 first.
    setFallback(false)
  }, [mp4Src])

  if (!mp4Src) {
    return (
      <div className="grid h-[140px] place-items-center rounded-xl border border-dashed border-stone-200 bg-stone-50 text-xs text-stone-500 dark:border-stone-800 dark:bg-stone-950/30 dark:text-stone-400">
        No {title.toLowerCase()} yet.
      </div>
    )
  }

  if (fallback) {
    const onFullscreen = async () => {
      const el = mjpegRef.current
      if (!allowFullscreen) return
      try {
        if (el && el.requestFullscreen) {
          await el.requestFullscreen()
          return
        }
      } catch {
        // ignore
      }
      // Fallback: open the MJPEG stream in a new tab/window.
      if (mjpegSrc) window.open(mjpegSrc, '_blank', 'noopener,noreferrer')
    }

    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div className="text-[11px] text-stone-600 dark:text-stone-400">
            MP4 playback failed in this browser. Showing MJPEG fallback.
          </div>
          {allowFullscreen ? (
            <button
              className="rounded-full border border-stone-200 bg-white px-2.5 py-1 text-[11px] font-medium shadow-sm hover:bg-stone-50 dark:border-stone-800 dark:bg-stone-900/60 dark:hover:bg-stone-900"
              onClick={onFullscreen}
              type="button"
            >
              Fullscreen
            </button>
          ) : null}
        </div>
        <div ref={mjpegRef}>
          <StreamView src={mjpegSrc} alt={`${title} MJPEG`} />
        </div>
      </div>
    )
  }

  return (
    <video
      className="w-full rounded-xl border border-stone-200 bg-stone-50 dark:border-stone-800 dark:bg-stone-950/40"
      controls
      src={mp4Src}
      onError={() => setFallback(true)}
    />
  )
}

function OverlayMjpegPlayer({
  baseSrc,
  overlaySrc,
  overlayContent,
  height = 260,
  overlayOpacity = 1,
  title = 'Overlay',
  allowFullscreen = false,
  replay3d = null,
  theme = 'dark',
}) {
  const wrapRef = useRef(null)
  const baseImgRef = useRef(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [show3D, setShow3D] = useState(false)

  useEffect(() => {
    const onChange = () => {
      try {
        setIsFullscreen(document.fullscreenElement === wrapRef.current)
      } catch {
        setIsFullscreen(false)
      }
    }
    try {
      document.addEventListener('fullscreenchange', onChange)
    } catch {
      // ignore
    }
    onChange()
    return () => {
      try {
        document.removeEventListener('fullscreenchange', onChange)
      } catch {
        // ignore
      }
    }
  }, [])

  useEffect(() => {
    // 3D view is fullscreen-only.
    if (!isFullscreen) setShow3D(false)
  }, [isFullscreen])

  const onFullscreen = async () => {
    if (!allowFullscreen) return
    const el = wrapRef.current
    try {
      if (el?.requestFullscreen) {
        await el.requestFullscreen()
        return
      }
      if (el?.webkitRequestFullscreen) {
        el.webkitRequestFullscreen()
        return
      }
    } catch {
      // ignore
    }
    if (baseSrc) window.open(baseSrc, '_blank', 'noopener,noreferrer')
  }

  const onCloseFullscreen = async () => {
    const doc = wrapRef.current?.ownerDocument || document
    try {
      if (doc.fullscreenElement && doc.exitFullscreen) {
        await doc.exitFullscreen()
        return
      }
      if (doc.webkitFullscreenElement && doc.webkitExitFullscreen) {
        doc.webkitExitFullscreen()
        return
      }
      if (wrapRef.current?.classList?.contains('fallback-fullscreen')) {
        wrapRef.current.classList.remove('fallback-fullscreen')
      }
    } catch {
      // ignore
    }
  }

  const can3D =
    !!replay3d &&
    Array.isArray(replay3d.matWindow) &&
    replay3d.matWindow.length > 0

  const canPoseOverlay =
    !!replay3d &&
    Array.isArray(replay3d.poseWindow) &&
    replay3d.poseWindow.length > 0

  if (!baseSrc) {
    return (
      <div className="grid place-items-center rounded-xl border border-dashed border-stone-200 bg-stone-50 text-xs text-stone-500 dark:border-stone-800 dark:bg-stone-950/30 dark:text-stone-400" style={{ height }}>
        No {title.toLowerCase()}.
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] text-stone-600 dark:text-stone-400">
          Video with mat overlay.
        </div>
        {allowFullscreen ? (
          <button
            className="rounded-full border border-stone-200 bg-white px-2.5 py-1 text-[11px] font-medium shadow-sm hover:bg-stone-50 dark:border-stone-800 dark:bg-stone-900/60 dark:hover:bg-stone-900"
            onClick={onFullscreen}
            type="button"
          >
            Fullscreen
          </button>
        ) : null}
      </div>

      <div
        ref={wrapRef}
        className="relative overflow-hidden rounded-xl border border-stone-200 bg-stone-50 dark:border-stone-800 dark:bg-stone-950/40"
        style={{ height }}
      >
        {allowFullscreen && isFullscreen ? (
          <div className="absolute right-2 top-2 z-[120] flex items-center gap-2">
            {can3D ? (
              <button
                type="button"
                onClick={() => setShow3D((v) => !v)}
                className={`grid h-9 place-items-center rounded-full border px-3 text-[12px] font-semibold shadow-sm backdrop-blur ${
                  show3D
                    ? 'border-white/20 bg-white/20 text-white hover:bg-white/24'
                    : 'border-white/15 bg-black/45 text-white hover:bg-black/60'
                }`}
                aria-pressed={show3D}
                title="Toggle 3D replay"
              >
                3D
              </button>
            ) : null}
            <button
              type="button"
              onClick={onCloseFullscreen}
              className="grid h-9 w-9 place-items-center rounded-full border border-white/15 bg-black/65 text-sm font-semibold text-white shadow-lg backdrop-blur hover:bg-black/80"
              aria-label="Close fullscreen"
              title="Close"
            >
              X
            </button>
          </div>
        ) : null}
        <div
          className={`absolute inset-0 transition-opacity duration-300 ${
            show3D ? 'opacity-0' : 'opacity-100'
          }`}
          style={{ pointerEvents: show3D ? 'none' : 'auto' }}
        >
          <img
            ref={baseImgRef}
            src={baseSrc}
            alt="Video stream"
            className="absolute inset-0 h-full w-full object-contain"
            loading="eager"
            decoding="async"
            referrerPolicy="no-referrer"
          />
          {isFullscreen && canPoseOverlay && !show3D ? (
            <PoseOverlayCanvas
              mediaRef={baseImgRef}
              poseWindow={replay3d.poseWindow}
              poseMeta={replay3d.poseMeta}
              theme={theme}
            />
          ) : null}
          {!show3D && (overlaySrc || overlayContent) ? (
            <div
              className="pointer-events-none absolute bottom-2 left-2 overflow-hidden rounded-lg border border-stone-200/60 bg-transparent dark:border-stone-800/70"
              style={{
                width: '26%',
                maxWidth: 200,
                minWidth: 130,
              }}
            >
              <div className="relative">
                {overlayContent ? (
                  <div className="w-full">{overlayContent}</div>
                ) : overlaySrc ? (
                  <img
                    src={overlaySrc}
                    alt="Mat overlay"
                    className="h-auto w-full object-contain"
                    style={{
                      opacity: overlayOpacity,
                    }}
                    loading="eager"
                    decoding="async"
                    referrerPolicy="no-referrer"
                  />
                ) : null}
              </div>
            </div>
          ) : null}
        </div>

        {can3D ? (
          <div
            className={`absolute inset-0 transition-opacity duration-300 ${
              show3D ? 'opacity-100' : 'opacity-0'
            }`}
            style={{ pointerEvents: show3D ? 'auto' : 'none' }}
          >
            {show3D ? (
              <RaidReplay3D
                matWindow={replay3d.matWindow}
                poseWindow={replay3d.poseWindow}
                poseMeta={replay3d.poseMeta}
                event={replay3d.event}
                courtMeta={replay3d.courtMeta}
                videoSrc={baseSrc}
                videoFileSrc={replay3d.videoFileSrc}
                theme={theme}
              />
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  )
}

function PoseOverlayCanvas({ mediaRef, poseWindow, poseMeta, theme = 'dark' }) {
  const canvasRef = useRef(null)
  const isDark = theme === 'dark'

  useEffect(() => {
    const canvas = canvasRef.current
    const media = mediaRef.current
    if (!canvas || !media || !Array.isArray(poseWindow) || poseWindow.length === 0) return

    const keypointNames =
      Array.isArray(poseMeta?.keypoint_names) && poseMeta.keypoint_names.length
        ? poseMeta.keypoint_names
        : [
            'nose',
            'left_eye',
            'right_eye',
            'left_ear',
            'right_ear',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
            'left_hip',
            'right_hip',
            'left_knee',
            'right_knee',
            'left_ankle',
            'right_ankle',
          ]
    const edges =
      Array.isArray(poseMeta?.skeleton_edges) && poseMeta.skeleton_edges.length
        ? poseMeta.skeleton_edges
        : [
            [5, 7],
            [7, 9],
            [6, 8],
            [8, 10],
            [5, 6],
            [5, 11],
            [6, 12],
            [11, 13],
            [13, 15],
            [12, 14],
            [14, 16],
            [11, 12],
          ]

    let raf = 0
    let startedAt = performance.now()

    const draw = () => {
      const ctx = canvas.getContext('2d')
      const hostW = canvas.clientWidth || 1
      const hostH = canvas.clientHeight || 1
      const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1))
      if (canvas.width !== Math.floor(hostW * dpr) || canvas.height !== Math.floor(hostH * dpr)) {
        canvas.width = Math.floor(hostW * dpr)
        canvas.height = Math.floor(hostH * dpr)
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.save()
      ctx.scale(dpr, dpr)

      const mediaW = media.naturalWidth || media.videoWidth || 0
      const mediaH = media.naturalHeight || media.videoHeight || 0
      if (mediaW > 0 && mediaH > 0) {
        const s = Math.min(hostW / mediaW, hostH / mediaH)
        const drawW = mediaW * s
        const drawH = mediaH * s
        const ox = (hostW - drawW) / 2
        const oy = (hostH - drawH) / 2
        const idx = Math.floor(((performance.now() - startedAt) / 1000) * 30) % poseWindow.length
        const frame = poseWindow[idx] || {}
        const players = Array.isArray(frame.players) ? frame.players : []

        players.slice(0, 3).forEach((player, playerIdx) => {
          const color =
            playerIdx === 0
              ? isDark
                ? 'rgba(245,222,179,0.96)'
                : 'rgba(120,92,40,0.96)'
              : playerIdx === 1
                ? isDark
                  ? 'rgba(226,232,240,0.96)'
                  : 'rgba(31,41,55,0.96)'
                : isDark
                  ? 'rgba(148,163,184,0.94)'
                  : 'rgba(71,85,105,0.94)'
          const points = Array.isArray(player?.keypoints) ? player.keypoints : []
          const byName = new Map()
          for (let i = 0; i < points.length; i++) {
            const kp = points[i]
            const x = Number(kp?.x)
            const y = Number(kp?.y)
            if (!Number.isFinite(x) || !Number.isFinite(y)) continue
            byName.set(kp?.name || keypointNames[i] || `kp_${i}`, { x, y })
          }

          ctx.strokeStyle = color
          ctx.fillStyle = color
          ctx.lineWidth = 3
          for (const [aIdx, bIdx] of edges) {
            const a = byName.get(keypointNames[aIdx])
            const b = byName.get(keypointNames[bIdx])
            if (!a || !b) continue
            ctx.beginPath()
            ctx.moveTo(ox + a.x * s, oy + a.y * s)
            ctx.lineTo(ox + b.x * s, oy + b.y * s)
            ctx.stroke()
          }
          for (const kp of byName.values()) {
            ctx.beginPath()
            ctx.arc(ox + kp.x * s, oy + kp.y * s, 3.3, 0, Math.PI * 2)
            ctx.fill()
          }
        })
      }

      ctx.restore()
      raf = requestAnimationFrame(draw)
    }

    raf = requestAnimationFrame(draw)
    return () => {
      if (raf) cancelAnimationFrame(raf)
    }
  }, [mediaRef, poseWindow, poseMeta, isDark])

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 z-[2] h-full w-full" />
}

function _fmtPid(pid) {
  const n = Number(pid)
  if (!Number.isFinite(n)) return String(pid ?? '-')
  return `ID ${String(n).padStart(2, '0')}`
}

function _speed(vel) {
  if (!Array.isArray(vel) || vel.length < 2) return 0
  const vx = Number(vel[0]) || 0
  const vy = Number(vel[1]) || 0
  return Math.sqrt(vx * vx + vy * vy)
}


function GalleryGrid({ players, raiderId }) {
  if (!Array.isArray(players) || players.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-stone-200 bg-stone-50 px-3 py-3 text-xs text-stone-600 dark:border-stone-800 dark:bg-stone-950/30 dark:text-stone-400">
        No live gallery yet.
      </div>
    )
  }

  const items = players.slice().sort((a, b) => (a?.id ?? 0) - (b?.id ?? 0))
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3">
      {items.map((p) => {
        const isRaider = raiderId != null && Number(p?.id) === Number(raiderId)
        const visible = !!p?.visible
        const age = Number(p?.age ?? 0)
        const spd = _speed(p?.velocity)
        const flowPoints = Number(p?.flow_points ?? 0)
        const hsv = Array.isArray(p?.hsv_bins5) ? p.hsv_bins5 : null
        return (
          <div
            key={p?.id ?? Math.random()}
            className="rounded-xl border border-stone-200 bg-white p-3 shadow-sm dark:border-stone-800 dark:bg-stone-950/20 dark:shadow-none"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="truncate text-xs font-semibold text-stone-900 dark:text-stone-50">
                  {isRaider ? 'RAIDER' : _fmtPid(p?.id)}
                </div>
                <div className="mt-0.5 text-[11px] text-stone-600 dark:text-stone-400">
                  {visible ? 'visible' : `lost (age ${age})`}
                </div>
              </div>
              <div className="shrink-0">
                {isRaider ? (
                  <Badge tone="amber">focus</Badge>
                ) : visible ? (
                  <Badge tone="emerald">live</Badge>
                ) : (
                  <Badge tone="slate">hold</Badge>
                )}
              </div>
            </div>

            <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-stone-600 dark:text-stone-400">
              <div className="rounded-lg bg-stone-50 px-2 py-1 dark:bg-stone-900/40">
                <div className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-500">
                  court
                </div>
                <div className="tabular-nums">
                  {(p?.court_pos?.[0] ?? 0).toFixed?.(2) ?? '-'}w
                </div>
                <div className="tabular-nums">
                  {(p?.court_pos?.[1] ?? 0).toFixed?.(2) ?? '-'}d
                </div>
              </div>
              <div className="rounded-lg bg-stone-50 px-2 py-1 dark:bg-stone-900/40">
                <div className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-500">
                  speed
                </div>
                <div className="tabular-nums">{spd.toFixed(2)}</div>
                <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-stone-200 dark:bg-stone-800">
                  <div
                    className="h-full rounded-full bg-stone-700/60 dark:bg-stone-200/60"
                    style={{ width: `${Math.min(100, spd * 14)}%` }}
                  />
                </div>
              </div>
            </div>

            <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-stone-600 dark:text-stone-400">
              <div className="rounded-lg bg-stone-50 px-2 py-1 dark:bg-stone-900/40">
                <div className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-500">
                  flow pts
                </div>
                <div className="tabular-nums">{flowPoints}</div>
              </div>
              <div className="rounded-lg bg-stone-50 px-2 py-1 dark:bg-stone-900/40">
                <div className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-500">
                  hsv bins
                </div>
                {hsv ? (
                  <div className="mt-1 flex items-end gap-0.5">
                    {hsv.slice(0, 5).map((v, idx) => {
                      const n = Number(v) || 0
                      const h = Math.max(2, Math.min(14, Math.round(n * 14)))
                      return (
                        <div
                          key={idx}
                          className="w-2 rounded-sm bg-stone-700/50 dark:bg-stone-200/40"
                          style={{ height: `${h}px` }}
                          title={String(n.toFixed?.(4) ?? n)}
                        />
                      )
                    })}
                  </div>
                ) : (
                  <div className="mt-1 text-[11px] text-stone-500 dark:text-stone-500">
                    -
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function TripletList({ title, items, raiderId, kind }) {
  const list = Array.isArray(items) ? items.slice(-12).reverse() : []
  return (
    <div className="rounded-xl border border-stone-200 bg-white p-3 dark:border-stone-800 dark:bg-stone-950/20">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-semibold text-stone-700 dark:text-stone-200">
          {title}
        </div>
        <Badge tone="slate">{list.length}</Badge>
      </div>
      {list.length ? (
        <div className="mt-2 space-y-1.5">
          {list.map((t, idx) => {
            const s = Number(t?.S)
            const o = kind === 'HLI' ? String(t?.O ?? '-') : Number(t?.O)
            const isRaiderS = raiderId != null && s === Number(raiderId)
            const isRaiderO = raiderId != null && Number(o) === Number(raiderId)
            const labelS = isRaiderS ? 'RAIDER' : _fmtPid(s)
            const labelO = kind === 'HLI' ? String(o) : isRaiderO ? 'RAIDER' : _fmtPid(o)
            return (
              <div
                key={`${idx}-${t?.frame ?? ''}`}
                className="rounded-lg bg-stone-50 px-2 py-1 text-[11px] text-stone-700 dark:bg-stone-900/40 dark:text-stone-200"
                title={JSON.stringify(t)}
              >
                <span className="tabular-nums text-stone-500 dark:text-stone-400">
                  f{t?.frame ?? '-'}
                </span>{' '}
                <span className="font-mono">
                  {'<'}
                  {labelS}, {t?.I ?? '-'}, {labelO}
                  {'>'}
                </span>
                <span className="ml-2 tabular-nums text-stone-500 dark:text-stone-400">
                  d {Number(t?.dist ?? 0).toFixed(2)}
                  {kind === 'HHI' ? ` | rv ${Number(t?.rel_vel ?? 0).toFixed(2)}` : ''}
                  {kind === 'HLI' ? ` | ${t?.active ? 'touch' : 'near'}` : ''}
                </span>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="mt-2 text-xs text-stone-600 dark:text-stone-400">
          No {kind} triplets yet.
        </div>
      )}
    </div>
  )
}

function ScoreStrip({ aName = 'Team A', bName = 'Team B', a = 0, b = 0 }) {
  return (
    <div className="rounded-2xl border border-stone-200 bg-white/70 px-4 py-3 shadow-sm backdrop-blur dark:border-stone-800 dark:bg-stone-950/30 dark:shadow-none">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-stone-600 dark:text-stone-400">
            Score
          </div>
          <div className="mt-0.5 truncate text-sm font-semibold text-stone-900 dark:text-stone-50">
            {aName} <span className="text-stone-300">vs</span> {bName}
          </div>
        </div>

        <div className="flex items-baseline gap-2">
          <div className="tabular-nums text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-50">
            {a}
          </div>
          <div className="text-stone-400">:</div>
          <div className="tabular-nums text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-50">
            {b}
          </div>
        </div>
      </div>
    </div>
  )
}

function App() {
  const initialTheme = useMemo(() => {
    const saved = localStorage.getItem(LS_THEME)
    return saved === 'dark' ? 'dark' : 'light'
  }, [])
  const [theme, setTheme] = useState(initialTheme)

  useEffect(() => {
    const isDark = theme === 'dark'
    document.documentElement.classList.toggle('dark', isDark)
    document.documentElement.style.colorScheme = isDark ? 'dark' : 'light'
    localStorage.setItem(LS_THEME, theme)
  }, [theme])

  const initialBackend = useMemo(() => {
    const fromStorage = localStorage.getItem(LS_BACKEND_HTTP)
    const fromEnv = import.meta.env.VITE_BACKEND_HTTP
    return normalizeBaseUrl(fromStorage || fromEnv || 'http://localhost:8000')
  }, [])

  const [backendHttp, setBackendHttp] = useState(initialBackend)
  const [backendHttpDraft, setBackendHttpDraft] = useState(initialBackend)
  const baseUrl = useMemo(() => normalizeBaseUrl(backendHttp), [backendHttp])

  const [conn, setConn] = useState({ mode: 'idle', lastAt: null, error: null })
  const [live, setLive] = useState(null)
  const [health, setHealth] = useState({
    ok: false,
    live: false,
    archive: false,
    run_id: null,
  })

  const [offlineOutputs, setOfflineOutputs] = useState(() => {
    try {
      const raw = localStorage.getItem(LS_LAST_OUTPUTS)
      return raw ? JSON.parse(raw) : null
    } catch {
      return null
    }
  })
  const lastRunIdRef = useRef(null)
  const wasOnlineRef = useRef(false)

  const showDashboard = health.ok || !!offlineOutputs
  const showLive = health.ok && health.live

  const eventMapRef = useRef(new Map())
  const [events, setEvents] = useState([])
  const [selectedEventId, setSelectedEventId] = useState(null)
  const selectedEvent = useMemo(() => {
    if (!selectedEventId) return null
    return events.find((e) => e.id === selectedEventId) || null
  }, [events, selectedEventId])
  const [selectedDetails, setSelectedDetails] = useState({
    status: 'idle',
    error: null,
    data: null,
  })
  const selectedMatSnapshot = useMemo(() => {
    const ev = selectedDetails?.data?.archive_event
    const window = ev?.mat_window
    if (!Array.isArray(window) || window.length === 0) return null

    const targetFrame = selectedEvent?.frame ?? ev?.frame
    const exact = window.find((s) => Number(s?.frame) === Number(targetFrame))
    return exact || window[Math.floor(window.length / 2)] || null
  }, [selectedDetails, selectedEvent])

  useEffect(() => {
    if (!showLive) return
    // When the live pipeline attaches, discard any previously loaded archive state
    // so the UI reflects the current run only.
    eventMapRef.current = new Map()
    setEvents([])
    setSelectedEventId(null)
    setSelectedDetails({ status: 'idle', error: null, data: null })
  }, [showLive])

  const [latestVideos, setLatestVideos] = useState({
    processed: null,
    report: null,
  })

  const endpoints = useMemo(() => {
    if (!baseUrl) return null
    return {
      health: `${baseUrl}/api/health`,
      stateStream: `${baseUrl}/api/state/stream`,
      state: `${baseUrl}/api/state`,
      inputStream: `${baseUrl}/api/input/stream`,
      processingStream: `${baseUrl}/api/vis/stream`,
      matStream: `${baseUrl}/api/mat/stream`,
      latestVideos: `${baseUrl}/api/videos/latest`,
      videoFile: (name) => `${baseUrl}/api/videos/file/${encodeURIComponent(name)}`,
      videoMjpeg: (name) =>
        `${baseUrl}/api/videos/mjpeg/${encodeURIComponent(name)}`,
      videoMjpegVis: (name) =>
        `${baseUrl}/api/videos/mjpeg/vis/${encodeURIComponent(name)}`,
      videoMjpegMat: (name) =>
        `${baseUrl}/api/videos/mjpeg/mat/${encodeURIComponent(name)}`,
      eventDetails: (eventId) =>
        `${baseUrl}/api/events/details/${encodeURIComponent(eventId)}`,
      eventClipMjpeg: (clipId) =>
        `${baseUrl}/api/events/clip_mjpeg/${encodeURIComponent(clipId)}`,
      eventClipMjpegVis: (clipId) =>
        `${baseUrl}/api/events/clip_mjpeg/vis/${encodeURIComponent(clipId)}`,
      eventClipMjpegMat: (clipId) =>
        `${baseUrl}/api/events/clip_mjpeg/mat/${encodeURIComponent(clipId)}`,
      archiveEvents: `${baseUrl}/api/archive/events`,
    }
  }, [baseUrl])

  useEffect(() => {
    if (!endpoints) return
    localStorage.setItem(LS_BACKEND_HTTP, baseUrl)
  }, [endpoints, baseUrl])

  useEffect(() => {
    if (!endpoints) {
      setHealth({ ok: false, live: false, archive: false, run_id: null })
      return
    }
    let timer = null
    const tick = async () => {
      try {
        const res = await fetch(endpoints.health, { cache: 'no-store' })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const payload = await res.json()
        setHealth({
          ok: !!payload?.ok,
          live: !!payload?.live,
          archive: !!payload?.archive,
          run_id: payload?.run_id ?? null,
        })
      } catch {
        setHealth({ ok: false, live: false, archive: false, run_id: null })
      }
    }
    tick()
    timer = setInterval(tick, 1000)
    return () => {
      if (timer) clearInterval(timer)
    }
  }, [endpoints, showLive])

  // Prefer SSE for state updates; fall back to polling if SSE fails (e.g. proxy/corp networks).
  useEffect(() => {
    if (!endpoints || !showLive) {
      setConn({ mode: 'idle', lastAt: null, error: null })
      setLive(null)
      return
    }

    let closed = false
    let pollTimer = null
    let es = null

    const setStateFromPayload = (payload) => {
      if (closed) return
      setLive(payload)
      setConn({ mode: 'live', lastAt: Date.now(), error: null })

      const incoming = Array.isArray(payload?.events) ? payload.events : []
      if (!incoming.length) return

      const map = eventMapRef.current
      for (const ev of incoming) {
        if (!ev || !ev.id) continue
        const prev = map.get(ev.id)
        map.set(ev.id, { ...prev, ...ev })
      }
      const all = Array.from(map.values())
      all.sort((a, b) => (a.frame ?? 0) - (b.frame ?? 0))
      const tail = all.slice(-200)
      eventMapRef.current = new Map(tail.map((e) => [e.id, e]))
      setEvents(tail)
    }

    const startPolling = () => {
      if (closed) return
      if (pollTimer) return
      setConn((c) => ({ ...c, mode: 'polling' }))
      pollTimer = setInterval(async () => {
        try {
          const res = await fetch(endpoints.state, { cache: 'no-store' })
          if (res.status === 204) return
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          const payload = await res.json()
          setStateFromPayload(payload)
        } catch (err) {
          setConn({ mode: 'offline', lastAt: null, error: String(err?.message || err) })
        }
      }, 250)
    }

    try {
      setConn({ mode: 'connecting', lastAt: null, error: null })
      es = new EventSource(endpoints.stateStream)
      es.onmessage = (msg) => {
        try {
          const payload = JSON.parse(msg.data)
          setStateFromPayload(payload)
        } catch {
          // Ignore malformed messages.
        }
      }
      es.onerror = () => {
        try {
          es?.close()
        } catch {
          // ignore
        }
        es = null
        startPolling()
      }
    } catch (err) {
      setConn({ mode: 'offline', lastAt: null, error: String(err?.message || err) })
      startPolling()
    }

    return () => {
      closed = true
      try {
        es?.close()
      } catch {
        // ignore
      }
      if (pollTimer) clearInterval(pollTimer)
    }
  }, [endpoints, showLive])

  useEffect(() => {
    if (!selectedEventId || !endpoints) {
      setSelectedDetails({ status: 'idle', error: null, data: null })
      return
    }
    let alive = true
    setSelectedDetails({ status: 'loading', error: null, data: null })
    fetch(endpoints.eventDetails(selectedEventId), { cache: 'no-store' })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((payload) => {
        if (!alive) return
        setSelectedDetails({ status: 'ready', error: null, data: payload })
      })
      .catch((err) => {
        if (!alive) return
        setSelectedDetails({
          status: 'error',
          error: String(err?.message || err),
          data: null,
        })
      })
    return () => {
      alive = false
    }
  }, [selectedEventId, endpoints])

  useEffect(() => {
    if (!selectedEventId) return
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setSelectedEventId(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [selectedEventId])

  useEffect(() => {
    if (!endpoints) return
    let timer = null
    const tick = async () => {
      try {
        const res = await fetch(endpoints.latestVideos, { cache: 'no-store' })
        if (!res.ok) return
        const payload = await res.json()
        setLatestVideos(payload)

        if (payload?.processed || payload?.report) {
          const snapshot = {
            processed: payload.processed ?? null,
            report: payload.report ?? null,
            at: Date.now(),
          }
          setOfflineOutputs(snapshot)
          try {
            localStorage.setItem(LS_LAST_OUTPUTS, JSON.stringify(snapshot))
          } catch {
            // ignore
          }
        }
      } catch {
        // optional
      }
    }
    tick()
    timer = setInterval(tick, 2000)
    return () => {
      if (timer) clearInterval(timer)
    }
  }, [endpoints])

  useEffect(() => {
    // If the API is up but the live pipeline isn't attached, show archived events.
    if (!endpoints || !health.ok || showLive) return
    let timer = null
    const tick = async () => {
      try {
        const res = await fetch(endpoints.archiveEvents, { cache: 'no-store' })
        if (!res.ok) return
        const payload = await res.json()
        const incoming = Array.isArray(payload?.events) ? payload.events : []
        if (!incoming.length) {
          // Keep whatever we already have, but don't throw.
          return
        }

        const map = eventMapRef.current
        for (const e of incoming) {
          const id = `${e?.type}|${e?.frame}|${e?.subject}|${e?.object}`
          if (!e) continue
          const prev = map.get(id)
          map.set(id, {
            ...prev,
            id,
            frame: e?.frame,
            type: e?.type,
            subject: e?.subject,
            object: e?.object,
            conf: e?.confidence ?? e?.conf,
            factor_conf: e?.factor_confidence ?? e?.factor_conf,
            requires_visual_confirmation: e?.requires_visual_confirmation ?? false,
            classifier_label: e?.classifier_label,
            classifier_valid_prob: e?.classifier_valid_prob,
            guaranteed_by_classifier: e?.guaranteed_by_classifier,
          })
        }
        const all = Array.from(map.values())
        all.sort((a, b) => (a.frame ?? 0) - (b.frame ?? 0))
        const tail = all.slice(-200)
        eventMapRef.current = new Map(tail.map((ev) => [ev.id, ev]))
        setEvents(tail)
      } catch {
        // ignore
      }
    }
    tick()
    timer = setInterval(tick, 2000)
    return () => {
      if (timer) clearInterval(timer)
    }
  }, [endpoints, health.ok, showLive])

  const validatedTouchScore = useMemo(() => {
    let score = 0
    for (const ev of events) {
      if (ev?.type === 'CONFIRMED_RAIDER_DEFENDER_CONTACT' && ev?.classifier_label === 'valid') {
        score += 1
      }
    }
    return score
  }, [events])

  const connBadge = useMemo(() => {
    if (health.ok && !showLive) return <Badge tone="slate">ARCHIVE</Badge>
    if (conn.mode === 'live') return <Badge tone="emerald">LIVE</Badge>
    if (conn.mode === 'polling') return <Badge tone="amber">POLLING</Badge>
    if (conn.mode === 'connecting') return <Badge tone="slate">CONNECTING</Badge>
    if (conn.mode === 'offline') return <Badge tone="rose">OFFLINE</Badge>
    return <Badge tone="slate">IDLE</Badge>
  }, [conn.mode, health.ok, showLive])

  const scoreboard = useMemo(() => {
    const attacker = live?.score_attacker ?? 0
    const defender = live?.score_defender ?? 0
    return { attacker, defender }
  }, [live])

  const displayedScoreboard = useMemo(() => {
    // For now, treat each classifier-validated touch as +1 to Team A.
    // If the backend connection drops, we reset scores (avoid stale UI).
    if (!health.ok) return { attacker: 0, defender: 0 }
    return {
      attacker: (scoreboard.attacker ?? 0) + (validatedTouchScore ?? 0),
      defender: scoreboard.defender ?? 0,
    }
  }, [health.ok, scoreboard.attacker, scoreboard.defender, validatedTouchScore])

  useEffect(() => {
    if (!showDashboard) setSelectedEventId(null)
  }, [showDashboard])

  useEffect(() => {
    // If we transition from offline -> online, don't keep showing cached outputs.
    if (health.ok && !wasOnlineRef.current) {
      wasOnlineRef.current = true
      setLatestVideos({ processed: null, report: null })
      eventMapRef.current = new Map()
      setEvents([])
      setSelectedEventId(null)
      setSelectedDetails({ status: 'idle', error: null, data: null })
      lastRunIdRef.current = health.run_id ?? null
      return
    }
    if (!health.ok) {
      wasOnlineRef.current = false
      // Connection is down: clear live/derived state so we don't display stale scores/events.
      setLive(null)
      eventMapRef.current = new Map()
      setEvents([])
      setSelectedEventId(null)
      setSelectedDetails({ status: 'idle', error: null, data: null })
    }
  }, [health.ok, health.run_id])

  useEffect(() => {
    // When a new live run starts, clear previous run outputs/archives immediately.
    if (!health.ok || !showLive) return
    const nextRun = health.run_id ?? null
    const prevRun = lastRunIdRef.current
    if (nextRun && prevRun && nextRun !== prevRun) {
      setLatestVideos({ processed: null, report: null })
      eventMapRef.current = new Map()
      setEvents([])
      setSelectedEventId(null)
      setSelectedDetails({ status: 'idle', error: null, data: null })
    }
    lastRunIdRef.current = nextRun || prevRun
  }, [health.ok, showLive, health.run_id])

  return (
    <div className="min-h-screen bg-stone-50 text-stone-900 dark:bg-stone-950 dark:text-stone-50">
      <div className="app-bg" />

      <header className="w-full px-3 pb-4 pt-10 sm:px-4 lg:px-5">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="truncate text-2xl font-bold tracking-tight sm:text-3xl">
                Kabaddi Live Dashboard
              </h1>
              {connBadge}
            </div>
            <p className="mt-1 text-xs text-stone-600 dark:text-stone-400">
              Input, tracking, mat, interaction graph, events, and classifier
              validation.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              className="rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs shadow-sm hover:bg-stone-50 dark:border-stone-800 dark:bg-stone-900/60 dark:hover:bg-stone-900"
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
            >
              {theme === 'dark' ? 'Dark' : 'Light'}
            </button>

            {showDashboard ? (
              <>
                <div className="rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs dark:border-stone-800 dark:bg-stone-900/60">
                  <span className="text-stone-500 dark:text-stone-400">
                    Validated touches:
                  </span>{' '}
                  <span className="font-semibold tabular-nums">{validatedTouchScore}</span>
                  
                </div>
                <div className="rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs dark:border-stone-800 dark:bg-stone-900/60">
                  <span className="text-stone-500 dark:text-stone-400">
                    Score A/D:
                  </span>{' '}
                  <span className="font-semibold tabular-nums">
                    {scoreboard.attacker}/{scoreboard.defender}
                  </span>
                </div>
                <div className="rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs dark:border-stone-800 dark:bg-stone-900/60">
                  <span className="text-stone-500 dark:text-stone-400">Frame:</span>{' '}
                  <span className="font-semibold tabular-nums">
                    {live?.frame_idx ?? '-'}
                  </span>
                </div>
              </>
            ) : null}
          </div>
        </div>

        <div className="mt-4 flex flex-col gap-2 sm:flex-row sm:items-center">
          <div className="flex-1">
            <label className="block text-[11px] font-medium text-stone-600 dark:text-stone-400">
              Backend base URL (FastAPI)
            </label>
            <div className="mt-1 flex gap-2">
              <input
                value={backendHttpDraft}
                onChange={(e) => setBackendHttpDraft(e.target.value)}
                placeholder="http://localhost:8000"
                className="w-full rounded-xl border border-stone-200 bg-white px-3 py-2 text-sm shadow-sm outline-none focus:border-stone-400 dark:border-stone-800 dark:bg-stone-950/40 dark:focus:border-stone-600"
              />
              <button
                className="rounded-xl border border-stone-200 bg-white px-3 py-2 text-sm font-medium shadow-sm hover:bg-stone-50 dark:border-stone-800 dark:bg-stone-900/60 dark:hover:bg-stone-900"
                onClick={() => setBackendHttp(normalizeBaseUrl(backendHttpDraft))}
              >
                Connect
              </button>
            </div>
          </div>
          {conn.error ? (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/40 dark:text-rose-100">
              {conn.error}
            </div>
          ) : null}
        </div>
      </header>

      {!health.ok && offlineOutputs ? (
        <main className="w-full px-3 pb-10 sm:px-4 lg:px-5">
          <Panel title="Last Known Outputs" right={<Badge tone="slate">OFFLINE</Badge>} density="compact">
            <div className="text-xs text-stone-600 dark:text-stone-400">
              Backend API is unreachable. Showing the last known output filenames from this browser.
            </div>
            <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-stone-200 bg-white px-3 py-3 text-sm dark:border-stone-800 dark:bg-stone-950/20">
                <div className="text-[11px] font-medium text-stone-600 dark:text-stone-400">
                  Processed (tracked)
                </div>
                <div className="mt-1 break-all text-xs text-stone-900 dark:text-stone-100">
                  {offlineOutputs.processed || 'None'}
                </div>
              </div>
              <div className="rounded-xl border border-stone-200 bg-white px-3 py-3 text-sm dark:border-stone-800 dark:bg-stone-950/20">
                <div className="text-[11px] font-medium text-stone-600 dark:text-stone-400">
                  Confirmed report
                </div>
                <div className="mt-1 break-all text-xs text-stone-900 dark:text-stone-100">
                  {offlineOutputs.report || 'None'}
                </div>
              </div>
            </div>
          </Panel>
        </main>
      ) : showDashboard ? (
        <main className="w-full px-3 pb-10 sm:px-4 lg:px-5">
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
          <div className="lg:col-span-8">
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <Panel
                title="Input Video (Backend)"
                right={<Badge tone="slate">MJPEG</Badge>}
              >
                <StreamView
                  src={endpoints?.inputStream}
                  alt="Input stream"
                  height={420}
                />
              </Panel>

              <Panel
                title="Processed / Tracked Video"
                right={<Badge tone="slate">MJPEG</Badge>}
              >
                <StreamView
                  src={endpoints?.processingStream}
                  alt="Processed stream"
                  height={420}
                />
              </Panel>
            </div>

            <div className="mt-3">
              <ScoreStrip
                aName="Team A"
                bName="Team B"
                a={displayedScoreboard.attacker}
                b={displayedScoreboard.defender}
              />
            </div>
          </div>

          <div className="lg:col-span-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="col-span-1">
                <Panel title="2D Mat" density="compact">
                  <div className="grid w-full place-items-center overflow-hidden rounded-xl border border-stone-200 bg-stone-950/10 dark:border-stone-800 dark:bg-stone-950/40">
                    <CourtMat2D
                      players={live?.gallery}
                      raiderId={live?.raider_id}
                      height={200}
                      theme={theme}
                    />
                  </div>
                </Panel>
              </div>

              <div className="col-span-1">
                <Panel title="Graph" density="compact">
                  <div className="h-[200px] overflow-hidden">
                    <Graph2D graph={live?.graph} height={200} />
                  </div>
                </Panel>
              </div>

              <div className="col-span-2">
                <Panel
                  title="Signals (HHI / HLI)"
                  right={<Badge tone="slate">f{live?.frame_idx ?? '-'}</Badge>}
                  density="compact"
                >
                  <div className="h-[260px] overflow-auto pr-1">
                    <div className="mb-3 flex flex-wrap items-center gap-2">
                      {live?.action_summary?.accuracy_metrics ? (
                        <>
                          <Badge tone="slate">
                            acc{' '}
                            {(
                              Number(live.action_summary.accuracy_metrics.estimated_accuracy ?? 0) *
                              100
                            ).toFixed(1)}
                            %
                          </Badge>
                          <Badge tone="slate">
                            high{' '}
                            {(
                              Number(live.action_summary.accuracy_metrics.high_confidence_rate ?? 0) *
                              100
                            ).toFixed(1)}
                            %
                          </Badge>
                          <Badge tone="slate">
                            actions {live.action_summary.accuracy_metrics.total_actions ?? 0}
                          </Badge>
                          <Badge tone="slate">
                            factor{' '}
                            {Number(live.action_summary.accuracy_metrics.factor_consistency ?? 0).toFixed(
                              2
                            )}
                          </Badge>
                        </>
                      ) : (
                        <Badge tone="slate">waiting for AFGN metrics</Badge>
                      )}
                    </div>

                    <div className="space-y-3">
                      <TripletList
                        title="HHI: player-to-player"
                        items={live?.hhi}
                        raiderId={live?.raider_id}
                        kind="HHI"
                      />
                      <TripletList
                        title="HLI: player-to-line"
                        items={live?.hli}
                        raiderId={live?.raider_id}
                        kind="HLI"
                      />
                    </div>
                  </div>
                </Panel>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-3">
          <Panel title="Gallery" density="compact">
            <div className="h-[320px] overflow-auto pr-1">
              <GalleryGrid players={live?.gallery} raiderId={live?.raider_id} />
            </div>
          </Panel>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
          <Panel
            title="Confirmed Events (AFGN)"
            right={<Badge tone="slate">{events.length}</Badge>}
          >
            <div className="max-h-[420px] space-y-2 overflow-auto pr-1">
              {events.length ? (
                events
                  .slice()
                  .reverse()
                  .map((ev) => {
                    const isContact = ev?.type === 'CONFIRMED_RAIDER_DEFENDER_CONTACT'
                    const label = ev?.classifier_label
                    const tone =
                      label === 'valid'
                        ? 'emerald'
                        : label === 'invalid'
                          ? 'rose'
                          : label
                            ? 'amber'
                            : 'slate'
                    return (
                      <button
                        key={ev.id}
                        className="w-full rounded-xl border border-stone-200 bg-white px-3 py-2 text-left hover:bg-stone-50 dark:border-stone-800 dark:bg-stone-950/20 dark:hover:bg-stone-900/40"
                        onClick={() => setSelectedEventId(ev.id)}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="min-w-0">
                            <div className="truncate text-sm font-semibold text-stone-900 dark:text-stone-50">
                              {formatEventType(ev.type)}
                            </div>
                            <div className="mt-0.5 text-[11px] text-stone-600 dark:text-stone-400">
                              <span className="tabular-nums">
                                f{ev.frame ?? '-'}
                              </span>
                              <span className="text-stone-300"> · </span>
                              <span className="tabular-nums">
                                conf {ev.conf ?? '-'}
                              </span>
                              <span className="text-stone-300"> · </span>
                              <span className="tabular-nums">
                                factor {ev.factor_conf ?? '-'}
                              </span>
                            </div>
                          </div>
                          <div className="shrink-0">
                            {ev.requires_visual_confirmation ? (
                              <Badge tone="amber">needs check</Badge>
                            ) : (
                              <Badge tone="slate">auto</Badge>
                            )}
                          </div>
                        </div>

                        <div className="mt-2 flex flex-wrap items-center gap-2">
                          <Badge tone="slate">
                            s/o {ev.subject ?? '-'} / {ev.object ?? '-'}
                          </Badge>
                          {label ? (
                            <Badge tone={tone}>
                              classifier: {label}
                              {typeof ev.classifier_valid_prob === 'number'
                                ? ` (${ev.classifier_valid_prob})`
                                : ''}
                            </Badge>
                          ) : (
                            <Badge tone="slate">classifier: pending</Badge>
                          )}
                          {ev.guaranteed_by_classifier ? (
                            <Badge tone="emerald">guaranteed</Badge>
                          ) : null}
                          {isContact && label === 'valid' ? (
                            <Badge tone="emerald">+1</Badge>
                          ) : null}
                        </div>
                      </button>
                    )
                  })
              ) : (
                <div className="rounded-xl border border-dashed border-stone-200 bg-stone-50 px-3 py-3 text-xs text-stone-600">
                  Waiting for confirmed events. Start the backend processing loop
                  and ensure `api_server.py` is being started by `Court_code2.py`.
                </div>
              )}
            </div>
          </Panel>

          <Panel title="Latest Output Videos" density="compact">
            <div className="space-y-3">
              <div className="text-xs text-stone-600 dark:text-stone-400">
                Shows the latest generated `.mp4` outputs once processing finishes
                writing them to `Kabaddi_video_processing/Videos/`.
              </div>
              <div className="space-y-2">
                <div className="text-[11px] font-medium text-stone-600 dark:text-stone-400">
                  Processed (tracked)
                </div>
                <VideoPreview
                  title="Processed video"
                  mp4Src={
                    latestVideos.processed && endpoints
                      ? endpoints.videoFile(latestVideos.processed)
                      : ''
                  }
                  mjpegSrc={
                    latestVideos.processed && endpoints
                      ? endpoints.videoMjpeg(latestVideos.processed)
                      : ''
                  }
                />
              </div>

              <div className="space-y-2">
                <div className="text-[11px] font-medium text-stone-600 dark:text-stone-400">
                  Confirmed report
                </div>
                <VideoPreview
                  title="Report video"
                  mp4Src={
                    latestVideos.report && endpoints
                      ? endpoints.videoFile(latestVideos.report)
                      : ''
                  }
                  mjpegSrc={
                    latestVideos.report && endpoints
                      ? endpoints.videoMjpeg(latestVideos.report)
                      : ''
                  }
                />
              </div>
            </div>
          </Panel>

          <Panel title="Live State (Debug)" density="compact">
            <pre className="max-h-[420px] overflow-auto rounded-xl border border-stone-200 bg-stone-950 p-3 text-[11px] leading-relaxed text-stone-100 dark:border-stone-800">
              {JSON.stringify(live, null, 2)}
            </pre>
          </Panel>
        </div>
      </main>
      ) : null}

      {showDashboard && selectedEventId ? (
        <div
          className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-4 backdrop-blur-sm"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setSelectedEventId(null)
          }}
        >
          <div className="w-full max-w-5xl overflow-hidden rounded-2xl border border-stone-200 bg-white shadow-xl dark:border-stone-800 dark:bg-stone-950">
            <div className="flex items-start justify-between gap-3 border-b border-stone-100 px-4 py-3 dark:border-stone-800">
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold text-stone-900 dark:text-stone-50">
                  {selectedEvent ? formatEventType(selectedEvent.type) : 'Event'}
                </div>
                <div className="mt-0.5 text-[11px] text-stone-600 dark:text-stone-400">
                  {selectedEventId}
                </div>
              </div>
              <button
                className="rounded-xl border border-stone-200 bg-white px-3 py-1.5 text-xs font-medium hover:bg-stone-50 dark:border-stone-800 dark:bg-stone-900/60 dark:hover:bg-stone-900"
                onClick={() => setSelectedEventId(null)}
              >
                Close
              </button>
            </div>

            <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-2">
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="slate">AFGN conf {selectedEvent?.conf ?? '-'}</Badge>
                  <Badge tone="slate">
                    factor {selectedEvent?.factor_conf ?? '-'}
                  </Badge>
                  <Badge tone="slate">
                    classifier {selectedEvent?.classifier_label ?? 'pending'}
                    {typeof selectedEvent?.classifier_valid_prob === 'number'
                      ? ` (${selectedEvent.classifier_valid_prob})`
                      : ''}
                  </Badge>
                  {selectedEvent?.guaranteed_by_classifier ? (
                    <Badge tone="emerald">guaranteed</Badge>
                  ) : null}
                </div>

                <div className="rounded-2xl border border-stone-200 bg-stone-50 p-3 dark:border-stone-800 dark:bg-stone-900/30">
                  <div className="text-[11px] font-medium text-stone-600 dark:text-stone-400">
                    Exact window frames (exported clip)
                  </div>
                  {selectedDetails.status === 'ready' &&
                  selectedDetails.data?.payload?.event ? (
                    <div className="mt-1 text-[11px] text-stone-600 dark:text-stone-400">
                      <span className="tabular-nums">
                        window f
                        {selectedDetails.data.payload.event.window_start ?? '-'}-
                        {selectedDetails.data.payload.event.window_end ?? '-'}
                      </span>
                      {Array.isArray(selectedDetails.data.payload?.payload?.window_frames) ? (
                        <span className="ml-2 tabular-nums">
                          ({selectedDetails.data.payload.payload.window_frames.length} frames)
                        </span>
                      ) : null}
                    </div>
                  ) : null}
                  <div className="mt-2">
                    {selectedDetails.status === 'ready' &&
                    selectedDetails.data?.clip_url ? (
                      endpoints && selectedDetails.data?.clip_id ? (
                        <OverlayMjpegPlayer
                          title="Event clip"
                          baseSrc={endpoints.eventClipMjpegVis(selectedDetails.data.clip_id)}
                          replay3d={{
                            matWindow:
                              selectedDetails.data?.archive_event?.mat_window ||
                              selectedDetails.data?.payload?.payload?.mat_window ||
                              [],
                            poseWindow:
                              selectedDetails.data?.payload?.payload?.pose_window || [],
                            poseMeta:
                              selectedDetails.data?.payload?.payload?.pose_meta || null,
                            videoFileSrc: `${baseUrl}${selectedDetails.data.clip_url}`,
                            event:
                              selectedDetails.data?.payload?.event ||
                              selectedDetails.data?.archive_event ||
                              selectedEvent ||
                              null,
                            courtMeta:
                              selectedDetails.data?.court_meta ||
                              selectedDetails.data?.payload?.payload?.court_meta ||
                              null,
                          }}
                          overlayContent={
                            <div className="rounded-md bg-black/20 p-1">
                              <CourtMat2D
                                players={selectedMatSnapshot?.players ?? live?.gallery}
                                raiderId={selectedMatSnapshot?.raider_id ?? live?.raider_id}
                                height={110}
                                theme={theme}
                              />
                            </div>
                          }
                          height={260}
                          overlayOpacity={1}
                          allowFullscreen
                          theme={theme}
                        />
                      ) : (
                        <VideoPreview
                          title="Event clip"
                          mp4Src={`${baseUrl}${selectedDetails.data.clip_url}`}
                          mjpegSrc={
                            endpoints && selectedDetails.data?.clip_id
                              ? endpoints.eventClipMjpeg(selectedDetails.data.clip_id)
                              : ''
                          }
                          allowFullscreen
                        />
                      )
                    ) : selectedDetails.status === 'loading' ? (
                      <div className="grid h-[220px] place-items-center rounded-xl border border-dashed border-stone-200 bg-white text-xs text-stone-500 dark:border-stone-800 dark:bg-stone-950/30 dark:text-stone-400">
                        Loading clip...
                      </div>
                    ) : (
                      <div className="grid h-[220px] place-items-center rounded-xl border border-dashed border-stone-200 bg-white text-xs text-stone-500 dark:border-stone-800 dark:bg-stone-950/30 dark:text-stone-400">
                        Clip not available yet (wait for exporter/classifier step).
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="text-[11px] font-medium text-stone-600 dark:text-stone-400">
                  AFGN reasoning + features (payload)
                </div>

                {selectedDetails.status === 'error' ? (
                  <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800 dark:border-rose-900/60 dark:bg-rose-950/40 dark:text-rose-100">
                    {selectedDetails.error}
                  </div>
                ) : null}

                <pre className="max-h-[420px] overflow-auto rounded-xl border border-stone-200 bg-stone-950 p-3 text-[11px] leading-relaxed text-stone-100 dark:border-stone-800">
                  {selectedDetails.status === 'ready'
                    ? JSON.stringify(selectedDetails.data?.payload, null, 2)
                    : selectedDetails.status === 'loading'
                      ? 'Loading payload...'
                      : 'Payload not available yet.'}
                </pre>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default App
