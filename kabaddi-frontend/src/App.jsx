import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import Graph2D from './Graph2D.jsx'
import CourtMat2D from './CourtMat2D.jsx'
import RaidReplay3D from './RaidReplay3D.jsx'
import ArchitecturePage from './ArchitecturePage.jsx'

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

function PlayerAvatarIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 64 64"
      className="h-9 w-9"
      fill="none"
    >
      <circle cx="32" cy="24" r="11" fill="rgba(255,255,255,0.92)" />
      <path
        d="M16 54c1.8-10.2 8.8-16 16-16s14.2 5.8 16 16"
        fill="rgba(255,255,255,0.92)"
      />
    </svg>
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
  const replayAudioRef = useRef(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [show3D, setShow3D] = useState(false)
  const [overlayFrameIndex, setOverlayFrameIndex] = useState(0)
  const [isReplayAudioMuted, setIsReplayAudioMuted] = useState(true)

  const archiveMatWindow =
    Array.isArray(replay3d?.matWindow) && replay3d.matWindow.length > 0
      ? replay3d.matWindow
      : null

  useEffect(() => {
    setOverlayFrameIndex(0)
  }, [archiveMatWindow, baseSrc])

  useEffect(() => {
    if (!archiveMatWindow || archiveMatWindow.length <= 1) return
    const timer = window.setInterval(() => {
      setOverlayFrameIndex((prev) => (prev + 1) % archiveMatWindow.length)
    }, 1000 / 12)
    return () => window.clearInterval(timer)
  }, [archiveMatWindow])

  const animatedOverlaySnapshot = archiveMatWindow
    ? archiveMatWindow[Math.max(0, Math.min(archiveMatWindow.length - 1, overlayFrameIndex))]
    : null

  const can3D =
    !!replay3d &&
    Array.isArray(replay3d.matWindow) &&
    replay3d.matWindow.length > 0

  const canPoseOverlay =
    !!replay3d &&
    Array.isArray(replay3d.poseWindow) &&
    replay3d.poseWindow.length > 0

  useEffect(() => {
    // 3D view is expanded-view only.
    if (!isFullscreen) setShow3D(false)
  }, [isFullscreen])

  useEffect(() => {
    const audio = replayAudioRef.current
    if (!audio) return undefined

    audio.muted = isReplayAudioMuted

    if (!show3D || !can3D) {
      audio.pause()
      audio.currentTime = 0
      return undefined
    }

    const playPromise = audio.play()
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(() => {})
    }

    return () => {
      audio.pause()
    }
  }, [show3D, can3D, isReplayAudioMuted])

  useEffect(() => {
    if (!isFullscreen || typeof document === 'undefined') return undefined

    const { body, documentElement } = document
    const prevBodyOverflow = body.style.overflow
    const prevHtmlOverflow = documentElement.style.overflow
    const prevBodyOverscroll = body.style.overscrollBehavior
    const prevHtmlOverscroll = documentElement.style.overscrollBehavior

    body.style.overflow = 'hidden'
    documentElement.style.overflow = 'hidden'
    body.style.overscrollBehavior = 'none'
    documentElement.style.overscrollBehavior = 'none'

    return () => {
      body.style.overflow = prevBodyOverflow
      documentElement.style.overflow = prevHtmlOverflow
      body.style.overscrollBehavior = prevBodyOverscroll
      documentElement.style.overscrollBehavior = prevHtmlOverscroll
    }
  }, [isFullscreen])

  const onFullscreen = async () => {
    if (!allowFullscreen) return
    setIsFullscreen(true)
  }

  const onCloseFullscreen = async () => {
    setIsFullscreen(false)
    setShow3D(false)
  }

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

      {isFullscreen ? (
        <div
          className="fixed inset-0 z-[120] bg-black/55 backdrop-blur-sm"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) onCloseFullscreen()
          }}
        />
      ) : null}

      <div
        ref={wrapRef}
        className={`overflow-hidden border border-stone-200 bg-stone-50 dark:border-stone-800 dark:bg-stone-950/40 ${
          isFullscreen
            ? 'fixed inset-4 z-[130] rounded-2xl shadow-2xl'
            : 'relative rounded-xl'
        }`}
        style={isFullscreen ? undefined : { height }}
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
                  <div className="w-full">
                    {typeof overlayContent === 'function'
                      ? overlayContent(animatedOverlaySnapshot)
                      : overlayContent}
                  </div>
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
              <>
                <audio
                  ref={replayAudioRef}
                  src="/soundpm.mp3"
                  loop
                  muted={isReplayAudioMuted}
                  preload="auto"
                />
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
              </>
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

function hsvSwatchColor(binIdx, value, theme = 'dark') {
  const strength = Math.max(0.08, Math.min(1, Number(value) || 0))
  const hue = Math.round((binIdx / 5) * 360)
  const sat = Math.round(42 + strength * 50)
  const light = theme === 'dark' ? Math.round(28 + strength * 26) : Math.round(62 - strength * 16)
  return `hsl(${hue} ${sat}% ${light}%)`
}

function rgbSwatch(color) {
  if (!Array.isArray(color) || color.length < 3) return 'rgb(148 163 184)'
  const r = Math.max(0, Math.min(255, Number(color[0]) || 0))
  const g = Math.max(0, Math.min(255, Number(color[1]) || 0))
  const b = Math.max(0, Math.min(255, Number(color[2]) || 0))
  return `rgb(${r} ${g} ${b})`
}

function playerContributionScore(pid, events, currentRaid, raiderId) {
  let total = 0
  for (const ev of Array.isArray(events) ? events : []) {
    const sameRaid =
      currentRaid?.raid_label ? String(ev?.raid_label || '') === String(currentRaid.raid_label) : true
    if (!sameRaid) continue
    const involved =
      Number(ev?.subject) === Number(pid) ||
      Number(ev?.object) === Number(pid)
    if (!involved) continue
    if (String(ev?.classifier_label || '').toLowerCase() === 'valid') total += 1
    if (Number(pid) === Number(raiderId) && String(ev?.type || '').includes('BONUS')) total += 1
  }
  return total
}

function Speedometer({ value = 0, theme = 'dark' }) {
  const v = Math.max(0, Number(value) || 0)
  const capped = Math.min(1, v / 7)
  const angle = -120 + capped * 240
  const isDark = theme === 'dark'
  const needleColor = isDark ? '#f5deb3' : '#7c5b2a'
  const arcBg = isDark ? 'rgba(148,163,184,0.22)' : 'rgba(148,163,184,0.35)'
  const arcFg = isDark ? 'rgba(226,232,240,0.92)' : 'rgba(51,65,85,0.92)'

  return (
    <div className="rounded-xl bg-white px-3 py-2 text-[11px] text-stone-600 dark:bg-stone-950/40 dark:text-stone-300">
      <div className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-500">Speed</div>
      <div className="mt-2 flex items-center gap-3">
        <div className="relative h-14 w-20 shrink-0">
          <svg viewBox="0 0 120 80" className="h-full w-full">
            <path
              d="M 16 64 A 44 44 0 0 1 104 64"
              fill="none"
              stroke={arcBg}
              strokeWidth="10"
              strokeLinecap="round"
            />
            <path
              d="M 16 64 A 44 44 0 0 1 104 64"
              fill="none"
              stroke={arcFg}
              strokeWidth="10"
              strokeLinecap="round"
              pathLength="100"
              strokeDasharray={`${capped * 100} 100`}
            />
            <g transform={`rotate(${angle} 60 64)`}>
              <line
                x1="60"
                y1="64"
                x2="60"
                y2="26"
                stroke={needleColor}
                strokeWidth="4"
                strokeLinecap="round"
              />
            </g>
            <circle cx="60" cy="64" r="5.5" fill={needleColor} />
          </svg>
        </div>
        <div>
          <div className="text-lg font-semibold tabular-nums text-stone-900 dark:text-stone-50">
            {v.toFixed(2)}
          </div>
          <div className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-500">
            units / frame
          </div>
        </div>
      </div>
    </div>
  )
}

function TeamScoreboard({
  players,
  raiderId,
  currentRaid,
  teamScores = { A: 0, B: 0 },
  events,
  theme = 'dark',
}) {
  if (!Array.isArray(players) || players.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-stone-200 bg-stone-50 px-3 py-3 text-xs text-stone-600 dark:border-stone-800 dark:bg-stone-950/30 dark:text-stone-400">
        No live team board yet.
      </div>
    )
  }

  const attackingTeam = currentRaid?.attacking_team || 'A'
  const defendingTeam = attackingTeam === 'A' ? 'B' : 'A'
  const isDark = theme === 'dark'
  const items = players
    .slice()
    .sort((a, b) => {
      const aRaid = Number(a?.id) === Number(raiderId) ? 0 : 1
      const bRaid = Number(b?.id) === Number(raiderId) ? 0 : 1
      return aRaid - bRaid || (a?.id ?? 0) - (b?.id ?? 0)
    })

  const grouped = { A: [], B: [] }
  for (const p of items) {
    const team = Number(p?.id) === Number(raiderId) ? attackingTeam : defendingTeam
    grouped[team].push(p)
  }

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
      {[
        { team: 'A', items: grouped.A },
        { team: 'B', items: grouped.B },
      ].map((section) => (
        <div
          key={section.team}
          className="rounded-2xl border border-stone-200 bg-white/90 p-3 shadow-sm dark:border-stone-800 dark:bg-stone-950/25 dark:shadow-none"
        >
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <div className="text-sm font-semibold text-stone-900 dark:text-stone-50">
                  Team {section.team}
                </div>
                {section.team === attackingTeam ? (
                  <Badge tone="amber">attacking</Badge>
                ) : (
                  <Badge tone="slate">defending</Badge>
                )}
              </div>
              <div className="mt-1 text-[11px] text-stone-600 dark:text-stone-400">
                {currentRaid?.raid_label || 'live raid'} ? {section.items.length} player
                {section.items.length === 1 ? '' : 's'}
              </div>
            </div>
            <div className="rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 text-right dark:border-stone-800 dark:bg-stone-900/50">
              <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-stone-500 dark:text-stone-400">
                Team Score
              </div>
              <div className="mt-1 text-2xl font-semibold tabular-nums text-stone-900 dark:text-stone-50">
                {Number(teamScores?.[section.team] ?? 0)}
              </div>
            </div>
          </div>

          <div className="mt-3 space-y-2">
            {section.items.length ? section.items.map((p) => {
              const isRaider = raiderId != null && Number(p?.id) === Number(raiderId)
              const visible = !!p?.visible
              const age = Number(p?.age ?? 0)
              const spd = _speed(p?.velocity)
              const flowPoints = Number(p?.flow_points ?? 0)
              const hsv = Array.isArray(p?.hsv_bins5) ? p.hsv_bins5 : []
              const dominantColors = Array.isArray(p?.dominant_colors) ? p.dominant_colors : []
              const individualScore = playerContributionScore(p?.id, events, currentRaid, raiderId)
              const avatarBase = dominantColors.length >= 2
                ? `linear-gradient(135deg, ${rgbSwatch(dominantColors[0])}, ${rgbSwatch(dominantColors[1])})`
                : dominantColors.length === 1
                  ? rgbSwatch(dominantColors[0])
                  : hsv.length
                    ? `linear-gradient(135deg, ${hsvSwatchColor(0, hsv[0], theme)}, ${hsvSwatchColor(2, hsv[2] ?? hsv[0], theme)})`
                    : isDark
                      ? 'linear-gradient(135deg, rgba(148,163,184,0.6), rgba(71,85,105,0.9))'
                      : 'linear-gradient(135deg, rgba(226,232,240,1), rgba(148,163,184,0.9))'

              return (
                <div
                  key={p?.id ?? Math.random()}
                  className="rounded-2xl border border-stone-200 bg-stone-50/70 p-3 dark:border-stone-800 dark:bg-stone-900/35"
                >
                  <div className="flex items-start gap-3">
                    <div
                      className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl text-sm font-semibold text-white shadow-sm"
                      style={{ background: avatarBase }}
                    >
                      <PlayerAvatarIcon />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="truncate text-sm font-semibold text-stone-900 dark:text-stone-50">
                          {isRaider ? `Raider ? ${_fmtPid(p?.id)}` : _fmtPid(p?.id)}
                        </div>
                        {isRaider ? (
                          <Badge tone="amber">focus</Badge>
                        ) : visible ? (
                          <Badge tone="emerald">live</Badge>
                        ) : (
                          <Badge tone="slate">hold</Badge>
                        )}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-stone-600 dark:text-stone-400">
                        <span>{visible ? 'visible now' : `lost (age ${age})`}</span>
                        <span>court {Number(p?.court_pos?.[0] ?? 0).toFixed(2)}, {Number(p?.court_pos?.[1] ?? 0).toFixed(2)}</span>
                        <span>bbox {Array.isArray(p?.bbox) ? p.bbox.map((v) => Math.round(Number(v) || 0)).join(', ') : '-'}</span>
                      </div>
                    </div>
                    <div className="rounded-xl border border-stone-200 bg-white px-3 py-2 text-right dark:border-stone-800 dark:bg-stone-950/40">
                      <div className="text-[10px] font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">
                        Individual
                      </div>
                      <div className="mt-0.5 text-lg font-semibold tabular-nums text-stone-900 dark:text-stone-50">
                        {individualScore}
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 grid grid-cols-2 gap-2 lg:grid-cols-4">
                    <Speedometer value={spd} theme={theme} />
                    <div className="rounded-xl bg-white px-3 py-2 text-[11px] text-stone-600 dark:bg-stone-950/40 dark:text-stone-300">
                      <div className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-500">Flow Pts</div>
                      <div className="mt-1 font-medium tabular-nums text-stone-900 dark:text-stone-50">{flowPoints}</div>
                    </div>
                    <div className="rounded-xl bg-white px-3 py-2 text-[11px] text-stone-600 dark:bg-stone-950/40 dark:text-stone-300">
                      <div className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-500">Track Age</div>
                      <div className="mt-1 font-medium tabular-nums text-stone-900 dark:text-stone-50">{age}</div>
                    </div>
                    <div className="rounded-xl bg-white px-3 py-2 text-[11px] text-stone-600 dark:bg-stone-950/40 dark:text-stone-300">
                      <div className="text-[10px] uppercase tracking-wide text-stone-500 dark:text-stone-500">Role</div>
                      <div className="mt-1 font-medium text-stone-900 dark:text-stone-50">{isRaider ? 'raider' : 'defender'}</div>
                    </div>
                  </div>

                  <div className="mt-3 rounded-xl border border-stone-200 bg-white px-3 py-2 dark:border-stone-800 dark:bg-stone-950/40">
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-stone-500 dark:text-stone-400">
                        Current Color Palette
                      </div>
                      <div className="text-[10px] text-stone-500 dark:text-stone-400">
                        major colors inside bbox
                      </div>
                    </div>
                    <div className="mt-2 grid grid-cols-4 gap-2">
                      {(dominantColors.length ? dominantColors : []).slice(0, 4).map((color, idx) => (
                        <div key={idx} className="min-w-0">
                          <div
                            className="h-11 rounded-xl"
                            style={{
                              background: rgbSwatch(color),
                              boxShadow: isDark ? 'inset 0 0 0 1px rgba(255,255,255,0.08)' : 'inset 0 0 0 1px rgba(15,23,42,0.08)',
                            }}
                            title={`rgb(${color.join(', ')})`}
                          />
                          <div className="mt-1 truncate text-center text-[10px] tabular-nums text-stone-500 dark:text-stone-400">
                            {Array.isArray(color) ? color.join(',') : '-'}
                          </div>
                        </div>
                      ))}
                      {!dominantColors.length ? (
                        <div className="col-span-4 rounded-lg border border-dashed border-stone-200 bg-stone-50 px-3 py-2 text-[11px] text-stone-500 dark:border-stone-800 dark:bg-stone-900/30 dark:text-stone-400">
                          Waiting for live dominant colors from the current bounding box.
                        </div>
                      ) : null}
                    </div>
                  </div>
                </div>
              )
            }) : (
              <div className="rounded-xl border border-dashed border-stone-200 bg-stone-50 px-3 py-3 text-xs text-stone-600 dark:border-stone-800 dark:bg-stone-950/30 dark:text-stone-400">
                No players mapped to Team {section.team} yet.
              </div>
            )}
          </div>
        </div>
      ))}
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

function FrameEventsList({ frameIdx, hhi, hli, events }) {
  const currentFrame = Number(frameIdx)
  const hhiItems = []
  const hliItems = []

  if (Number.isFinite(currentFrame)) {
    for (const item of Array.isArray(hhi) ? hhi : []) {
      if (Number(item?.frame) !== currentFrame) continue
      hhiItems.push({
        key: `hhi-${item?.S}-${item?.O}-${item?.I}`,
        tone: 'slate',
        title: 'HHI contact',
        detail: `${_fmtPid(item?.S)} -> ${_fmtPid(item?.O)} | ${item?.I ?? '-'} | d ${Number(item?.dist ?? 0).toFixed(2)} | rv ${Number(item?.rel_vel ?? 0).toFixed(2)}`,
      })
    }
    for (const item of Array.isArray(events) ? events : []) {
      if (Number(item?.frame) !== currentFrame) continue
      const detail = `${item?.subject ?? '-'} / ${item?.object ?? '-'} | conf ${item?.conf ?? '-'} | factor ${item?.factor_conf ?? '-'}`
      if (String(item?.type || '').includes('CONTACT')) {
        hhiItems.push({
          key: item?.id ?? `ev-hhi-${item?.type}-${item?.frame}`,
          tone: item?.classifier_label === 'valid' ? 'emerald' : 'rose',
          title: formatEventType(item?.type),
          detail,
        })
      } else {
        hliItems.push({
          key: item?.id ?? `ev-hli-${item?.type}-${item?.frame}`,
          tone: item?.classifier_label === 'valid' ? 'emerald' : 'amber',
          title: formatEventType(item?.type),
          detail,
        })
      }
    }
    for (const item of Array.isArray(hli) ? hli : []) {
      if (Number(item?.frame) !== currentFrame) continue
      hliItems.push({
        key: `hli-${item?.S}-${item?.O}-${item?.I}`,
        tone: item?.active ? 'amber' : 'slate',
        title: item?.active ? 'Line touch' : 'Line near',
        detail: `${_fmtPid(item?.S)} -> ${String(item?.O ?? '-')} | ${item?.I ?? '-'} | d ${Number(item?.dist ?? 0).toFixed(2)}`,
      })
    }
  }

  return (
    <div className="space-y-3">
      {[
        { title: 'HHI events', items: hhiItems },
        { title: 'HLI events', items: hliItems },
      ].map((section) => (
        <div
          key={section.title}
          className="rounded-xl border border-stone-200 bg-white p-3 dark:border-stone-800 dark:bg-stone-950/20"
        >
          <div className="flex items-center justify-between gap-2">
            <div className="text-[11px] font-semibold text-stone-700 dark:text-stone-200">
              {section.title}
            </div>
            <Badge tone="slate">{section.items.length}</Badge>
          </div>
          {section.items.length ? (
            <div className="mt-2 space-y-1.5">
              {section.items.map((item) => (
                <div
                  key={item.key}
                  className="rounded-lg bg-stone-50 px-2 py-2 text-[11px] text-stone-700 dark:bg-stone-900/40 dark:text-stone-200"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-semibold">{item.title}</div>
                    <Badge tone={item.tone}>
                      {item.tone === 'emerald'
                        ? 'confirmed'
                        : item.tone === 'amber'
                          ? 'active'
                          : item.tone === 'rose'
                            ? 'rejected'
                            : 'live'}
                    </Badge>
                  </div>
                  <div className="mt-1 text-stone-500 dark:text-stone-400">
                    {item.detail}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-2 text-xs text-stone-600 dark:text-stone-400">
              No {section.title.toLowerCase()} at this frame.
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

function ScoreStripInner({
  aName = 'Team A',
  bName = 'Team B',
  a = 0,
  b = 0,
  currentRaid = null,
  raidSummaries = [],
}) {
  const [showReview, setShowReview] = useState(false)
  const items = raidSummaries
    .slice()
    .sort((x, y) => Number(x?.raid_index ?? 0) - Number(y?.raid_index ?? 0))

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
          {currentRaid?.raid_label ? (
            <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-stone-600 dark:text-stone-400">
              <Badge tone="slate">{currentRaid.raid_label}</Badge>
              <span>
                Team {currentRaid.attacking_team ?? '-'} raid
              </span>
              {currentRaid.raider_id != null ? (
                <span>Raider {_fmtPid(currentRaid.raider_id)}</span>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-baseline gap-2">
            <div className="tabular-nums text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-50">
              {a}
            </div>
            <div className="text-stone-400">:</div>
            <div className="tabular-nums text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-50">
              {b}
            </div>
          </div>
          <button
            type="button"
            onClick={() => setShowReview((v) => !v)}
            className="rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs font-medium shadow-sm hover:bg-stone-50 dark:border-stone-800 dark:bg-stone-900/60 dark:hover:bg-stone-900"
          >
            {showReview ? 'Hide Actions' : 'Review Actions'}
          </button>
        </div>
      </div>

      {showReview ? (
        <div className="mt-4 max-h-[340px] space-y-3 overflow-auto pr-1">
          {items.length ? (
            items.map((raid) => (
              <div
                key={`${raid?.raid_label ?? 'raid'}-${raid?.raid_index ?? 0}`}
                className="rounded-2xl border border-stone-200 bg-white px-3 py-3 dark:border-stone-800 dark:bg-stone-950/30"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone={raid?.status === 'live' ? 'amber' : 'slate'}>
                      {raid?.raid_label ?? `raid${raid?.raid_index ?? '-'}`}
                    </Badge>
                    <span className="text-sm font-semibold text-stone-900 dark:text-stone-50">
                      Team {raid?.attacking_team ?? '-'} raid
                    </span>
                    {raid?.raider_id != null ? (
                      <span className="text-xs text-stone-600 dark:text-stone-400">
                        Raider {_fmtPid(raid.raider_id)}
                      </span>
                    ) : null}
                  </div>
                  <div className="text-xs text-stone-600 dark:text-stone-400">
                    score after: {raid?.team_scores?.A ?? 0} - {raid?.team_scores?.B ?? 0}
                  </div>
                </div>

                <div className="mt-3 grid gap-3 lg:grid-cols-2">
                  <div className="rounded-xl bg-stone-50 px-3 py-2 dark:bg-stone-900/40">
                    <div className="text-[11px] font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">
                      Score Calculation
                    </div>
                    {Array.isArray(raid?.score_breakdown) && raid.score_breakdown.length ? (
                      <div className="mt-2 space-y-1.5 text-xs text-stone-700 dark:text-stone-200">
                        {raid.score_breakdown.map((item, idx) => (
                          <div key={`${idx}-${item?.frame ?? 0}`} className="rounded-lg bg-white px-2 py-2 dark:bg-stone-950/30">
                            <span className="font-semibold">f{item?.frame ?? '-'}</span>{' '}
                            <span>Team {item?.team ?? '-'} +{item?.delta ?? 0}</span>{' '}
                            <span className="text-stone-500 dark:text-stone-400">
                              ({item?.reason ?? 'score'})
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="mt-2 text-xs text-stone-500 dark:text-stone-400">
                        No score changes recorded for this raid.
                      </div>
                    )}
                  </div>

                  <div className="rounded-xl bg-stone-50 px-3 py-2 dark:bg-stone-900/40">
                    <div className="text-[11px] font-semibold uppercase tracking-wide text-stone-500 dark:text-stone-400">
                      Player Actions
                    </div>
                    {Array.isArray(raid?.actions) && raid.actions.length ? (
                      <div className="mt-2 space-y-1.5 text-xs text-stone-700 dark:text-stone-200">
                        {raid.actions.map((action, idx) => (
                          <div
                            key={`${idx}-${action?.frame ?? 0}`}
                            className={`rounded-lg px-2 py-2 ${
                              Number(action?.points ?? 0) > 0 || action?.highlight
                                ? 'border border-emerald-200 bg-emerald-50 dark:border-emerald-900/50 dark:bg-emerald-950/20'
                                : 'bg-white dark:bg-stone-950/30'
                            }`}
                          >
                            <div className="font-semibold">
                              f{action?.frame ?? '-'} | {action?.type ?? 'ACTION'}
                            </div>
                            <div className="mt-1 text-stone-600 dark:text-stone-400">
                              {action?.description ?? 'No description'}
                            </div>
                            <div className="mt-1 flex flex-wrap items-center gap-2 text-stone-500 dark:text-stone-500">
                              points {action?.points ?? 0} | conf {Number(action?.confidence ?? 0).toFixed(2)}
                              {Number(action?.points ?? 0) > 0 || action?.highlight ? (
                                <Badge tone="emerald">point action</Badge>
                              ) : null}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="mt-2 text-xs text-stone-500 dark:text-stone-400">
                        No actions recorded yet.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="rounded-xl border border-dashed border-stone-200 bg-stone-50 px-3 py-3 text-xs text-stone-600 dark:border-stone-800 dark:bg-stone-950/30 dark:text-stone-400">
              No raid review data yet.
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}

function ScoreStrip({
  aName = 'Team A',
  bName = 'Team B',
  a = 0,
  b = 0,
  currentRaid = null,
  raidSummaries = [],
}) {
  return (
    <ScoreStripInner
      aName={aName}
      bName={bName}
      a={a}
      b={b}
      currentRaid={currentRaid}
      raidSummaries={raidSummaries}
    />
  )
}

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard')
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
  const [pipelineStep, setPipelineStep] = useState(null)
  const [archiveReview, setArchiveReview] = useState(null)
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
  const isArchitecturePage = currentPage === 'architecture'

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

  const eventsByRaid = useMemo(() => {
    const grouped = new Map()
    for (const ev of Array.isArray(events) ? events : []) {
      const raidIndex = Number(ev?.raid_index)
      const raidLabel =
        ev?.raid_label || (Number.isFinite(raidIndex) && raidIndex > 0 ? `raid${raidIndex}` : 'Unassigned')
      if (!grouped.has(raidLabel)) {
        grouped.set(raidLabel, {
          raidLabel,
          raidIndex: Number.isFinite(raidIndex) ? raidIndex : Number.MAX_SAFE_INTEGER,
          items: [],
        })
      }
      grouped.get(raidLabel).items.push(ev)
    }
    return Array.from(grouped.values())
      .sort((a, b) => a.raidIndex - b.raidIndex || a.raidLabel.localeCompare(b.raidLabel))
      .map((group) => ({
        ...group,
        items: group.items.slice().sort((a, b) => Number(b?.frame ?? 0) - Number(a?.frame ?? 0)),
      }))
  }, [events])

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
      pipelineStream: `${baseUrl}/api/pipeline/stream`,
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
    if (!endpoints || !showLive) {
      setPipelineStep(null)
      return
    }

    let closed = false
    let es = null

    try {
      es = new EventSource(endpoints.pipelineStream)
      es.onmessage = (msg) => {
        if (closed) return
        try {
          const payload = JSON.parse(msg.data)
          setPipelineStep(payload)
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
      }
    } catch {
      setPipelineStep(null)
    }

    return () => {
      closed = true
      try {
        es?.close()
      } catch {
        // ignore
      }
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
        setArchiveReview({
          raidSummaries: Array.isArray(payload?.raid_summaries) ? payload.raid_summaries : [],
          teamScores: {
            A: Number(payload?.team_scores?.A ?? 0),
            B: Number(payload?.team_scores?.B ?? 0),
          },
          currentRaid:
            Array.isArray(payload?.raid_summaries) && payload.raid_summaries.length
              ? payload.raid_summaries[payload.raid_summaries.length - 1]
              : null,
        })
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

  const validatedTouchSummary = useMemo(() => {
    let total = 0
    let teamA = 0
    let teamB = 0
    for (const ev of events) {
      if (ev?.type === 'CONFIRMED_RAIDER_DEFENDER_CONTACT' && ev?.classifier_label === 'valid') {
        total += 1
        const team = String(ev?.attacking_team || '').toUpperCase()
        if (team === 'B') teamB += 1
        else teamA += 1
      }
    }
    return { total, teamA, teamB }
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
    const attacker = live?.score_attacker ?? archiveReview?.teamScores?.A ?? 0
    const defender = live?.score_defender ?? archiveReview?.teamScores?.B ?? 0
    return { attacker, defender }
  }, [archiveReview, live])

  const displayedScoreboard = useMemo(() => {
    if (!health.ok) return { attacker: 0, defender: 0 }
    return {
      attacker: scoreboard.attacker ?? 0,
      defender: scoreboard.defender ?? 0,
    }
  }, [health.ok, scoreboard.attacker, scoreboard.defender])

  useEffect(() => {
    if (!showDashboard || isArchitecturePage) setSelectedEventId(null)
  }, [showDashboard, isArchitecturePage])

  useEffect(() => {
    // If we transition from offline -> online, don't keep showing cached outputs.
    if (health.ok && !wasOnlineRef.current) {
      wasOnlineRef.current = true
      setLatestVideos({ processed: null, report: null })
      setArchiveReview(null)
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
      setArchiveReview(null)
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
      setArchiveReview(null)
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
                {isArchitecturePage ? 'Kabaddi System Architecture' : 'Kabaddi Live Dashboard'}
              </h1>
              {!isArchitecturePage ? connBadge : null}
            </div>
            <p className="mt-1 text-xs text-stone-600 dark:text-stone-400">
              {isArchitecturePage
                ? 'A visual map of the project workflow from raw raid video to validated outputs.'
                : 'Input, tracking, mat, interaction graph, events, and classifier validation.'}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              className="rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs shadow-sm hover:bg-stone-50 dark:border-stone-800 dark:bg-stone-900/60 dark:hover:bg-stone-900"
              onClick={() =>
                setCurrentPage((page) =>
                  page === 'dashboard' ? 'architecture' : 'dashboard',
                )
              }
            >
              {isArchitecturePage ? 'Open Dashboard' : 'Open Architecture'}
            </button>
            <button
              className="rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs shadow-sm hover:bg-stone-50 dark:border-stone-800 dark:bg-stone-900/60 dark:hover:bg-stone-900"
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
            >
              {theme === 'dark' ? 'Dark' : 'Light'}
            </button>

            {showDashboard && !isArchitecturePage ? (
              <>
                <div className="rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs dark:border-stone-800 dark:bg-stone-900/60">
                  <span className="text-stone-500 dark:text-stone-400">
                    Validated touches:
                  </span>{' '}
                  <span className="font-semibold tabular-nums">{validatedTouchSummary.total}</span>
                  
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

        {!isArchitecturePage ? (
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
        ) : null}
      </header>

      {isArchitecturePage ? (
        <ArchitecturePage pipelineStep={pipelineStep} showLive={showLive} />
      ) : !health.ok && offlineOutputs ? (
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
                title={`Input Video (Backend)${live?.raid_label ? ` · ${live.raid_label}` : ''}`}
                right={<Badge tone="slate">MJPEG</Badge>}
              >
                <StreamView
                  src={endpoints?.inputStream}
                  alt="Input stream"
                  height={420}
                />
              </Panel>

              <Panel
                title={`Processed / Tracked Video${live?.raid_label ? ` · ${live.raid_label}` : ''}`}
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
                currentRaid={live?.current_raid ?? archiveReview?.currentRaid ?? null}
                raidSummaries={live?.raid_summaries ?? archiveReview?.raidSummaries ?? []}
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

                    <FrameEventsList
                      frameIdx={live?.frame_idx}
                      hhi={live?.hhi}
                      hli={live?.hli}
                      events={live?.events}
                    />
                  </div>
                </Panel>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-3">
          <Panel title="Team Board" density="compact">
            <div className="h-[420px] overflow-auto pr-1">
              <TeamScoreboard
                players={live?.gallery}
                raiderId={live?.raider_id}
                currentRaid={live?.current_raid ?? archiveReview?.currentRaid ?? null}
                teamScores={{
                  A: displayedScoreboard.attacker,
                  B: displayedScoreboard.defender,
                }}
                events={events}
                theme={theme}
              />
            </div>
          </Panel>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
          <Panel
            title="Confirmed Events (AFGN)"
            right={<Badge tone="slate">{events.length}</Badge>}
          >
            <div className="max-h-[420px] space-y-2 overflow-auto pr-1">
              {eventsByRaid.length ? (
                eventsByRaid.map((group) => (
                  <div
                    key={group.raidLabel}
                    className="rounded-2xl border border-stone-200 bg-stone-50/70 p-2 dark:border-stone-800 dark:bg-stone-950/20"
                  >
                    <div className="mb-2 flex items-center justify-between gap-2 px-1">
                      <div className="flex items-center gap-2">
                        <Badge tone="slate">{group.raidLabel}</Badge>
                        <div className="text-xs font-medium text-stone-600 dark:text-stone-400">
                          {group.items.length} event{group.items.length === 1 ? '' : 's'}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      {group.items.map((ev) => {
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
                                  <span className="text-stone-300"> | </span>
                                  <span className="tabular-nums">
                                    conf {ev.conf ?? '-'}
                                  </span>
                                  <span className="text-stone-300"> | </span>
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
                      })}
                    </div>
                  </div>
                ))
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

      {showDashboard && !isArchitecturePage && selectedEventId ? (
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
                          overlayContent={(overlaySnapshot) => (
                            <div className="rounded-md bg-black/20 p-1">
                              <CourtMat2D
                                players={
                                  overlaySnapshot?.players ??
                                  selectedMatSnapshot?.players ??
                                  live?.gallery
                                }
                                raiderId={
                                  overlaySnapshot?.raider_id ??
                                  selectedMatSnapshot?.raider_id ??
                                  live?.raider_id
                                }
                                height={110}
                                theme={theme}
                              />
                            </div>
                          )}
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

