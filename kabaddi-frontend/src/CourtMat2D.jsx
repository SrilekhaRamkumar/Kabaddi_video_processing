import { useMemo } from 'react'

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n))
}

function fmtId(id) {
  const n = Number(id)
  if (!Number.isFinite(n)) return String(id ?? '-')
  return String(n).padStart(2, '0')
}

export default function CourtMat2D({
  players,
  raiderId,
  height = 220,
  theme,
}) {
  const items = useMemo(() => {
    if (!Array.isArray(players)) return []
    return players
      .filter((p) => Array.isArray(p?.court_pos) && p.court_pos.length >= 2)
      .map((p) => ({
        id: p.id,
        visible: !!p.visible,
        x: Number(p.court_pos[0]) || 0,
        y: Number(p.court_pos[1]) || 0,
      }))
  }, [players])

  const isDark = theme
    ? theme === 'dark'
    : typeof document !== 'undefined'
      ? document.documentElement.classList.contains('dark')
      : true

  // Court dimensions in meters (same as backend homography mapping).
  const COURT_W = 10.0
  const COURT_H = 6.5
  const BAULK_Y = 3.75
  const BONUS_Y = 4.75
  const LOBBY_L = 0.75
  const LOBBY_R = 9.25

  const stroke = isDark ? '#cbd5e1' : '#334155' // slate-300 vs slate-700
  const stroke2 = isDark ? '#94a3b8' : '#475569' // slate-400 vs slate-600
  const text = isDark ? '#e2e8f0' : '#0f172a' // slate-200 vs slate-900
  const faint = isDark ? 'rgba(255, 255, 255, 0.35)' : 'rgba(15, 23, 42, 0.35)'

  return (
    <div style={{ width: '100%', height }} className="relative overflow-hidden">
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${COURT_W} ${COURT_H}`}
        preserveAspectRatio="xMidYMid meet"
        className="h-full w-full"
        style={{ overflow: 'hidden', display: 'block' }}
      >
        {/* Transparent background by default */}

        {/* Outer boundary */}
        <rect
          x="0"
          y="0"
          width={COURT_W}
          height={COURT_H}
          fill="transparent"
          stroke={stroke}
          strokeWidth="0.08"
        />

        {/* Baulk & bonus lines */}
        <line
          x1="0"
          y1={COURT_H - BAULK_Y}
          x2={COURT_W}
          y2={COURT_H - BAULK_Y}
          stroke={stroke2}
          strokeWidth="0.05"
          opacity="0.9"
        />
        <line
          x1="0"
          y1={COURT_H - BONUS_Y}
          x2={COURT_W}
          y2={COURT_H - BONUS_Y}
          stroke={stroke2}
          strokeWidth="0.05"
          opacity="0.9"
        />

        {/* Lobby bounds */}
        <line
          x1={LOBBY_L}
          y1="0"
          x2={LOBBY_L}
          y2={COURT_H}
          stroke={stroke2}
          strokeWidth="0.04"
          opacity="0.8"
        />
        <line
          x1={LOBBY_R}
          y1="0"
          x2={LOBBY_R}
          y2={COURT_H}
          stroke={stroke2}
          strokeWidth="0.04"
          opacity="0.8"
        />

        {/* Labels */}
        <text
          x="0.35"
          y={COURT_H - BAULK_Y - 0.12}
          fontSize="0.35"
          fill={faint}
        >
          baulk
        </text>
        <text
          x="0.35"
          y={COURT_H - BONUS_Y - 0.12}
          fontSize="0.35"
          fill={faint}
        >
          bonus
        </text>

        {/* Players */}
        {items.map((p) => {
          const isRaider =
            raiderId != null && Number(p.id) === Number(raiderId)
          const x = clamp(p.x, 0, COURT_W)
          const y = clamp(p.y, 0, COURT_H)
          // SVG origin is top-left; backend y=0 is near middle line, but our court coords
          // treat y increasing towards end line. Flip for display.
          const fy = COURT_H - y
          const baseFill = isRaider
            ? isDark
              ? '#eab308' // amber-500
              : '#a16207' // amber-700
            : isDark
              ? '#e2e8f0' // slate-200
              : '#0f172a' // slate-900
          const fill = p.visible
            ? baseFill
            : isDark
              ? 'rgba(226,232,240,0.25)'
              : 'rgba(15,23,42,0.20)'
          const ring = isRaider
            ? isDark
              ? '#f59e0b'
              : '#b45309'
            : stroke
          return (
            <g key={String(p.id)}>
              <circle
                cx={x}
                cy={fy}
                r={isRaider ? 0.18 : 0.16}
                fill={fill}
                stroke={ring}
                strokeWidth="0.04"
              />
              <text
                x={x + 0.22}
                y={fy + 0.10}
                fontSize="0.32"
                fill={text}
                opacity={p.visible ? 0.9 : 0.55}
              >
                {isRaider ? 'R' : fmtId(p.id)}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
