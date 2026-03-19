import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { clone as skeletonClone } from 'three/addons/utils/SkeletonUtils.js'

const DEFAULT_KEYPOINT_NAMES = [
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

const DEFAULT_SKELETON_EDGES = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
]

let mannequinAssetPromise = null

function loadMannequinAsset() {
  if (!mannequinAssetPromise) {
    mannequinAssetPromise = new Promise((resolve, reject) => {
      const loader = new GLTFLoader()
      loader.load(
        '/man.glb',
        (gltf) => resolve(gltf),
        undefined,
        (err) => reject(err),
      )
    })
  }
  return mannequinAssetPromise
}

function _asNum(v, fb = null) {
  const n = Number(v)
  return Number.isFinite(n) ? n : fb
}

function _asInt(v, fb = null) {
  const n = Number(v)
  return Number.isFinite(n) ? Math.trunc(n) : fb
}

function _clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n))
}

function buildCourtLines(meta, color) {
  const pts = []
  const pushSeg = (x1, z1, x2, z2) => {
    pts.push(x1, 0.002, z1, x2, 0.002, z2)
  }

  pushSeg(0, 0, meta.courtW, 0)
  pushSeg(meta.courtW, 0, meta.courtW, meta.courtH)
  pushSeg(meta.courtW, meta.courtH, 0, meta.courtH)
  pushSeg(0, meta.courtH, 0, 0)
  pushSeg(0, meta.courtH - meta.baulkY, meta.courtW, meta.courtH - meta.baulkY)
  pushSeg(0, meta.courtH - meta.bonusY, meta.courtW, meta.courtH - meta.bonusY)
  pushSeg(meta.lobbyLeftX, 0, meta.lobbyLeftX, meta.courtH)
  pushSeg(meta.lobbyRightX, 0, meta.lobbyRightX, meta.courtH)

  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 })
  const lines = new THREE.LineSegments(geom, mat)
  lines.frustumCulled = false
  return { lines, geom, mat }
}

function makeTextSprite(label, theme = 'dark') {
  const canvas = document.createElement('canvas')
  canvas.width = 256
  canvas.height = 64
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.font = '600 28px Georgia, serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillStyle = theme === 'dark' ? 'rgba(226,232,240,0.92)' : 'rgba(51,65,85,0.92)'
  ctx.fillText(label, canvas.width / 2, canvas.height / 2)

  const texture = new THREE.CanvasTexture(canvas)
  texture.needsUpdate = true
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  })
  const sprite = new THREE.Sprite(material)
  sprite.scale.set(1.55, 0.38, 1)
  return { sprite, texture, material }
}

function keypointMap(playerPose, keypointNames) {
  const map = new Map()
  const points = Array.isArray(playerPose?.keypoints) ? playerPose.keypoints : []
  for (let i = 0; i < points.length; i++) {
    const kp = points[i]
    const name = kp?.name || keypointNames[i] || `kp_${i}`
    const x = _asNum(kp?.x)
    const y = _asNum(kp?.y)
    const confidence = _asNum(kp?.confidence, 1)
    if (x == null || y == null) continue
    map.set(name, { x, y, confidence })
  }
  return map
}

function midpoint(a, b) {
  if (!a || !b) return null
  const out = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
  if (Number.isFinite(a.z) && Number.isFinite(b.z)) out.z = (a.z + b.z) / 2
  return out
}

function estimateScaleMeters(poseMap) {
  const ys = Array.from(poseMap.values())
    .map((kp) => _asNum(kp?.y))
    .filter((v) => v != null)
  if (ys.length < 3) return 0.0034
  const span = Math.max(1, Math.max(...ys) - Math.min(...ys))
  return 1.68 / span
}

function participantIdsForReplay(matWindow, poseWindow, event) {
  const ids = new Set()
  const mid = Array.isArray(matWindow) && matWindow.length ? matWindow[Math.floor(matWindow.length / 2)] : null
  const rid = _asInt(mid?.raider_id)
  if (rid != null) ids.add(rid)

  const sub = _asInt(event?.subject ?? event?.S)
  const obj = _asInt(event?.object ?? event?.O)
  if (sub != null) ids.add(sub)
  if (obj != null) ids.add(obj)

  const poseMid =
    Array.isArray(poseWindow) && poseWindow.length
      ? poseWindow[Math.floor(poseWindow.length / 2)]
      : null
  const posePlayers = Array.isArray(poseMid?.players) ? poseMid.players : []
  for (const player of posePlayers) {
    const pid = _asInt(player?.id)
    if (pid == null) continue
    ids.add(pid)
    if (ids.size >= 3) break
  }

  const matPlayers = Array.isArray(mid?.players) ? mid.players : []
  if (ids.size < 3) {
    for (const player of matPlayers) {
      const pid = _asInt(player?.id)
      if (pid == null) continue
      ids.add(pid)
      if (ids.size >= 3) break
    }
  }

  return Array.from(ids).slice(0, 3)
}

function findBoneByPattern(root, patterns) {
  const list = []
  root.traverse((obj) => {
    if (obj?.isBone) list.push(obj)
  })
  for (const pattern of patterns) {
    const found = list.find((bone) => pattern.test(String(bone.name || '')))
    if (found) return found
  }
  return null
}

function buildBoneMap(root) {
  return {
    hips: findBoneByPattern(root, [/hips?/i, /pelvis/i, /root/i]),
    spine: findBoneByPattern(root, [/spine/i, /spine1/i, /spine_01/i]),
    chest: findBoneByPattern(root, [/chest/i, /spine2/i, /spine_02/i, /upperchest/i]),
    neck: findBoneByPattern(root, [/neck/i]),
    head: findBoneByPattern(root, [/head/i]),
    leftUpperArm: findBoneByPattern(root, [/leftarm/i, /left_upperarm/i, /mixamorigleftarm/i, /l.*upper.*arm/i]),
    leftLowerArm: findBoneByPattern(root, [/leftforearm/i, /left_lowerarm/i, /mixamorigleftforearm/i, /l.*lower.*arm/i]),
    leftHand: findBoneByPattern(root, [/lefthand/i, /mixamoriglefthand/i, /l.*hand/i]),
    rightUpperArm: findBoneByPattern(root, [/rightarm/i, /right_upperarm/i, /mixamorigrightarm/i, /r.*upper.*arm/i]),
    rightLowerArm: findBoneByPattern(root, [/rightforearm/i, /right_lowerarm/i, /mixamorigrightforearm/i, /r.*lower.*arm/i]),
    rightHand: findBoneByPattern(root, [/righthand/i, /mixamorigrighthand/i, /r.*hand/i]),
    leftUpperLeg: findBoneByPattern(root, [/leftupleg/i, /leftthigh/i, /mixamorigleftupleg/i, /l.*upper.*leg/i, /l.*thigh/i]),
    leftLowerLeg: findBoneByPattern(root, [/leftleg/i, /leftcalf/i, /mixamorigleftleg/i, /l.*lower.*leg/i, /l.*calf/i]),
    leftFoot: findBoneByPattern(root, [/leftfoot/i, /mixamorigleftfoot/i, /l.*foot/i]),
    rightUpperLeg: findBoneByPattern(root, [/rightupleg/i, /rightthigh/i, /mixamorigrightupleg/i, /r.*upper.*leg/i, /r.*thigh/i]),
    rightLowerLeg: findBoneByPattern(root, [/rightleg/i, /rightcalf/i, /mixamorigrightleg/i, /r.*lower.*leg/i, /r.*calf/i]),
    rightFoot: findBoneByPattern(root, [/rightfoot/i, /mixamorigrightfoot/i, /r.*foot/i]),
  }
}

function applyBoneDirection(bone, from, to) {
  if (!bone || !from || !to) return
  const dir = new THREE.Vector3().subVectors(to, from)
  if (dir.lengthSq() < 1e-8) return
  dir.normalize()
  const parentQuat = bone.parent?.getWorldQuaternion(new THREE.Quaternion()) || new THREE.Quaternion()
  const targetQuat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir)
  parentQuat.invert()
  bone.quaternion.copy(parentQuat.multiply(targetQuat))
}

function tintModel(root, color, isDark) {
  root.traverse((obj) => {
    if (!obj?.isMesh || !obj.material) return
    obj.frustumCulled = false
    obj.castShadow = false
    obj.receiveShadow = false
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material]
    mats.forEach((mat) => {
      if (mat?.color) mat.color.copy(color)
      if (mat?.emissive) mat.emissive.copy(color).multiplyScalar(isDark ? 0.05 : 0.02)
      if ('transparent' in mat) mat.transparent = true
      if ('opacity' in mat) mat.opacity = 0.92
    })
  })
}

function normalizeModelPlacement(root) {
  const box = new THREE.Box3().setFromObject(root)
  if (!box.isEmpty()) {
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    root.position.x -= center.x
    root.position.z -= center.z
    root.position.y -= box.min.y
    const maxDim = Math.max(size.x || 1, size.y || 1, size.z || 1)
    const targetHeight = 1.34
    const scale = targetHeight / Math.max(0.01, size.y || maxDim)
    root.scale.setScalar(scale)
    root.userData.modelHeight = targetHeight
    root.userData.baseOffsetY = root.position.y
  }
}

function lerpVector3(target, next, alpha = 0.22) {
  if (!target || !next) return
  target.lerp(next, alpha)
}

function slerpQuaternion(target, next, alpha = 0.18) {
  if (!target || !next) return
  target.slerp(next, alpha)
}

function currentIndex(videoEl, fallbackMs, frameCount, fps) {
  if (!frameCount) return 0
  if (videoEl && videoEl.readyState >= 2 && Number.isFinite(videoEl.currentTime)) {
    return Math.max(0, Math.min(frameCount - 1, Math.floor(videoEl.currentTime * fps) % frameCount))
  }
  return Math.max(0, Math.min(frameCount - 1, Math.floor((fallbackMs / 1000) * fps) % frameCount))
}

export default function RaidReplay3D({
  matWindow,
  poseWindow,
  poseMeta,
  event,
  courtMeta,
  videoSrc,
  videoFileSrc,
  theme = 'dark',
}) {
  const mountRef = useRef(null)
  const videoElRef = useRef(null)
  const imgElRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const [previewVideoOk, setPreviewVideoOk] = useState(true)
  const [stats, setStats] = useState({ detected: 0, matched: 0 })
  const [modelReady, setModelReady] = useState(false)
  const [modelError, setModelError] = useState(null)

  const isDark = theme === 'dark'
  const keypointNames = Array.isArray(poseMeta?.keypoint_names) && poseMeta.keypoint_names.length
    ? poseMeta.keypoint_names
    : DEFAULT_KEYPOINT_NAMES
  const skeletonEdges = Array.isArray(poseMeta?.skeleton_edges) && poseMeta.skeleton_edges.length
    ? poseMeta.skeleton_edges
    : DEFAULT_SKELETON_EDGES

  const meta = useMemo(() => {
    const m = courtMeta && typeof courtMeta === 'object' ? courtMeta : {}
    return {
      courtW: _asNum(m.court_w, 10.0) || 10.0,
      courtH: _asNum(m.court_h, 6.5) || 6.5,
      baulkY: _asNum(m.baulk_y, 3.75) || 3.75,
      bonusY: _asNum(m.bonus_y, 4.75) || 4.75,
      lobbyLeftX: _asNum(m.lobby_left_x, 0.75) || 0.75,
      lobbyRightX: _asNum(m.lobby_right_x, 9.25) || 9.25,
      camera: String(m.camera ?? ''),
    }
  }, [courtMeta])

  const participants = useMemo(
    () => participantIdsForReplay(matWindow, poseWindow, event),
    [matWindow, poseWindow, event],
  )

  useEffect(() => {
    const canvas = overlayCanvasRef.current
    const videoEl = videoElRef.current
    const imgEl = imgElRef.current
    if (!canvas) return

    let raf = 0
    const draw = () => {
      const ctx = canvas.getContext('2d')
      const cw = canvas.clientWidth || 1
      const ch = canvas.clientHeight || 1
      const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1))
      if (canvas.width !== Math.floor(cw * dpr) || canvas.height !== Math.floor(ch * dpr)) {
        canvas.width = Math.floor(cw * dpr)
        canvas.height = Math.floor(ch * dpr)
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.save()
      ctx.scale(dpr, dpr)

      const media =
        videoEl && previewVideoOk && videoEl.readyState >= 2 && videoEl.videoWidth > 0 && videoEl.videoHeight > 0
          ? videoEl
          : imgEl
      const mediaW = media?.videoWidth || media?.naturalWidth || 0
      const mediaH = media?.videoHeight || media?.naturalHeight || 0
      if (mediaW > 0 && mediaH > 0 && Array.isArray(poseWindow) && poseWindow.length) {
        const s = Math.min(cw / mediaW, ch / mediaH)
        const w = mediaW * s
        const h = mediaH * s
        const ox = (cw - w) / 2
        const oy = (ch - h) / 2
        const idx = currentIndex(videoEl, 0, poseWindow.length, 30)
        const poseFrame = poseWindow[idx] || {}
        const players = Array.isArray(poseFrame.players) ? poseFrame.players : []
        setStats({
          detected: Array.isArray(poseFrame.detections) ? poseFrame.detections.length : players.length,
          matched: players.length,
        })

        players.slice(0, 3).forEach((player, playerIdx) => {
          const stroke =
            playerIdx === 0
              ? isDark
                ? 'rgba(245, 222, 179, 0.95)'
                : 'rgba(120, 92, 40, 0.95)'
              : playerIdx === 1
                ? isDark
                  ? 'rgba(226, 232, 240, 0.94)'
                  : 'rgba(31, 41, 55, 0.94)'
                : isDark
                  ? 'rgba(148, 163, 184, 0.92)'
                  : 'rgba(71, 85, 105, 0.92)'
          const map = keypointMap(player, keypointNames)
          ctx.lineWidth = 3
          ctx.strokeStyle = stroke
          for (const [aIdx, bIdx] of skeletonEdges) {
            const a = map.get(keypointNames[aIdx])
            const b = map.get(keypointNames[bIdx])
            if (!a || !b) continue
            ctx.beginPath()
            ctx.moveTo(ox + a.x * s, oy + a.y * s)
            ctx.lineTo(ox + b.x * s, oy + b.y * s)
            ctx.stroke()
          }
          for (const kp of map.values()) {
            ctx.beginPath()
            ctx.fillStyle = stroke
            ctx.arc(ox + kp.x * s, oy + kp.y * s, 3.2, 0, Math.PI * 2)
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
  }, [poseWindow, keypointNames, skeletonEdges, isDark, previewVideoOk])

  useEffect(() => {
    const el = mountRef.current
    if (!el || !Array.isArray(matWindow) || matWindow.length === 0) return

    const scene = new THREE.Scene()
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))
    renderer.setClearColor(0x000000, 0)
    el.appendChild(renderer.domElement)

    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 200)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 2.5
    controls.maxDistance = 20
    controls.maxPolarAngle = Math.PI * 0.49

    const ambient = new THREE.AmbientLight(0xffffff, isDark ? 0.3 : 0.38)
    scene.add(ambient)
    const key = new THREE.DirectionalLight(0xffffff, 1.1)
    key.position.set(6, 10, 4)
    scene.add(key)
    const fill = new THREE.DirectionalLight(0xffffff, 0.35)
    fill.position.set(-4, 6, -3)
    scene.add(fill)

    const groundGeom = new THREE.PlaneGeometry(meta.courtW + 2, meta.courtH + 2)
    const groundMat = new THREE.MeshStandardMaterial({
      color: isDark ? 0x0b1220 : 0xf8fafc,
      roughness: 0.96,
      metalness: 0,
      transparent: true,
      opacity: isDark ? 0.55 : 0.82,
    })
    const ground = new THREE.Mesh(groundGeom, groundMat)
    ground.rotation.x = -Math.PI / 2
    scene.add(ground)

    const { lines: courtLines, geom: linesGeom, mat: linesMat } = buildCourtLines(
      meta,
      new THREE.Color(isDark ? 0xe2e8f0 : 0x334155),
    )
    courtLines.position.set(-meta.courtW / 2, 0, -meta.courtH / 2)
    scene.add(courtLines)

    const courtLabelAssets = []
    const addCourtLabel = (text, x, z) => {
      const asset = makeTextSprite(text, isDark ? 'dark' : 'light')
      asset.sprite.position.set(x, 0.08, z)
      scene.add(asset.sprite)
      courtLabelAssets.push(asset)
    }
    addCourtLabel('Bonus', -meta.courtW * 0.22, (meta.courtH - meta.bonusY) - meta.courtH / 2 + 0.16)
    addCourtLabel('Baulk', -meta.courtW * 0.22, (meta.courtH - meta.baulkY) - meta.courtH / 2 + 0.16)
    addCourtLabel('Lobby', meta.lobbyLeftX - meta.courtW / 2 + 0.12, meta.courtH * 0.38 - meta.courtH / 2)
    addCourtLabel('Lobby', meta.lobbyRightX - meta.courtW / 2 - 0.12, meta.courtH * 0.38 - meta.courtH / 2)

    const posFromCourt = (courtPos) => {
      const x = _asNum(courtPos?.[0])
      const y = _asNum(courtPos?.[1])
      if (x == null || y == null) return null
      const cx = _clamp(x, 0, meta.courtW)
      const cy = _clamp(y, 0, meta.courtH)
      return new THREE.Vector3(cx - meta.courtW / 2, 0, (meta.courtH - cy) - meta.courtH / 2)
    }

    const mid = matWindow[Math.floor(matWindow.length / 2)] || {}
    const midPlayers = Array.isArray(mid.players) ? mid.players : []
    const focusTargets = participants
      .map((pid) => {
        const match = midPlayers.find((player) => Number(player?.id) === Number(pid))
        return posFromCourt(match?.court_pos)
      })
      .filter(Boolean)
    const focus =
      focusTargets.length > 0
        ? focusTargets.reduce((acc, v) => acc.add(v), new THREE.Vector3()).multiplyScalar(1 / focusTargets.length)
        : new THREE.Vector3(0, 0, 0)
    controls.target.copy(focus)
    const camDir =
      String(meta.camera).toLowerCase().includes('cam2')
        ? new THREE.Vector3(-0.85, 0, 1)
        : new THREE.Vector3(0.85, 0, 1)
    camDir.normalize()
    camera.position.set(focus.x + camDir.x * 8.5, 5.6, focus.z + camDir.z * 8.5)
    camera.lookAt(focus)
    controls.update()

    const tmpMid = new THREE.Vector3()
    const tmpDir = new THREE.Vector3()
    const jointGeom = new THREE.SphereGeometry(0.045, 10, 10)
    const rigGroup = new THREE.Group()
    scene.add(rigGroup)
    const rigs = new Map()
    const mannequinGroup = new THREE.Group()
    scene.add(mannequinGroup)

    const rigColor = (idx) =>
      idx === 0
        ? new THREE.Color(isDark ? 0xf5deb3 : 0x8b6b2f)
        : idx === 1
          ? new THREE.Color(isDark ? 0xe2e8f0 : 0x1f2937)
          : new THREE.Color(isDark ? 0x94a3b8 : 0x475569)

    for (let i = 0; i < participants.length; i++) {
      const pid = Number(participants[i])
      const color = rigColor(i)
      const lineMaterial = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.96 })
      const lineGeometry = new THREE.BufferGeometry()
      lineGeometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(new Array(skeletonEdges.length * 2 * 3).fill(0), 3),
      )
      const lineSegments = new THREE.LineSegments(lineGeometry, lineMaterial)
      lineSegments.visible = false

      const joints = new Map()
      const jointMaterial = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.35,
        metalness: 0.08,
        emissive: color.clone().multiplyScalar(isDark ? 0.12 : 0.04),
      })
      for (const name of keypointNames) {
        const joint = new THREE.Mesh(jointGeom, jointMaterial)
        joint.visible = false
        rigGroup.add(joint)
        joints.set(name, joint)
      }
      rigGroup.add(lineSegments)
      rigs.set(pid, { lineSegments, lineGeometry, lineMaterial, joints, jointMaterial, mannequin: null, boneMap: null })
    }

    let alive = true
    loadMannequinAsset()
      .then((gltf) => {
        if (!alive) return
        participants.forEach((pid, idx) => {
          const rig = rigs.get(Number(pid))
          if (!rig || rig.mannequin) return
          const root = skeletonClone(gltf.scene)
          const color = rigColor(idx)
          tintModel(root, color, isDark)
          normalizeModelPlacement(root)
          const boneMap = buildBoneMap(root)
          if (boneMap.hips) {
            const hipDrop = Math.max(0, Number(boneMap.hips.position.y || 0)) * Number(root.scale.y || 1) * 0.92
            root.position.y -= hipDrop
            root.userData.hipDrop = hipDrop
          }
          const groundedBox = new THREE.Box3().setFromObject(root)
          if (!groundedBox.isEmpty()) {
            root.position.y -= groundedBox.min.y
          }
          const anchor = new THREE.Group()
          anchor.position.set(0, 0, 0)
          anchor.add(root)
          mannequinGroup.add(anchor)
          rig.mannequin = anchor
          rig.boneMap = boneMap
        })
        setModelReady(true)
        setModelError(null)
      })
      .catch((err) => {
        if (!alive) return
        setModelReady(false)
        setModelError(String(err?.message || err))
      })

    let held = false
    let fallbackMs = 0
    let lastMs = performance.now()
    const videoEl = videoElRef.current

    const onStart = () => {
      held = true
      try {
        if (videoEl && previewVideoOk) videoEl.pause()
      } catch {
        // ignore
      }
    }
    const onEnd = () => {
      held = false
      lastMs = performance.now()
      try {
        if (videoEl && previewVideoOk && typeof videoEl.play === 'function') {
          const playPromise = videoEl.play()
          if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(() => {
              setPreviewVideoOk(false)
            })
          }
        }
      } catch {
        setPreviewVideoOk(false)
      }
    }
    controls.addEventListener('start', onStart)
    controls.addEventListener('end', onEnd)

    let raf = 0
    const animate = () => {
      const now = performance.now()
      if (!held) fallbackMs += now - lastMs
      lastMs = now

      const idx = currentIndex(videoEl, fallbackMs, matWindow.length, 30)
      const matFrame = matWindow[idx] || {}
      const poseFrame = Array.isArray(poseWindow) ? poseWindow[idx] || {} : {}
      const posePlayers = Array.isArray(poseFrame.players) ? poseFrame.players : []
      const poseById = new Map(posePlayers.map((player) => [Number(player?.id), player]))
      const matPlayers = Array.isArray(matFrame.players) ? matFrame.players : []
      const matById = new Map(matPlayers.map((player) => [Number(player?.id), player]))

      for (const pid of participants) {
        const rig = rigs.get(Number(pid))
        if (!rig) continue

        const posePlayer = poseById.get(Number(pid))
        const tracked = matById.get(Number(pid))
        const root = posFromCourt(posePlayer?.court_pos || tracked?.court_pos)
        const positions = rig.lineGeometry.attributes.position.array

        if (!posePlayer || !root) {
          rig.lineSegments.visible = false
          rig.joints.forEach((joint) => {
            joint.visible = false
          })
          if (rig.mannequin) rig.mannequin.visible = false
          continue
        }

        const map = keypointMap(posePlayer, keypointNames)
        const leftAnkle = map.get('left_ankle')
        const rightAnkle = map.get('right_ankle')
        const leftHip = map.get('left_hip')
        const rightHip = map.get('right_hip')
        const ankleMid = midpoint(leftAnkle, rightAnkle)
        const hipMid = midpoint(leftHip, rightHip)
        const anchor = ankleMid || hipMid
        if (!anchor) {
          rig.lineSegments.visible = false
          rig.joints.forEach((joint) => {
            joint.visible = false
          })
          if (rig.mannequin) rig.mannequin.visible = false
          continue
        }

        const scale = estimateScaleMeters(map)
        const to3 = (kp) => {
          if (!kp) return null
          const dx = (kp.x - anchor.x) * scale * 0.78
          const dy = (anchor.y - kp.y) * scale
          const dz = (kp.x - anchor.x) * scale * 0.14
          return new THREE.Vector3(root.x + dx, Math.max(0.03, dy), root.z + dz)
        }

        let visibleEdges = 0
        let cursor = 0
        for (const [aIdx, bIdx] of skeletonEdges) {
          const a = to3(map.get(keypointNames[aIdx]))
          const b = to3(map.get(keypointNames[bIdx]))
          if (a && b) {
            positions[cursor++] = a.x
            positions[cursor++] = a.y
            positions[cursor++] = a.z
            positions[cursor++] = b.x
            positions[cursor++] = b.y
            positions[cursor++] = b.z
            visibleEdges += 1
          }
        }
        rig.lineGeometry.setDrawRange(0, visibleEdges * 2)
        rig.lineGeometry.attributes.position.needsUpdate = true
        rig.lineSegments.visible = !rig.mannequin && visibleEdges > 0

        rig.joints.forEach((joint, name) => {
          const kp = to3(map.get(name))
          if (!kp) {
            joint.visible = false
            return
          }
          joint.visible = !rig.mannequin
          joint.position.copy(kp)
        })

        const nose = to3(map.get('nose'))
        const neck = midpoint(to3(map.get('left_shoulder')), to3(map.get('right_shoulder')))
        if (nose && neck) {
          tmpMid.copy(nose).add(neck).multiplyScalar(0.5)
          tmpDir.subVectors(neck, nose)
        }

        if (rig.mannequin && rig.boneMap) {
          rig.mannequin.visible = true
          const leftShoulder3 = to3(map.get('left_shoulder'))
          const rightShoulder3 = to3(map.get('right_shoulder'))
          const leftElbow3 = to3(map.get('left_elbow'))
          const rightElbow3 = to3(map.get('right_elbow'))
          const leftWrist3 = to3(map.get('left_wrist'))
          const rightWrist3 = to3(map.get('right_wrist'))
          const leftKnee3 = to3(map.get('left_knee'))
          const rightKnee3 = to3(map.get('right_knee'))
          const leftAnkle3 = to3(map.get('left_ankle'))
          const rightAnkle3 = to3(map.get('right_ankle'))
          const hip3 = to3(hipMid || midpoint(leftHip, rightHip))
          if (hip3) {
            const desiredPos = new THREE.Vector3(root.x, 0, root.z)
            lerpVector3(rig.mannequin.position, desiredPos, 0.24)
            const shoulderSpan = midpoint(leftShoulder3, rightShoulder3)
            const hipSpan = midpoint(to3(map.get('left_hip')), to3(map.get('right_hip')))
            const facingBase = shoulderSpan || hipSpan
            if (facingBase && leftShoulder3 && rightShoulder3) {
              const across = new THREE.Vector3().subVectors(rightShoulder3, leftShoulder3)
              const forward = new THREE.Vector3(across.z, 0, -across.x).normalize()
              if (forward.lengthSq() > 1e-6) {
                const yaw = Math.atan2(forward.x, forward.z)
                const targetQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw)
                slerpQuaternion(rig.mannequin.quaternion, targetQuat, 0.18)
              }
            }
            if (rig.boneMap.hips) {
              rig.boneMap.hips.position.x = 0
              rig.boneMap.hips.position.z = 0
            }
          }

          const chest3 = midpoint(leftShoulder3, rightShoulder3)

          applyBoneDirection(rig.boneMap.spine, hip3, chest3)
          applyBoneDirection(rig.boneMap.chest, chest3, neck || chest3)
          applyBoneDirection(rig.boneMap.neck, chest3, neck || chest3)

          applyBoneDirection(rig.boneMap.leftUpperArm, leftShoulder3, leftElbow3)
          applyBoneDirection(rig.boneMap.leftLowerArm, leftElbow3, leftWrist3)
          applyBoneDirection(rig.boneMap.leftHand, leftElbow3, leftWrist3)
          applyBoneDirection(rig.boneMap.rightUpperArm, rightShoulder3, rightElbow3)
          applyBoneDirection(rig.boneMap.rightLowerArm, rightElbow3, rightWrist3)
          applyBoneDirection(rig.boneMap.rightHand, rightElbow3, rightWrist3)

          applyBoneDirection(rig.boneMap.leftUpperLeg, hip3, leftKnee3)
          applyBoneDirection(rig.boneMap.leftLowerLeg, leftKnee3, leftAnkle3)
          applyBoneDirection(rig.boneMap.leftFoot, leftKnee3, leftAnkle3)
          applyBoneDirection(rig.boneMap.rightUpperLeg, hip3, rightKnee3)
          applyBoneDirection(rig.boneMap.rightLowerLeg, rightKnee3, rightAnkle3)
          applyBoneDirection(rig.boneMap.rightFoot, rightKnee3, rightAnkle3)
          if (rig.boneMap.head) {
            rig.boneMap.head.rotation.x *= 0.7
            rig.boneMap.head.rotation.y *= 0.7
            rig.boneMap.head.rotation.z *= 0.7
          }
        }
      }

      controls.update()
      renderer.render(scene, camera)
      raf = requestAnimationFrame(animate)
    }

    const resize = () => {
      const w = el.clientWidth || 1
      const h = el.clientHeight || 1
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()
    raf = requestAnimationFrame(animate)

    return () => {
      alive = false
      if (raf) cancelAnimationFrame(raf)
      try {
        controls.removeEventListener('start', onStart)
        controls.removeEventListener('end', onEnd)
      } catch {
        // ignore
      }
      try {
        controls.dispose()
      } catch {
        // ignore
      }
      try {
        ro.disconnect()
      } catch {
        // ignore
      }
      rigs.forEach((rig) => {
        try {
          rig.lineGeometry.dispose()
        } catch {
          // ignore
        }
        try {
          rig.lineMaterial.dispose()
        } catch {
          // ignore
        }
        try {
          rig.jointMaterial.dispose()
        } catch {
          // ignore
        }
      })
      try {
        jointGeom.dispose()
      } catch {
        // ignore
      }
      mannequinGroup.traverse((obj) => {
        if (obj?.isMesh) {
          try {
            obj.geometry?.dispose?.()
          } catch {
            // ignore
          }
        }
      })
      try {
        linesGeom.dispose()
      } catch {
        // ignore
      }
      try {
        linesMat.dispose()
      } catch {
        // ignore
      }
      try {
        groundGeom.dispose()
      } catch {
        // ignore
      }
      try {
        groundMat.dispose()
      } catch {
        // ignore
      }
      for (const asset of courtLabelAssets) {
        try {
          scene.remove(asset.sprite)
        } catch {
          // ignore
        }
        try {
          asset.texture.dispose()
        } catch {
          // ignore
        }
        try {
          asset.material.dispose()
        } catch {
          // ignore
        }
      }
      try {
        renderer.dispose()
      } catch {
        // ignore
      }
      try {
        renderer.domElement?.remove()
      } catch {
        // ignore
      }
    }
  }, [matWindow, poseWindow, participants, meta, keypointNames, skeletonEdges, isDark])

  const hasPoseWindow = Array.isArray(poseWindow) && poseWindow.length > 0

  return (
    <div className="relative h-full w-full overflow-hidden">
      <div
        ref={mountRef}
        className="absolute inset-0 z-10"
        style={{
          background: isDark
            ? 'radial-gradient(1000px 540px at 28% 20%, rgba(148,163,184,0.10), transparent 60%), radial-gradient(900px 520px at 72% 74%, rgba(148,163,184,0.08), transparent 62%)'
            : 'radial-gradient(1000px 540px at 28% 20%, rgba(15,23,42,0.06), transparent 60%), radial-gradient(900px 520px at 72% 74%, rgba(15,23,42,0.05), transparent 62%)',
        }}
      />

      <div className="pointer-events-none absolute left-3 top-3 z-20 rounded-full border border-white/10 bg-black/35 px-3 py-1 text-[11px] font-medium text-white backdrop-blur">
        3D stick-figure replay from archived YOLO poses
      </div>
      <div className="pointer-events-none absolute left-3 top-12 z-20 rounded-full border border-white/10 bg-black/35 px-3 py-1 text-[11px] font-medium text-white backdrop-blur">
        poses {stats.detected} | matched {stats.matched}
      </div>
      <div className="pointer-events-none absolute left-3 top-[84px] z-20 rounded-full border border-white/10 bg-black/35 px-3 py-1 text-[11px] font-medium text-white backdrop-blur">
        mannequin {modelReady ? 'ready' : modelError ? 'fallback' : 'loading'}
      </div>

      <div
        className="absolute bottom-4 right-4 z-20 overflow-hidden rounded-2xl border border-white/10 bg-black/35 shadow-2xl backdrop-blur"
        style={{
          width: 'min(560px, 40vw)',
          height: 'min(320px, 28vw)',
          minWidth: 360,
          minHeight: 220,
        }}
      >
        {videoFileSrc ? (
          <video
            ref={videoElRef}
            src={videoFileSrc}
            className="absolute inset-0 h-full w-full object-contain"
            autoPlay
            loop
            muted
            playsInline
            preload="auto"
            onError={() => setPreviewVideoOk(false)}
            style={{ display: previewVideoOk ? 'block' : 'none' }}
          />
        ) : null}
        {videoSrc ? (
          <img
            ref={imgElRef}
            src={videoSrc}
            alt="Raid preview"
            className="absolute inset-0 h-full w-full object-contain"
            loading="eager"
            decoding="async"
            style={{ display: previewVideoOk && videoFileSrc ? 'none' : 'block' }}
          />
        ) : null}
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 h-full w-full"
          style={{ pointerEvents: 'none' }}
        />
      </div>

      {!hasPoseWindow ? (
        <div className="absolute bottom-4 left-4 z-20 rounded-xl border border-amber-200 bg-amber-50/90 px-3 py-2 text-xs text-amber-900 shadow-sm dark:border-amber-900/60 dark:bg-amber-950/50 dark:text-amber-100">
          No archived pose window yet. Re-run the backend once so confirmed events are exported with YOLO bones.
        </div>
      ) : null}
      {modelError ? (
        <div className="absolute bottom-16 left-4 z-20 rounded-xl border border-amber-200 bg-amber-50/90 px-3 py-2 text-xs text-amber-900 shadow-sm dark:border-amber-900/60 dark:bg-amber-950/50 dark:text-amber-100">
          Could not load or rig `man.glb`. Showing stick-figure fallback. {modelError}
        </div>
      ) : null}
    </div>
  )
}
