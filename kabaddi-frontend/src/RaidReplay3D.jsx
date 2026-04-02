import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js'

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

const MANNEQUIN_TARGET_HEIGHT = 1.42
const MANNEQUIN_Z_OFFSET = 0
const MANNEQUIN_Y_OFFSET = 0.04
const MANNEQUIN_ASSET_PATH = '/mann.glb'
const MANNEQUIN_ASSET_LABEL = 'mann.glb'
const MANNEQUIN_RENDER_PIXEL_RATIO = 0.85
const SCROLL_FRAME_DELTA_DIVISOR = 36
const SCROLL_FRAME_DELTA_MAX = 8
const RAID_REPLAY_CAMERA_LOCK_KEY = 'kabaddi-raid-replay-camera-lock-v1'
const ANIMATION_ASSET_PATHS = {
  idle: '/Idle.fbx',
  crouchedWalking: '/Crouched Walking.fbx',
  runForward: '/Run Forward.fbx',
  runningBackward: '/Running Backward.fbx',
  changeDirection: '/Change Direction.fbx',
  crouchToStand: '/Crouch To Stand.fbx',
}

let mannequinAssetPromise = null
let animationAssetsPromise = null
let mannequinBoneDebugPrinted = false

function loadMannequinAsset() {
  if (!mannequinAssetPromise) {
    mannequinAssetPromise = new Promise((resolve, reject) => {
      const loader = new GLTFLoader()
      loader.load(
        MANNEQUIN_ASSET_PATH,
        (gltf) => resolve(gltf),
        undefined,
        (err) => reject(err),
      )
    })
  }
  return mannequinAssetPromise
}

function loadAnimationAssets() {
  if (!animationAssetsPromise) {
    animationAssetsPromise = new Promise((resolve, reject) => {
      const loader = new FBXLoader()
      const entries = Object.entries(ANIMATION_ASSET_PATHS)
      Promise.all(
        entries.map(
          ([key, path]) =>
            new Promise((res, rej) => {
              loader.load(
                path,
                (fbx) => {
                  const clip = Array.isArray(fbx.animations) ? fbx.animations[0] : null
                  if (!clip) {
                    rej(new Error(`No animation clip found in ${path}`))
                    return
                  }
                  clip.name = key
                  res([key, { root: fbx, clip }])
                },
                undefined,
                (err) => rej(err),
              )
            }),
        ),
      )
        .then((loaded) => resolve(new Map(loaded)))
        .catch(reject)
    })
  }
  return animationAssetsPromise
}

function listBoneNames(root) {
  const names = []
  root.traverse((obj) => {
    if (obj?.isBone) names.push(String(obj.name || ''))
  })
  return names
}

function buildRetargetNameMap(boneMap) {
  const map = {}
  const assign = (targetBone, sourceName) => {
    if (targetBone?.name) map[targetBone.name] = sourceName
  }
  assign(boneMap?.hips, 'mixamorigHips')
  assign(boneMap?.spine, 'mixamorigSpine')
  assign(boneMap?.spineMid, 'mixamorigSpine1')
  assign(boneMap?.chest, 'mixamorigSpine2')
  assign(boneMap?.neck, 'mixamorigNeck')
  assign(boneMap?.head, 'mixamorigHead')
  assign(boneMap?.leftShoulder, 'mixamorigLeftShoulder')
  assign(boneMap?.leftUpperArm, 'mixamorigLeftArm')
  assign(boneMap?.leftLowerArm, 'mixamorigLeftForeArm')
  assign(boneMap?.leftHand, 'mixamorigLeftHand')
  assign(boneMap?.rightShoulder, 'mixamorigRightShoulder')
  assign(boneMap?.rightUpperArm, 'mixamorigRightArm')
  assign(boneMap?.rightLowerArm, 'mixamorigRightForeArm')
  assign(boneMap?.rightHand, 'mixamorigRightHand')
  assign(boneMap?.leftUpperLeg, 'mixamorigLeftUpLeg')
  assign(boneMap?.leftLowerLeg, 'mixamorigLeftLeg')
  assign(boneMap?.leftFoot, 'mixamorigLeftFoot')
  assign(boneMap?.leftToe, 'mixamorigLeftToeBase')
  assign(boneMap?.rightUpperLeg, 'mixamorigRightUpLeg')
  assign(boneMap?.rightLowerLeg, 'mixamorigRightLeg')
  assign(boneMap?.rightFoot, 'mixamorigRightFoot')
  assign(boneMap?.rightToe, 'mixamorigRightToeBase')
  return map
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

function formatReplaySourceLabel(src) {
  if (!src) return 'Kabaddi Replay'
  try {
    const raw = String(src).split('?')[0]
    const last = raw.split('/').filter(Boolean).pop() || raw
    return decodeURIComponent(last)
  } catch {
    return String(src)
  }
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

function midpoint3(a, b) {
  if (!a || !b) return null
  return a.clone().add(b).multiplyScalar(0.5)
}

function estimateScaleMeters(poseMap) {
  const ys = Array.from(poseMap.values())
    .map((kp) => _asNum(kp?.y))
    .filter((v) => v != null)
  if (ys.length < 3) return 0.0034
  const span = Math.max(1, Math.max(...ys) - Math.min(...ys))
  return 1.68 / span
}

function hasValidCourtPos(player) {
  return (
    Array.isArray(player?.court_pos) &&
    player.court_pos.length >= 2 &&
    Number.isFinite(Number(player.court_pos[0])) &&
    Number.isFinite(Number(player.court_pos[1]))
  )
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

function mixamorigBonePattern(name) {
  return new RegExp(`mixamorig\\d*:?${name}$`, 'i')
}

function buildBoneMap(root) {
  return {
    root: findBoneByPattern(root, [/^_rootjoint$/i, /^wiest$/i, /^root$/i]),
    hips: findBoneByPattern(root, [/^hips?_/i, /^hips?$/i, /^hip$/i, /pelvis/i, /^wiest$/i, /root/i, mixamorigBonePattern('Hips')]),
    spine: findBoneByPattern(root, [/^spine_?\d*/i, /^spine$/i, /^chest$/i, /spine1/i, /spine_01/i, mixamorigBonePattern('Spine')]),
    spineMid: findBoneByPattern(root, [/^spine1_?\d*/i, /spine1/i, /spine_02/i, mixamorigBonePattern('Spine1')]),
    chest: findBoneByPattern(root, [/^spine2_?\d*/i, /^spine1_?\d*/i, /^chest$/i, /spine2/i, /spine_02/i, /upperchest/i, mixamorigBonePattern('Spine2')]),
    neck: findBoneByPattern(root, [/neck/i]),
    head: findBoneByPattern(root, [/head/i]),
    headTop: findBoneByPattern(root, [/headtop/i, /head.*end/i, mixamorigBonePattern('HeadTop_End')]),
    leftShoulder: findBoneByPattern(root, [/^leftshoulder_/i, /^KTFL$/i, /leftshoulder/i, /leftclavicle/i, /l.*shoulder/i, /l.*clav/i, mixamorigBonePattern('LeftShoulder')]),
    leftUpperArm: findBoneByPattern(root, [/^leftarm_/i, /^upperarmL$/i, /leftarm/i, /left_upperarm/i, /mixamorigleftarm/i, /upper.*arm.*l/i, mixamorigBonePattern('LeftArm')]),
    leftLowerArm: findBoneByPattern(root, [/^leftforearm_/i, /^lowerarmL$/i, /leftforearm/i, /left_lowerarm/i, /mixamorigleftforearm/i, /lower.*arm.*l/i, mixamorigBonePattern('LeftForeArm')]),
    leftHand: findBoneByPattern(root, [/^lefthand_/i, /^handL$/i, /lefthand/i, /mixamoriglefthand/i, /hand.*l/i, mixamorigBonePattern('LeftHand')]),
    rightShoulder: findBoneByPattern(root, [/^rightshoulder_/i, /^KTFR$/i, /rightshoulder/i, /rightclavicle/i, /r.*shoulder/i, /r.*clav/i, mixamorigBonePattern('RightShoulder')]),
    rightUpperArm: findBoneByPattern(root, [/^rightarm_/i, /^upperarmR$/i, /rightarm/i, /right_upperarm/i, /mixamorigrightarm/i, /upper.*arm.*r/i, mixamorigBonePattern('RightArm')]),
    rightLowerArm: findBoneByPattern(root, [/^rightforearm_/i, /^lowerarmR$/i, /rightforearm/i, /right_lowerarm/i, /mixamorigrightforearm/i, /lower.*arm.*r/i, mixamorigBonePattern('RightForeArm')]),
    rightHand: findBoneByPattern(root, [/^righthand_/i, /^handR$/i, /righthand/i, /mixamorigrighthand/i, /hand.*r/i, mixamorigBonePattern('RightHand')]),
    leftUpperLeg: findBoneByPattern(root, [/^leftupleg_/i, /^upperlegL$/i, /leftupleg/i, /leftthigh/i, /mixamorigleftupleg/i, /upper.*leg.*l/i, /thigh.*l/i, mixamorigBonePattern('LeftUpLeg')]),
    leftLowerLeg: findBoneByPattern(root, [/^leftleg_/i, /^lowerlegL$/i, /leftleg/i, /leftcalf/i, /mixamorigleftleg/i, /lower.*leg.*l/i, /calf.*l/i, mixamorigBonePattern('LeftLeg')]),
    leftFoot: findBoneByPattern(root, [/^leftfoot_/i, /^footL$/i, /^tooseL$/i, /leftfoot/i, /mixamorigleftfoot/i, /foot.*l/i, mixamorigBonePattern('LeftFoot')]),
    leftToe: findBoneByPattern(root, [/lefttoe/i, /toebase.*l/i, mixamorigBonePattern('LeftToeBase'), mixamorigBonePattern('LeftToe_End')]),
    rightUpperLeg: findBoneByPattern(root, [/^rightupleg_/i, /^upperlegR$/i, /rightupleg/i, /rightthigh/i, /mixamorigrightupleg/i, /upper.*leg.*r/i, /thigh.*r/i, mixamorigBonePattern('RightUpLeg')]),
    rightLowerLeg: findBoneByPattern(root, [/^rightleg_/i, /^lowerlegR$/i, /rightleg/i, /rightcalf/i, /mixamorigrightleg/i, /lower.*leg.*r/i, /calf.*r/i, mixamorigBonePattern('RightLeg')]),
    rightFoot: findBoneByPattern(root, [/^rightfoot_/i, /^footR$/i, /^tooseR$/i, /rightfoot/i, /mixamorigrightfoot/i, /foot.*r/i, mixamorigBonePattern('RightFoot')]),
    rightToe: findBoneByPattern(root, [/righttoe/i, /toebase.*r/i, mixamorigBonePattern('RightToeBase'), mixamorigBonePattern('RightToe_End')]),
  }
}

function captureBoneRestPose(boneMap) {
  const out = {}
  for (const [name, bone] of Object.entries(boneMap || {})) {
    if (!bone) continue
    let aimAxisLocal = new THREE.Vector3(0, 1, 0)
    const childBone = bone.children.find((child) => child?.isBone && child.position.lengthSq() > 1e-8)
    if (childBone) {
      aimAxisLocal = childBone.position.clone().normalize()
    } else if (bone.position.lengthSq() > 1e-8) {
      aimAxisLocal = bone.position.clone().normalize()
    }
    bone.userData.aimAxisLocal = aimAxisLocal.clone()
    out[name] = {
      position: bone.position.clone(),
      quaternion: bone.quaternion.clone(),
      scale: bone.scale.clone(),
      aimAxisLocal: aimAxisLocal.clone(),
    }
  }
  return out
}

function restoreBoneRestPose(boneMap, restPose) {
  for (const [name, bone] of Object.entries(boneMap || {})) {
    if (!bone) continue
    const rest = restPose?.[name]
    if (!rest) continue
    bone.position.copy(rest.position)
    bone.quaternion.copy(rest.quaternion)
    bone.scale.copy(rest.scale)
    bone.userData.aimAxisLocal = rest.aimAxisLocal?.clone?.() || new THREE.Vector3(0, 1, 0)
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

function applyBoneDirectionWeighted(bone, from, to, weight = 1) {
  if (!bone || !from || !to) return
  const dir = new THREE.Vector3().subVectors(to, from)
  if (dir.lengthSq() < 1e-8) return
  dir.normalize()
  const parentQuat = bone.parent?.getWorldQuaternion(new THREE.Quaternion()) || new THREE.Quaternion()
  const aimAxis = bone.userData?.aimAxisLocal?.clone?.() || new THREE.Vector3(0, 1, 0)
  if (aimAxis.lengthSq() < 1e-8) aimAxis.set(0, 1, 0)
  aimAxis.normalize()
  const targetQuat = new THREE.Quaternion().setFromUnitVectors(aimAxis, dir)
  parentQuat.invert()
  const localTarget = parentQuat.multiply(targetQuat)
  bone.quaternion.slerp(localTarget, _clamp(weight, 0, 1))
}

function blendTarget(actual, fallback, fallbackWeight = 0.14) {
  if (actual && fallback) return actual.clone().lerp(fallback, _clamp(fallbackWeight, 0, 1))
  if (actual) return actual.clone()
  if (fallback) return fallback.clone()
  return null
}

function projectAlongBone(from, to, length) {
  if (!from || !to || !Number.isFinite(Number(length))) return null
  const dir = new THREE.Vector3().subVectors(to, from)
  if (dir.lengthSq() < 1e-8) return null
  dir.normalize()
  return to.clone().addScaledVector(dir, length)
}

function buildPoseDrivenTargets({
  posePoints,
  fallbackTargets,
  movementDir,
}) {
  const actualHip = midpoint3(posePoints.leftHip, posePoints.rightHip)
  const actualChest = midpoint3(posePoints.leftShoulder, posePoints.rightShoulder)
  const actualHead =
    posePoints.nose ||
    midpoint3(posePoints.leftEye, posePoints.rightEye) ||
    midpoint3(posePoints.leftEar, posePoints.rightEar)

  const hip = blendTarget(actualHip, fallbackTargets.hip, 0.08)
  const chest = blendTarget(actualChest, fallbackTargets.chest, 0.1)
  const head = blendTarget(actualHead, fallbackTargets.head, 0.2)
  const neck = blendTarget(midpoint3(chest, head), midpoint3(fallbackTargets.chest, fallbackTargets.head), 0.2)
  const spineMid = blendTarget(midpoint3(hip, chest), midpoint3(fallbackTargets.hip, fallbackTargets.chest), 0.14)

  const leftShoulder = blendTarget(posePoints.leftShoulder, fallbackTargets.leftShoulder, 0.08)
  const rightShoulder = blendTarget(posePoints.rightShoulder, fallbackTargets.rightShoulder, 0.08)
  const leftElbow = blendTarget(posePoints.leftElbow, fallbackTargets.leftElbow, 0.1)
  const rightElbow = blendTarget(posePoints.rightElbow, fallbackTargets.rightElbow, 0.1)
  const leftWrist = blendTarget(posePoints.leftWrist, fallbackTargets.leftWrist, 0.12)
  const rightWrist = blendTarget(posePoints.rightWrist, fallbackTargets.rightWrist, 0.12)
  const leftHip = blendTarget(posePoints.leftHip, fallbackTargets.leftHip, 0.08)
  const rightHip = blendTarget(posePoints.rightHip, fallbackTargets.rightHip, 0.08)
  const leftKnee = blendTarget(posePoints.leftKnee, fallbackTargets.leftKnee, 0.1)
  const rightKnee = blendTarget(posePoints.rightKnee, fallbackTargets.rightKnee, 0.1)
  const leftAnkle = blendTarget(posePoints.leftAnkle, fallbackTargets.leftAnkle, 0.1)
  const rightAnkle = blendTarget(posePoints.rightAnkle, fallbackTargets.rightAnkle, 0.1)

  const move = movementDir && movementDir.lengthSq() > 1e-8 ? movementDir.clone().normalize() : fallbackTargets.forward.clone()
  const leftHand = blendTarget(
    projectAlongBone(leftElbow, leftWrist, 0.12),
    projectAlongBone(fallbackTargets.leftElbow, fallbackTargets.leftWrist, 0.12),
    0.25,
  )
  const rightHand = blendTarget(
    projectAlongBone(rightElbow, rightWrist, 0.12),
    projectAlongBone(fallbackTargets.rightElbow, fallbackTargets.rightWrist, 0.12),
    0.25,
  )
  const leftFoot = blendTarget(
    projectAlongBone(leftKnee, leftAnkle, 0.1) || (leftAnkle && move ? leftAnkle.clone().addScaledVector(move, 0.1) : null),
    projectAlongBone(fallbackTargets.leftKnee, fallbackTargets.leftAnkle, 0.1),
    0.22,
  )
  const rightFoot = blendTarget(
    projectAlongBone(rightKnee, rightAnkle, 0.1) || (rightAnkle && move ? rightAnkle.clone().addScaledVector(move, 0.1) : null),
    projectAlongBone(fallbackTargets.rightKnee, fallbackTargets.rightAnkle, 0.1),
    0.22,
  )
  const leftToe = blendTarget(
    leftFoot && move ? leftFoot.clone().addScaledVector(move, 0.08) : null,
    fallbackTargets.leftAnkle && fallbackTargets.forward
      ? fallbackTargets.leftAnkle.clone().addScaledVector(fallbackTargets.forward, 0.16)
      : null,
    0.28,
  )
  const rightToe = blendTarget(
    rightFoot && move ? rightFoot.clone().addScaledVector(move, 0.08) : null,
    fallbackTargets.rightAnkle && fallbackTargets.forward
      ? fallbackTargets.rightAnkle.clone().addScaledVector(fallbackTargets.forward, 0.16)
      : null,
    0.28,
  )
  const headTop = blendTarget(
    head && neck ? projectAlongBone(neck, head, 0.1) : null,
    fallbackTargets.head && fallbackTargets.chest ? projectAlongBone(fallbackTargets.chest, fallbackTargets.head, 0.12) : null,
    0.2,
  )

  return {
    forward: move,
    hip,
    spineMid,
    chest,
    neck,
    head,
    headTop,
    leftShoulder,
    rightShoulder,
    leftElbow,
    rightElbow,
    leftWrist,
    rightWrist,
    leftHand,
    rightHand,
    leftHip,
    rightHip,
    leftKnee,
    rightKnee,
    leftAnkle,
    rightAnkle,
    leftFoot,
    rightFoot,
    leftToe,
    rightToe,
  }
}

function tintModel(root, color, isDark) {
  root.traverse((obj) => {
    if (!obj?.isMesh || !obj.material) return
    obj.visible = true
    obj.frustumCulled = true
    obj.castShadow = false
    obj.receiveShadow = false
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material]
    mats.forEach((mat) => {
      if (mat?.color) mat.color.copy(color)
      if (mat?.emissive) mat.emissive.copy(color).multiplyScalar(isDark ? 0.04 : 0.02)
      if ('transparent' in mat) mat.transparent = false
      if ('opacity' in mat) mat.opacity = 1
      if ('alphaTest' in mat) mat.alphaTest = 0
      if ('depthWrite' in mat) mat.depthWrite = true
      if ('depthTest' in mat) mat.depthTest = true
      if ('toneMapped' in mat) mat.toneMapped = false
      if ('side' in mat) mat.side = THREE.FrontSide
      if ('flatShading' in mat) mat.flatShading = false
      if ('needsUpdate' in mat) mat.needsUpdate = true
    })
  })
}

function normalizeModelPlacement(root) {
  root.updateMatrixWorld(true)
  const box = new THREE.Box3().setFromObject(root)
  if (!box.isEmpty()) {
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    root.position.x -= center.x
    root.position.z -= center.z
    root.position.y -= box.min.y
    const maxDim = Math.max(size.x || 1, size.y || 1, size.z || 1)
    const targetHeight = MANNEQUIN_TARGET_HEIGHT
    const scale = targetHeight / Math.max(0.01, size.y || maxDim)
    root.scale.setScalar(scale)
    root.updateMatrixWorld(true)
    const groundedBox = new THREE.Box3().setFromObject(root)
    if (!groundedBox.isEmpty()) {
      const groundedCenter = groundedBox.getCenter(new THREE.Vector3())
      root.position.x -= groundedCenter.x
      root.position.z -= groundedCenter.z
      root.position.y -= groundedBox.min.y
    }
    root.userData.modelHeight = targetHeight
    root.userData.baseOffsetY = root.position.y
  }
}

function fitReplayCamera(camera, controls, focus, meta, aspect = 1) {
  if (!camera || !controls || !focus || !meta) return
  const safeAspect = Math.max(0.6, Number(aspect) || 1)
  const fov = THREE.MathUtils.degToRad(camera.fov || 55)
  const fitHeight = (meta.courtH || 6.5) * 1.05
  const fitWidth = (meta.courtW || 10) / safeAspect
  const fitSpan = Math.max(fitHeight, fitWidth)
  const distance = Math.max(5.8, fitSpan / (2 * Math.tan(fov / 2)))
  const camDir =
    String(meta.camera).toLowerCase().includes('cam2')
      ? new THREE.Vector3(-0.82, 0.48, 0.92)
      : new THREE.Vector3(0.82, 0.48, 0.92)
  camDir.normalize()
  controls.target.copy(focus)
  camera.position.copy(focus.clone().addScaledVector(camDir, distance * 1.08))
  camera.lookAt(focus)
}

function lerpVector3(target, next, alpha = 0.22) {
  if (!target || !next) return
  target.lerp(next, alpha)
}

function slerpQuaternion(target, next, alpha = 0.18) {
  if (!target || !next) return
  target.slerp(next, alpha)
}

function lerpNumber(a, b, t) {
  const na = Number(a)
  const nb = Number(b)
  if (!Number.isFinite(na) && !Number.isFinite(nb)) return null
  if (!Number.isFinite(na)) return nb
  if (!Number.isFinite(nb)) return na
  return na + (nb - na) * t
}

function interpolateCourtPos(a, b, t) {
  if (!Array.isArray(a) && !Array.isArray(b)) return null
  if (!Array.isArray(a)) return b
  if (!Array.isArray(b)) return a
  return [lerpNumber(a[0], b[0], t), lerpNumber(a[1], b[1], t)]
}

function interpolateKeypoints(aPoints, bPoints, keypointNames, t) {
  const out = []
  const maxLen = Math.max(
    Array.isArray(aPoints) ? aPoints.length : 0,
    Array.isArray(bPoints) ? bPoints.length : 0,
    keypointNames.length,
  )
  for (let i = 0; i < maxLen; i++) {
    const a = Array.isArray(aPoints) ? aPoints[i] : null
    const b = Array.isArray(bPoints) ? bPoints[i] : null
    const x = lerpNumber(a?.x, b?.x, t)
    const y = lerpNumber(a?.y, b?.y, t)
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    out.push({
      name: a?.name || b?.name || keypointNames[i] || `kp_${i}`,
      x,
      y,
      confidence: lerpNumber(a?.confidence, b?.confidence, t) ?? 1,
    })
  }
  return out
}

function interpolatePoseFrame(aFrame, bFrame, t, participants, keypointNames) {
  const aPlayers = new Map((Array.isArray(aFrame?.players) ? aFrame.players : []).map((p) => [Number(p?.id), p]))
  const bPlayers = new Map((Array.isArray(bFrame?.players) ? bFrame.players : []).map((p) => [Number(p?.id), p]))
  const ids = participants.length ? participants : Array.from(new Set([...aPlayers.keys(), ...bPlayers.keys()]))

  return {
    players: ids
      .map((pid) => {
        const a = aPlayers.get(Number(pid))
        const b = bPlayers.get(Number(pid))
        if (!a && !b) return null
        return {
          id: Number(pid),
          court_pos: interpolateCourtPos(a?.court_pos, b?.court_pos, t),
          keypoints: interpolateKeypoints(a?.keypoints, b?.keypoints, keypointNames, t),
        }
      })
      .filter(Boolean),
    detections: Array.isArray(aFrame?.detections)
      ? aFrame.detections
      : Array.isArray(bFrame?.detections)
        ? bFrame.detections
        : [],
  }
}

function buildProceduralKabaddiTargets({
  root,
  nextRoot,
  raiderRoot,
  isRaider,
  phase,
}) {
  const up = new THREE.Vector3(0, 1, 0)
  const move = nextRoot
    ? new THREE.Vector3().subVectors(nextRoot, root)
    : new THREE.Vector3(0, 0, 0)
  move.y = 0

  const forward = move.clone()
  if (forward.lengthSq() < 1e-5) {
    if (!isRaider && raiderRoot) {
      forward.copy(new THREE.Vector3().subVectors(raiderRoot, root))
      forward.y = 0
    }
  }
  if (forward.lengthSq() < 1e-5) forward.set(0, 0, 1)
  forward.normalize()

  const right = new THREE.Vector3().crossVectors(up, forward).normalize()
  const speed = _clamp(move.length() / 0.22, 0, 1.15)
  const distToRaider = !isRaider && raiderRoot ? raiderRoot.distanceTo(root) : 99
  const engaged = distToRaider < 1.35
  const wave = phase * Math.PI * 2
  const strideWave = Math.sin(wave)
  const counterWave = Math.sin(wave + Math.PI)
  const crouch = isRaider ? 0.22 : engaged ? 0.2 : 0.14
  const lean = isRaider ? 0.16 + speed * 0.12 : engaged ? 0.08 : 0.05
  const bob = Math.sin(wave * 2) * (0.02 + speed * 0.025)
  const stride = 0.08 + speed * 0.16
  const armSwing = 0.05 + speed * 0.11

  const hip = root.clone().addScaledVector(up, 0.86 - crouch * 0.18 + bob)
  const chest = hip.clone().addScaledVector(up, 0.4).addScaledVector(forward, lean)
  const head = chest.clone().addScaledVector(up, 0.34).addScaledVector(forward, 0.04)

  const leftHip = hip.clone().addScaledVector(right, -0.12)
  const rightHip = hip.clone().addScaledVector(right, 0.12)
  const leftShoulder = chest.clone().addScaledVector(right, -0.22).addScaledVector(up, 0.07)
  const rightShoulder = chest.clone().addScaledVector(right, 0.22).addScaledVector(up, 0.07)

  const leftKnee = leftHip
    .clone()
    .addScaledVector(forward, stride * strideWave * 0.65)
    .addScaledVector(up, -0.34 + Math.max(0, strideWave) * 0.09)
  const rightKnee = rightHip
    .clone()
    .addScaledVector(forward, stride * counterWave * 0.65)
    .addScaledVector(up, -0.34 + Math.max(0, counterWave) * 0.09)
  const leftAnkle = leftHip
    .clone()
    .addScaledVector(forward, stride * strideWave)
    .addScaledVector(up, -0.78 + Math.max(0, strideWave) * 0.05)
  const rightAnkle = rightHip
    .clone()
    .addScaledVector(forward, stride * counterWave)
    .addScaledVector(up, -0.78 + Math.max(0, counterWave) * 0.05)

  const armReach = isRaider ? 0.34 : engaged ? 0.4 : 0.22
  const guardLift = isRaider ? 0.08 : engaged ? 0.12 : 0.02
  const leftElbow = leftShoulder
    .clone()
    .addScaledVector(forward, armReach * 0.58 + armSwing * counterWave)
    .addScaledVector(up, -0.08 + guardLift)
  const rightElbow = rightShoulder
    .clone()
    .addScaledVector(forward, armReach * 0.58 + armSwing * strideWave)
    .addScaledVector(up, -0.08 + guardLift)
  const leftWrist = leftShoulder
    .clone()
    .addScaledVector(forward, armReach + armSwing * counterWave * 1.2)
    .addScaledVector(up, -0.16 + guardLift)
  const rightWrist = rightShoulder
    .clone()
    .addScaledVector(forward, armReach + armSwing * strideWave * 1.2)
    .addScaledVector(up, -0.16 + guardLift)

  if (!isRaider && engaged && raiderRoot) {
    const reachTarget = raiderRoot.clone().addScaledVector(up, 1.05)
    const sideDot = new THREE.Vector3().subVectors(reachTarget, root).dot(right)
    if (sideDot >= 0) {
      rightElbow.lerp(reachTarget, 0.42)
      rightWrist.lerp(reachTarget, 0.68)
    } else {
      leftElbow.lerp(reachTarget, 0.42)
      leftWrist.lerp(reachTarget, 0.68)
    }
  }

  return {
    forward,
    hip,
    chest,
    head,
    leftShoulder,
    rightShoulder,
    leftElbow,
    rightElbow,
    leftWrist,
    rightWrist,
    leftHip,
    rightHip,
    leftKnee,
    rightKnee,
    leftAnkle,
    rightAnkle,
  }
}

function findPlayerWindowEndpoints(matWindow, pid) {
  let first = null
  let last = null
  for (const frame of Array.isArray(matWindow) ? matWindow : []) {
    const players = Array.isArray(frame?.players) ? frame.players : []
    const player = players.find((entry) => Number(entry?.id) === Number(pid) && hasValidCourtPos(entry))
    if (!player) continue
    if (!first) first = player.court_pos
    last = player.court_pos
  }
  return { first, last }
}

function chooseKabaddiPreset({ isRaider, index, startCourt, endCourt }) {
  const dx = Math.abs((_asNum(endCourt?.[0], 0) || 0) - (_asNum(startCourt?.[0], 0) || 0))
  const dy = Math.abs((_asNum(endCourt?.[1], 0) || 0) - (_asNum(startCourt?.[1], 0) || 0))
  const distance = Math.hypot(dx, dy)
  if (isRaider) return distance > 1.25 ? 'raider_burst' : 'raider_probe'
  return index % 2 === 0 ? 'defender_guard' : 'defender_lunge'
}

function buildPresetKabaddiTargets({
  root,
  startRoot,
  endRoot,
  raiderRoot,
  isRaider,
  preset,
  progress,
  actorIndex,
}) {
  const up = new THREE.Vector3(0, 1, 0)
  const path = endRoot
    ? new THREE.Vector3().subVectors(endRoot, startRoot || root)
    : new THREE.Vector3(0, 0, 0)
  path.y = 0
  let forward = path.clone()
  if (forward.lengthSq() < 1e-6 && !isRaider && raiderRoot) {
    forward.copy(new THREE.Vector3().subVectors(raiderRoot, root))
    forward.y = 0
  }
  if (forward.lengthSq() < 1e-6) forward.set(0, 0, 1)
  forward.normalize()
  const right = new THREE.Vector3().crossVectors(up, forward).normalize()
  const phase = progress * Math.PI * (isRaider ? 3.6 : 2.8) + actorIndex * 0.9
  const wave = Math.sin(phase)
  const anti = Math.sin(phase + Math.PI)
  const pulse = Math.sin(progress * Math.PI)
  const towardRaider = !isRaider && raiderRoot
    ? new THREE.Vector3().subVectors(raiderRoot, root).setY(0)
    : null
  if (towardRaider && towardRaider.lengthSq() > 1e-6) towardRaider.normalize()

  let crouch = 0.14
  let lean = 0.06
  let stride = 0.1
  let armReach = 0.24
  let guardHeight = 0.05
  let shoulderBias = 0

  switch (preset) {
    case 'raider_probe':
      crouch = 0.2
      lean = 0.12 + pulse * 0.05
      stride = 0.12
      armReach = 0.28
      guardHeight = 0.08
      shoulderBias = 0.03
      break
    case 'raider_burst':
      crouch = 0.18
      lean = 0.18 + pulse * 0.08
      stride = 0.18
      armReach = 0.25
      guardHeight = 0.03
      shoulderBias = -0.02
      break
    case 'defender_lunge':
      crouch = 0.22
      lean = 0.1 + pulse * 0.04
      stride = 0.09
      armReach = 0.38
      guardHeight = 0.12
      shoulderBias = 0.06
      break
    case 'defender_guard':
    default:
      crouch = 0.16
      lean = 0.05
      stride = 0.07
      armReach = 0.22
      guardHeight = 0.11
      shoulderBias = 0.02
      break
  }

  const bob = Math.sin(phase * 2) * 0.018
  const hip = root.clone().addScaledVector(up, 0.84 - crouch * 0.2 + bob)
  const chest = hip.clone().addScaledVector(up, 0.39).addScaledVector(forward, lean)
  const head = chest.clone().addScaledVector(up, 0.34).addScaledVector(forward, 0.03)

  const leftHip = hip.clone().addScaledVector(right, -0.12)
  const rightHip = hip.clone().addScaledVector(right, 0.12)
  const leftShoulder = chest.clone().addScaledVector(right, -0.22).addScaledVector(up, 0.07 + shoulderBias)
  const rightShoulder = chest.clone().addScaledVector(right, 0.22).addScaledVector(up, 0.07 - shoulderBias)

  const leftKnee = leftHip.clone().addScaledVector(forward, stride * wave * 0.58).addScaledVector(up, -0.33 + Math.max(0, wave) * 0.08)
  const rightKnee = rightHip.clone().addScaledVector(forward, stride * anti * 0.58).addScaledVector(up, -0.33 + Math.max(0, anti) * 0.08)
  const leftAnkle = leftHip.clone().addScaledVector(forward, stride * wave).addScaledVector(up, -0.77 + Math.max(0, wave) * 0.04)
  const rightAnkle = rightHip.clone().addScaledVector(forward, stride * anti).addScaledVector(up, -0.77 + Math.max(0, anti) * 0.04)

  const leftElbow = leftShoulder.clone().addScaledVector(forward, armReach * 0.6 + anti * 0.05).addScaledVector(up, -0.08 + guardHeight)
  const rightElbow = rightShoulder.clone().addScaledVector(forward, armReach * 0.6 + wave * 0.05).addScaledVector(up, -0.08 + guardHeight)
  const leftWrist = leftShoulder.clone().addScaledVector(forward, armReach + anti * 0.08).addScaledVector(up, -0.16 + guardHeight)
  const rightWrist = rightShoulder.clone().addScaledVector(forward, armReach + wave * 0.08).addScaledVector(up, -0.16 + guardHeight)

  if (!isRaider && towardRaider && preset === 'defender_lunge') {
    const tackleTarget = raiderRoot.clone().addScaledVector(up, 0.92)
    const sideDot = towardRaider.dot(right)
    if (sideDot >= 0) {
      rightElbow.lerp(tackleTarget, 0.46)
      rightWrist.lerp(tackleTarget, 0.7)
    } else {
      leftElbow.lerp(tackleTarget, 0.46)
      leftWrist.lerp(tackleTarget, 0.7)
    }
  }

  return {
    forward,
    hip,
    chest,
    head,
    leftShoulder,
    rightShoulder,
    leftElbow,
    rightElbow,
    leftWrist,
    rightWrist,
    leftHip,
    rightHip,
    leftKnee,
    rightKnee,
    leftAnkle,
    rightAnkle,
  }
}

function chooseAnimationState({
  isRaider,
  progress,
  movement,
  moveDir,
  previousMoveDir,
  distToRaider,
}) {
  const speed = movement?.length?.() || 0
  const turning =
    previousMoveDir && moveDir && previousMoveDir.lengthSq() > 1e-6 && moveDir.lengthSq() > 1e-6
      ? previousMoveDir.angleTo(moveDir)
      : 0

  if (turning > 0.55) return 'changeDirection'
  if (progress < 0.12 && speed < 0.08) return 'crouchToStand'
  if (speed < 0.045) return 'idle'
  if (isRaider) {
    if (speed > 0.13) return 'runForward'
    return 'crouchedWalking'
  }
  if (Number.isFinite(distToRaider) && distToRaider < 1.15 && speed < 0.11) return 'runningBackward'
  if (speed > 0.14) return 'runForward'
  return 'crouchedWalking'
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
  const liveCameraRef = useRef(null)
  const liveControlsRef = useRef(null)
  const [previewVideoOk, setPreviewVideoOk] = useState(true)
  const [stats, setStats] = useState({ detected: 0, matched: 0 })
  const [modelReady, setModelReady] = useState(false)
  const [modelError, setModelError] = useState(null)
  const [boneDebugSummary, setBoneDebugSummary] = useState('')
  const [scrollIndicator, setScrollIndicator] = useState(null)
  const [cameraLocked, setCameraLocked] = useState(false)
  const [previewMinimized, setPreviewMinimized] = useState(false)
  const [hoveredCard, setHoveredCard] = useState(null)
  const scrubValueRef = useRef(0)
  const renderScrubRef = useRef(0)
  const totalFramesRef = useRef(0)
  const frameInfoRef = useRef({
    activeFrameIndex: 0,
    leftFrameIndex: 0,
    rightFrameIndex: 0,
    scrubMix: 0,
  })
  const pointerRef = useRef(new THREE.Vector2(2, 2))
  const hoverPointRef = useRef({ x: 0, y: 0 })
  const hoveredDataRef = useRef(null)
  const lastHoverSignatureRef = useRef('')

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
  const raiderId = useMemo(() => {
    const mid = Array.isArray(matWindow) && matWindow.length ? matWindow[Math.floor(matWindow.length / 2)] : null
    return _asInt(mid?.raider_id)
  }, [matWindow])
  const validFrameIndices = useMemo(() => {
    return Array.isArray(matWindow) ? matWindow.map((_, idx) => idx) : []
  }, [matWindow])
  const playerMotionPlan = useMemo(() => {
    const entries = new Map()
    for (let i = 0; i < participants.length; i++) {
      const pid = Number(participants[i])
      const { first, last } = findPlayerWindowEndpoints(matWindow, pid)
      entries.set(pid, {
        startCourt: first,
        endCourt: last || first,
        preset: chooseKabaddiPreset({
          isRaider: pid === Number(raiderId),
          index: i,
          startCourt: first,
          endCourt: last || first,
        }),
      })
    }
    return entries
  }, [matWindow, participants, raiderId])
  const totalFrames = useMemo(
    () => validFrameIndices.length,
    [validFrameIndices],
  )
  const replayTitle = useMemo(
    () => formatReplaySourceLabel(videoFileSrc || videoSrc || event?.clip_url || event?.video_path || event?.raid_label),
    [videoFileSrc, videoSrc, event],
  )
  const participantSummary = useMemo(() => {
    if (!participants.length) return 'No players'
    return participants
      .map((pid) => (Number(pid) === Number(raiderId) ? `Raider ${pid}` : `ID ${pid}`))
      .join(' • ')
  }, [participants, raiderId])
  const replaySubtitle = useMemo(() => {
    const frameStart = validFrameIndices[0] ?? 0
    const frameEnd = validFrameIndices[Math.max(0, validFrameIndices.length - 1)] ?? 0
    return `Frames ${frameStart}-${frameEnd} • ${participants.length} players • ${poseWindow?.length || 0} pose frames`
  }, [validFrameIndices, participants.length, poseWindow])
  const [scrubValue, setScrubValue] = useState(0)
  const scrubBaseIndex = Math.floor(scrubValue)
  const scrubMix = Math.max(0, Math.min(1, scrubValue - scrubBaseIndex))
  const leftFrameIndex = validFrameIndices[Math.min(scrubBaseIndex, Math.max(0, totalFrames - 1))] ?? 0
  const rightFrameIndex =
    validFrameIndices[Math.min(scrubBaseIndex + 1, Math.max(0, totalFrames - 1))] ?? leftFrameIndex
  const activeFrameFloat = Math.max(0, Math.min(Math.max(0, totalFrames - 1), scrubValue))
  const activeFrameIndex = validFrameIndices[Math.round(activeFrameFloat)] ?? 0

  useEffect(() => {
    scrubValueRef.current = scrubValue
    totalFramesRef.current = totalFrames
    frameInfoRef.current = {
      activeFrameIndex,
      leftFrameIndex,
      rightFrameIndex,
      scrubMix,
    }
  }, [scrubValue, totalFrames, activeFrameIndex, leftFrameIndex, rightFrameIndex, scrubMix])

  const lockCurrentView = () => {
    try {
      if (cameraLocked) {
        window.localStorage.removeItem(RAID_REPLAY_CAMERA_LOCK_KEY)
        setCameraLocked(false)
        return
      }
      const camera = liveCameraRef.current
      const controls = liveControlsRef.current
      if (!camera || !controls) return
      window.localStorage.setItem(
        RAID_REPLAY_CAMERA_LOCK_KEY,
        JSON.stringify({
          position: [camera.position.x, camera.position.y, camera.position.z],
          target: [controls.target.x, controls.target.y, controls.target.z],
        }),
      )
      setCameraLocked(true)
    } catch {
      // ignore camera lock write failures
    }
  }

  useEffect(() => {
    try {
      setCameraLocked(Boolean(window.localStorage.getItem(RAID_REPLAY_CAMERA_LOCK_KEY)))
    } catch {
      setCameraLocked(false)
    }
  }, [])

  useEffect(() => {
    setScrubValue(0)
    renderScrubRef.current = 0
    setScrollIndicator(null)
  }, [totalFrames])

  useEffect(() => {
    if (!scrollIndicator) return
    const timer = window.setTimeout(() => setScrollIndicator(null), 900)
    return () => window.clearTimeout(timer)
  }, [scrollIndicator])

  useEffect(() => {
    const videoEl = videoElRef.current
    if (!videoEl || !previewVideoOk || !totalFrames) return
    const nextTime = activeFrameFloat / 30
    const applySeek = () => {
      try {
        videoEl.pause()
        videoEl.currentTime = nextTime
      } catch {
        // ignore
      }
    }
    if (videoEl.readyState >= 1) {
      applySeek()
      return
    }
    const onMeta = () => applySeek()
    videoEl.addEventListener('loadedmetadata', onMeta, { once: true })
    return () => {
      videoEl.removeEventListener('loadedmetadata', onMeta)
    }
  }, [activeFrameFloat, previewVideoOk, totalFrames])

  useEffect(() => {
    const canvas = overlayCanvasRef.current
    const videoEl = videoElRef.current
    const imgEl = imgElRef.current
    if (!canvas) return

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
        const poseFrame = interpolatePoseFrame(
          poseWindow[leftFrameIndex] || {},
          poseWindow[rightFrameIndex] || {},
          scrubMix,
          participants,
          keypointNames,
        )
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
    }

    draw()
  }, [poseWindow, keypointNames, skeletonEdges, isDark, previewVideoOk, leftFrameIndex, rightFrameIndex, scrubMix, participants])

  useEffect(() => {
    const el = mountRef.current
    if (!el || !Array.isArray(matWindow) || matWindow.length === 0) return

    const scene = new THREE.Scene()
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(MANNEQUIN_RENDER_PIXEL_RATIO, window.devicePixelRatio || 1))
    renderer.setClearColor(0x000000, 0)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    el.appendChild(renderer.domElement)

    const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 200)
    const controls = new OrbitControls(camera, renderer.domElement)
    liveCameraRef.current = camera
    liveControlsRef.current = controls
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 2.5
    controls.maxDistance = 20
    controls.maxPolarAngle = Math.PI * 0.49
    controls.enableZoom = false

    const ambient = new THREE.AmbientLight(0xffffff, isDark ? 0.3 : 0.38)
    scene.add(ambient)
    const key = new THREE.DirectionalLight(0xffffff, 1.1)
    key.position.set(6, 10, 4)
    scene.add(key)
    const fill = new THREE.DirectionalLight(0xffffff, 0.35)
    fill.position.set(-4, 6, -3)
    scene.add(fill)
    const spotlight = new THREE.SpotLight(
      isDark ? 0xf8fafc : 0xffffff,
      isDark ? 1.4 : 1.1,
      28,
      Math.PI * 0.32,
      0.38,
      1,
    )
    spotlight.position.set(0, 10.5, 1.2)
    spotlight.target.position.set(0, 0, 0)
    spotlight.castShadow = false
    scene.add(spotlight)
    scene.add(spotlight.target)

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

    const mid = matWindow[validFrameIndices[Math.floor(validFrameIndices.length / 2)] ?? Math.floor(matWindow.length / 2)] || {}
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
    const firstValidFrameIdx = validFrameIndices[0] ?? 0
    const firstPlayers = Array.isArray(matWindow?.[firstValidFrameIdx]?.players) ? matWindow[firstValidFrameIdx].players : []
    const firstPositionFor = (pid) => {
      const match = firstPlayers.find((player) => Number(player?.id) === Number(pid))
      return posFromCourt(match?.court_pos)
    }
    fitReplayCamera(camera, controls, focus, meta, 1)
    try {
      const raw = window.localStorage.getItem(RAID_REPLAY_CAMERA_LOCK_KEY)
      if (raw) {
        const saved = JSON.parse(raw)
        const savedPos = Array.isArray(saved?.position) ? saved.position : null
        const savedTarget = Array.isArray(saved?.target) ? saved.target : null
        if (savedPos?.length === 3 && savedTarget?.length === 3) {
          camera.position.set(Number(savedPos[0]) || 0, Number(savedPos[1]) || 0, Number(savedPos[2]) || 0)
          controls.target.set(Number(savedTarget[0]) || 0, Number(savedTarget[1]) || 0, Number(savedTarget[2]) || 0)
        }
      }
    } catch {
      // ignore camera lock read failures
    }
    controls.update()

    const tmpMid = new THREE.Vector3()
    const tmpDir = new THREE.Vector3()
    const raycaster = new THREE.Raycaster()
    const jointGeom = new THREE.SphereGeometry(0.045, 10, 10)
    const rigGroup = new THREE.Group()
    scene.add(rigGroup)
    const rigs = new Map()
    const mannequinGroup = new THREE.Group()
    scene.add(mannequinGroup)

    const rigColor = (idx, pid) =>
      Number(pid) === Number(raiderId)
        ? new THREE.Color(isDark ? 0xfbbf24 : 0xb45309)
        : idx === 0
          ? new THREE.Color(isDark ? 0xf5deb3 : 0x8b6b2f)
          : idx === 1
            ? new THREE.Color(isDark ? 0xe2e8f0 : 0x1f2937)
            : new THREE.Color(isDark ? 0x94a3b8 : 0x475569)

    for (let i = 0; i < participants.length; i++) {
      const pid = Number(participants[i])
      const color = rigColor(i, pid)
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
      rigs.set(pid, {
        index: i,
        id: pid,
        lineSegments,
        lineGeometry,
        lineMaterial,
        joints,
        jointMaterial,
        mannequin: null,
        boneMap: null,
        restPose: null,
        label: null,
        labelTexture: null,
        labelMaterial: null,
        lastValidPosition: null,
        lastValidQuaternion: null,
        lastMoveDir: null,
        mixer: null,
        animationActions: null,
        activeActionKey: null,
        activeAction: null,
      })
    }

    let alive = true
    const clock = new THREE.Clock()
    Promise.all([loadMannequinAsset(), loadAnimationAssets().catch(() => null)])
      .then(([gltf, animationAssets]) => {
        if (!alive) return
        if (!mannequinBoneDebugPrinted) {
          const boneNames = listBoneNames(gltf.scene)
          mannequinBoneDebugPrinted = true
          console.groupCollapsed(`[${MANNEQUIN_ASSET_LABEL}] Bone names`)
          boneNames.forEach((name, idx) => console.log(`${idx}: ${name}`))
          console.groupEnd()
          setBoneDebugSummary(
            boneNames.length
              ? `${boneNames.length} bones | ${boneNames.slice(0, 6).join(', ')}${boneNames.length > 6 ? ' ...' : ''}`
              : `No bones found in ${MANNEQUIN_ASSET_LABEL}`,
          )
        }
        participants.forEach((pid, idx) => {
          const rig = rigs.get(Number(pid))
          if (!rig || rig.mannequin) return
          const root = SkeletonUtils.clone(gltf.scene)
          const color = rigColor(idx, pid)
          tintModel(root, color, isDark)
          normalizeModelPlacement(root)
          root.traverse((obj) => {
            if (obj?.isMesh) obj.userData.playerId = Number(pid)
          })
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
          const initialCourtPos = firstPositionFor(pid)
          const initialAnchorPos = initialCourtPos
            ? new THREE.Vector3(initialCourtPos.x, MANNEQUIN_Y_OFFSET, initialCourtPos.z + MANNEQUIN_Z_OFFSET)
            : new THREE.Vector3(0, MANNEQUIN_Y_OFFSET, 0)
          anchor.position.copy(initialAnchorPos)
          anchor.add(root)
          const labelAsset = makeTextSprite(
            Number(pid) === Number(raiderId) ? `RAIDER ${pid}` : `ID ${pid}`,
            isDark ? 'dark' : 'light',
          )
          labelAsset.sprite.position.set(0, MANNEQUIN_TARGET_HEIGHT + 0.28, 0)
          anchor.add(labelAsset.sprite)
          if (Number(pid) === Number(raiderId)) {
            const ringGeom = new THREE.RingGeometry(0.18, 0.28, 32)
            const ringMat = new THREE.MeshBasicMaterial({
              color: isDark ? 0xfbbf24 : 0xb45309,
              transparent: true,
              opacity: 0.9,
              side: THREE.DoubleSide,
            })
            const ring = new THREE.Mesh(ringGeom, ringMat)
            ring.rotation.x = -Math.PI / 2
            ring.position.set(0, 0.012, 0)
            ring.userData.playerId = Number(pid)
            anchor.add(ring)
            rig.raiderRing = ring
            rig.raiderRingMaterial = ringMat
            rig.raiderRingGeometry = ringGeom
          }
          mannequinGroup.add(anchor)
          rig.mannequin = anchor
          rig.boneMap = boneMap
          rig.restPose = captureBoneRestPose(boneMap)
          rig.label = labelAsset.sprite
          rig.labelTexture = labelAsset.texture
          rig.labelMaterial = labelAsset.material
          rig.lastValidPosition = initialAnchorPos.clone()
          rig.mixer = new THREE.AnimationMixer(root)
          rig.animationActions = new Map()
          if (animationAssets?.size) {
            const targetHelper = new THREE.SkeletonHelper(root)
            const nameMap = buildRetargetNameMap(boneMap)
            for (const [clipKey, clipAsset] of animationAssets.entries()) {
              try {
                const retargetedClip = SkeletonUtils.retargetClip(
                  targetHelper,
                  clipAsset.root,
                  clipAsset.clip,
                  {
                    names: nameMap,
                    preservePosition: false,
                    preserveHipPosition: false,
                    useFirstFramePosition: false,
                    hip: 'mixamorigHips',
                  },
                )
                if (!retargetedClip) continue
                const action = rig.mixer.clipAction(retargetedClip)
                action.enabled = true
                action.setLoop(THREE.LoopRepeat, Infinity)
                action.clampWhenFinished = false
                action.timeScale = 1
                action.setEffectiveWeight(0)
                action.play()
                rig.animationActions.set(clipKey, action)
              } catch {
                // ignore a clip that cannot be retargeted to this mannequin
              }
            }
          }
        })
        setModelReady(true)
        setModelError(null)
      })
      .catch((err) => {
        if (!alive) return
        setModelReady(false)
        setModelError(String(err?.message || err))
      })

    let raf = 0
    const animate = () => {
      const deltaSeconds = Math.min(0.05, clock.getDelta())
      const targetScrub = scrubValueRef.current
      const prevRenderScrub = renderScrubRef.current
      const nextRenderScrub =
        Math.abs(targetScrub - prevRenderScrub) < 0.001
          ? targetScrub
          : prevRenderScrub + (targetScrub - prevRenderScrub) * 0.55
      renderScrubRef.current = nextRenderScrub
      const currentLeftLogicalIndex = Math.min(
        Math.floor(nextRenderScrub),
        Math.max(0, totalFramesRef.current - 1),
      )
      const currentRightLogicalIndex = Math.min(
        currentLeftLogicalIndex + 1,
        Math.max(0, totalFramesRef.current - 1),
      )
      const currentLeftFrameIndex = validFrameIndices[currentLeftLogicalIndex] ?? 0
      const currentRightFrameIndex = validFrameIndices[currentRightLogicalIndex] ?? currentLeftFrameIndex
      const currentScrubMix = Math.max(0, Math.min(1, nextRenderScrub - currentLeftLogicalIndex))
      const currentActiveFrameIndex = validFrameIndices[Math.round(nextRenderScrub)] ?? currentLeftFrameIndex
      const matFrame = matWindow[currentActiveFrameIndex] || {}
      const sceneProgress = matWindow.length > 1 ? currentActiveFrameIndex / (matWindow.length - 1) : 0
      const poseFrame = interpolatePoseFrame(
        Array.isArray(poseWindow) ? poseWindow[currentLeftFrameIndex] || {} : {},
        Array.isArray(poseWindow) ? poseWindow[currentRightFrameIndex] || {} : {},
        currentScrubMix,
        participants,
        keypointNames,
      )
      const posePlayers = Array.isArray(poseFrame.players) ? poseFrame.players : []
      const poseById = new Map(posePlayers.map((player) => [Number(player?.id), player]))
      const matPlayers = Array.isArray(matFrame.players) ? matFrame.players : []
      const matById = new Map(matPlayers.map((player) => [Number(player?.id), player]))
      const leftMatPlayers = Array.isArray(matWindow[currentLeftFrameIndex]?.players)
        ? matWindow[currentLeftFrameIndex].players
        : []
      const rightMatPlayers = Array.isArray(matWindow[currentRightFrameIndex]?.players)
        ? matWindow[currentRightFrameIndex].players
        : []
      const leftMatById = new Map(leftMatPlayers.map((player) => [Number(player?.id), player]))
      const rightMatById = new Map(rightMatPlayers.map((player) => [Number(player?.id), player]))
      const raiderPlan = raiderId != null ? playerMotionPlan.get(Number(raiderId)) : null
      const raiderLeft = raiderId != null ? leftMatById.get(Number(raiderId)) : null
      const raiderRight = raiderId != null ? rightMatById.get(Number(raiderId)) : null
      const raiderCourt = interpolateCourtPos(
        raiderPlan?.startCourt || raiderLeft?.court_pos,
        raiderPlan?.endCourt || raiderRight?.court_pos || raiderLeft?.court_pos,
        sceneProgress,
      )
      const raiderRoot = posFromCourt(raiderCourt)

      for (const pid of participants) {
        const rig = rigs.get(Number(pid))
        if (!rig) continue

        const posePlayer = poseById.get(Number(pid))
        const tracked = matById.get(Number(pid))
        const trackedLeft = leftMatById.get(Number(pid))
        const trackedRight = rightMatById.get(Number(pid))
        const motionPlan = playerMotionPlan.get(Number(pid))
        const currentCourt = interpolateCourtPos(
          motionPlan?.startCourt || trackedLeft?.court_pos || tracked?.court_pos,
          motionPlan?.endCourt || trackedRight?.court_pos || tracked?.court_pos,
          sceneProgress,
        )
        const nextCourt = interpolateCourtPos(
          motionPlan?.startCourt || trackedLeft?.court_pos || tracked?.court_pos,
          motionPlan?.endCourt || trackedRight?.court_pos || tracked?.court_pos,
          Math.min(1, sceneProgress + 1 / Math.max(2, matWindow.length - 1)),
        )
        const root = posFromCourt(currentCourt)
        const nextRoot = posFromCourt(nextCourt)
        const positions = rig.lineGeometry.attributes.position.array

        if (!root) {
          rig.lineSegments.visible = false
          rig.joints.forEach((joint) => {
            joint.visible = false
          })
          if (rig.mannequin) {
            if (rig.lastValidPosition) {
              rig.mannequin.visible = true
              rig.mannequin.position.copy(rig.lastValidPosition)
              if (rig.lastValidQuaternion) rig.mannequin.quaternion.copy(rig.lastValidQuaternion)
              if (rig.label) rig.label.position.set(0, MANNEQUIN_TARGET_HEIGHT + 0.28, 0)
            } else {
              rig.mannequin.visible = false
            }
          }
          continue
        }

        const map = posePlayer ? keypointMap(posePlayer, keypointNames) : new Map()
        const leftAnkle = map.get('left_ankle')
        const rightAnkle = map.get('right_ankle')
        const leftHip = map.get('left_hip')
        const rightHip = map.get('right_hip')
        const ankleMid = midpoint(leftAnkle, rightAnkle)
        const hipMid = midpoint(leftHip, rightHip)
        const anchor = ankleMid || hipMid
        const scale = anchor ? estimateScaleMeters(map) : 0
        const to3 = (kp) => {
          if (!kp || !anchor || !scale) return null
          const dx = (kp.x - anchor.x) * scale * 0.78
          const dy = (anchor.y - kp.y) * scale
          const dz = (kp.x - anchor.x) * scale * 0.14
          return new THREE.Vector3(root.x + dx, Math.max(0.03, dy), root.z + dz)
        }
        const posePoints = {
          nose: to3(map.get('nose')),
          leftEye: to3(map.get('left_eye')),
          rightEye: to3(map.get('right_eye')),
          leftEar: to3(map.get('left_ear')),
          rightEar: to3(map.get('right_ear')),
          leftShoulder: to3(map.get('left_shoulder')),
          rightShoulder: to3(map.get('right_shoulder')),
          leftElbow: to3(map.get('left_elbow')),
          rightElbow: to3(map.get('right_elbow')),
          leftWrist: to3(map.get('left_wrist')),
          rightWrist: to3(map.get('right_wrist')),
          leftHip: to3(map.get('left_hip')),
          rightHip: to3(map.get('right_hip')),
          leftKnee: to3(map.get('left_knee')),
          rightKnee: to3(map.get('right_knee')),
          leftAnkle: to3(map.get('left_ankle')),
          rightAnkle: to3(map.get('right_ankle')),
        }

        let visibleEdges = 0
        let cursor = 0
        if (anchor) {
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

        const nose = posePoints.nose
        const neck = midpoint3(posePoints.leftShoulder, posePoints.rightShoulder)
        if (nose && neck) {
          tmpMid.copy(nose).add(neck).multiplyScalar(0.5)
          tmpDir.subVectors(neck, nose)
        }

        if (rig.mannequin && rig.boneMap) {
          rig.mannequin.visible = true
          restoreBoneRestPose(rig.boneMap, rig.restPose)
          rig.lineSegments.visible = false
          rig.joints.forEach((joint) => {
            joint.visible = false
          })

          const desiredPos = new THREE.Vector3(root.x, MANNEQUIN_Y_OFFSET, root.z + MANNEQUIN_Z_OFFSET)
          lerpVector3(rig.mannequin.position, desiredPos, 0.18)
          rig.lastValidPosition = desiredPos.clone()

          const fallbackTargets = buildPresetKabaddiTargets({
            root,
            startRoot: posFromCourt(motionPlan?.startCourt) || root,
            endRoot: posFromCourt(motionPlan?.endCourt) || nextRoot || root,
            raiderRoot,
            isRaider: Number(pid) === Number(raiderId),
            preset: motionPlan?.preset || 'defender_guard',
            progress: sceneProgress,
            actorIndex: rig.index,
          })
          const moveDir = new THREE.Vector3().subVectors(nextRoot || root, root)
          moveDir.y = 0
          const targets = buildPoseDrivenTargets({
            posePoints,
            fallbackTargets,
            movementDir: moveDir,
          })
          const yaw = Math.atan2(targets.forward.x, targets.forward.z)
          const targetQuat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw)
          slerpQuaternion(rig.mannequin.quaternion, targetQuat, 0.16)
          rig.lastValidQuaternion = targetQuat.clone()
          const distToRaider = raiderRoot ? raiderRoot.distanceTo(root) : Infinity

          if (rig.animationActions?.size) {
            const targetActionKey = chooseAnimationState({
              isRaider: Number(pid) === Number(raiderId),
              progress: sceneProgress,
              movement: moveDir,
              moveDir,
              previousMoveDir: rig.lastMoveDir,
              distToRaider,
            })
            const nextAction = rig.animationActions.get(targetActionKey) || rig.animationActions.get('idle')
            if (nextAction) {
              if (rig.activeAction !== nextAction) {
                if (rig.activeAction) rig.activeAction.crossFadeTo(nextAction, 0.28, true)
                nextAction.reset()
                nextAction.enabled = true
                nextAction.setEffectiveWeight(1)
                nextAction.play()
                rig.activeAction = nextAction
                rig.activeActionKey = targetActionKey
              }
              for (const action of rig.animationActions.values()) {
                if (action !== nextAction) action.enabled = true
              }
              rig.mixer.update(deltaSeconds)
            }
          }
          if (rig.boneMap.hips) {
            const restHips = rig.restPose?.hips?.position
            rig.boneMap.hips.position.x = restHips?.x ?? 0
            rig.boneMap.hips.position.y = restHips?.y ?? rig.boneMap.hips.position.y
            rig.boneMap.hips.position.z = restHips?.z ?? 0
          }
          applyBoneDirectionWeighted(rig.boneMap.spine, targets.hip, targets.spineMid || targets.chest, 0.34)
          applyBoneDirectionWeighted(rig.boneMap.spineMid, targets.spineMid || targets.hip, targets.chest, 0.36)
          applyBoneDirectionWeighted(rig.boneMap.chest, targets.chest, targets.neck || targets.head, 0.34)
          applyBoneDirectionWeighted(rig.boneMap.neck, targets.neck || targets.chest, targets.head, 0.3)
          applyBoneDirectionWeighted(rig.boneMap.head, targets.head, targets.headTop || targets.head, 0.24)
          applyBoneDirectionWeighted(rig.boneMap.leftShoulder, targets.leftShoulder, targets.leftElbow, 0.22)
          applyBoneDirectionWeighted(rig.boneMap.rightShoulder, targets.rightShoulder, targets.rightElbow, 0.22)
          applyBoneDirectionWeighted(rig.boneMap.leftUpperArm, targets.leftShoulder, targets.leftElbow, 0.68)
          applyBoneDirectionWeighted(rig.boneMap.leftLowerArm, targets.leftElbow, targets.leftWrist, 0.72)
          applyBoneDirectionWeighted(rig.boneMap.leftHand, targets.leftWrist, targets.leftHand, 0.56)
          applyBoneDirectionWeighted(rig.boneMap.rightUpperArm, targets.rightShoulder, targets.rightElbow, 0.68)
          applyBoneDirectionWeighted(rig.boneMap.rightLowerArm, targets.rightElbow, targets.rightWrist, 0.72)
          applyBoneDirectionWeighted(rig.boneMap.rightHand, targets.rightWrist, targets.rightHand, 0.56)
          applyBoneDirectionWeighted(rig.boneMap.leftUpperLeg, targets.leftHip, targets.leftKnee, 0.62)
          applyBoneDirectionWeighted(rig.boneMap.leftLowerLeg, targets.leftKnee, targets.leftAnkle, 0.68)
          applyBoneDirectionWeighted(rig.boneMap.leftFoot, targets.leftAnkle, targets.leftFoot, 0.44)
          applyBoneDirectionWeighted(rig.boneMap.leftToe, targets.leftFoot, targets.leftToe, 0.4)
          applyBoneDirectionWeighted(rig.boneMap.rightUpperLeg, targets.rightHip, targets.rightKnee, 0.62)
          applyBoneDirectionWeighted(rig.boneMap.rightLowerLeg, targets.rightKnee, targets.rightAnkle, 0.68)
          applyBoneDirectionWeighted(rig.boneMap.rightFoot, targets.rightAnkle, targets.rightFoot, 0.44)
          applyBoneDirectionWeighted(rig.boneMap.rightToe, targets.rightFoot, targets.rightToe, 0.4)
          rig.lastMoveDir = moveDir.lengthSq() > 1e-6 ? moveDir.clone().normalize() : rig.lastMoveDir

          if (rig.label) {
            rig.label.position.set(0, MANNEQUIN_TARGET_HEIGHT + 0.32, 0)
          }
        }
      }

      raycaster.setFromCamera(pointerRef.current, camera)
      const intersections = raycaster
        .intersectObjects(mannequinGroup.children, true)
        .find((hit) => Number.isFinite(Number(hit?.object?.userData?.playerId)))
      const hoveredPid = Number(intersections?.object?.userData?.playerId)
      if (Number.isFinite(hoveredPid)) {
        const tracked = matById.get(hoveredPid) || null
        const posed = poseById.get(hoveredPid) || null
        const details = {
          id: hoveredPid,
          role: hoveredPid === Number(raiderId) ? 'Raider' : 'Player',
          visible: tracked?.visible ?? null,
          courtPos: Array.isArray(tracked?.court_pos) ? tracked.court_pos : null,
          bbox: Array.isArray(tracked?.bbox) ? tracked.bbox : null,
          keypoints: Array.isArray(posed?.keypoints) ? posed.keypoints.length : 0,
          x: hoverPointRef.current.x,
          y: hoverPointRef.current.y,
        }
        hoveredDataRef.current = details
        const signature = JSON.stringify([
          details.id,
          currentActiveFrameIndex,
          details.visible,
          details.courtPos?.[0],
          details.courtPos?.[1],
          details.bbox?.join(','),
          details.keypoints,
          details.x,
          details.y,
        ])
        if (signature !== lastHoverSignatureRef.current) {
          lastHoverSignatureRef.current = signature
          setHoveredCard(details)
        }
      } else if (lastHoverSignatureRef.current) {
        hoveredDataRef.current = null
        lastHoverSignatureRef.current = ''
        setHoveredCard(null)
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
      if (!cameraLocked) fitReplayCamera(camera, controls, focus, meta, w / h)
      camera.updateProjectionMatrix()
      controls.update()
    }
    const onWheel = (ev) => {
      const delta = Number(ev.deltaY || 0)
      if (ev.ctrlKey) {
        ev.preventDefault()
        const zoomStep = delta > 0 ? 0.7 : -0.7
        const offset = new THREE.Vector3().subVectors(camera.position, controls.target)
        const distance = offset.length()
        const nextDistance = Math.max(controls.minDistance, Math.min(controls.maxDistance, distance + zoomStep))
        if (distance > 1e-6) {
          offset.setLength(nextDistance)
          camera.position.copy(controls.target.clone().add(offset))
        }
        controls.update()
        return
      }
      if (totalFrames <= 1) return
      ev.preventDefault()
      // Scroll down -> forward in time, scroll up -> backward in time.
      const direction = delta === 0 ? 0 : delta > 0 ? 1 : -1
      const scrollFrames = Math.max(
        1,
        Math.min(
          SCROLL_FRAME_DELTA_MAX,
          Math.round(Math.abs(delta) / SCROLL_FRAME_DELTA_DIVISOR),
        ),
      )
      const frameCount = totalFramesRef.current
      const next = Math.max(0, Math.min(frameCount - 1, scrubValueRef.current + direction * scrollFrames))
      renderScrubRef.current = next
      setScrubValue(next)
      setScrollIndicator({
        frame: Math.round(next),
        direction,
      })
    }
    const onMouseMove = (ev) => {
      const rect = renderer.domElement.getBoundingClientRect()
      pointerRef.current.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1
      pointerRef.current.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1
      hoverPointRef.current = {
        x: ev.clientX - rect.left + 12,
        y: ev.clientY - rect.top + 12,
      }
      if (hoveredDataRef.current) {
        setHoveredCard({
          ...hoveredDataRef.current,
          x: hoverPointRef.current.x,
          y: hoverPointRef.current.y,
        })
      }
    }
    const onMouseLeave = () => {
      pointerRef.current.set(2, 2)
      hoveredDataRef.current = null
      lastHoverSignatureRef.current = ''
      setHoveredCard(null)
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    window.addEventListener('resize', resize)
    document.addEventListener('fullscreenchange', resize)
    renderer.domElement.addEventListener('wheel', onWheel, { passive: false })
    renderer.domElement.addEventListener('mousemove', onMouseMove)
    renderer.domElement.addEventListener('mouseleave', onMouseLeave)
    resize()
    raf = requestAnimationFrame(animate)

    return () => {
      alive = false
      if (raf) cancelAnimationFrame(raf)
      try {
        renderer.domElement.removeEventListener('wheel', onWheel)
        renderer.domElement.removeEventListener('mousemove', onMouseMove)
        renderer.domElement.removeEventListener('mouseleave', onMouseLeave)
      } catch {
        // ignore
      }
      try {
        controls.dispose()
      } catch {
        // ignore
      }
      liveCameraRef.current = null
      liveControlsRef.current = null
      try {
        ro.disconnect()
      } catch {
        // ignore
      }
      try {
        window.removeEventListener('resize', resize)
        document.removeEventListener('fullscreenchange', resize)
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
        try {
          rig.labelTexture?.dispose?.()
        } catch {
          // ignore
        }
        try {
          rig.labelMaterial?.dispose?.()
        } catch {
          // ignore
        }
        try {
          rig.raiderRingGeometry?.dispose?.()
        } catch {
          // ignore
        }
        try {
          rig.raiderRingMaterial?.dispose?.()
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
  }, [matWindow, poseWindow, participants, meta, keypointNames, skeletonEdges, isDark, cameraLocked])

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

      <div className="pointer-events-none absolute left-4 top-4 z-20 max-w-[min(62rem,72vw)] rounded-2xl border border-white/10 bg-black/28 px-4 py-3 text-white shadow-xl backdrop-blur-md">
        <div className="text-lg font-semibold tracking-[0.02em] text-white/96">
          {replayTitle}
        </div>
        <div className="mt-1 text-[12px] text-white/78">
          {participantSummary}
        </div>
        <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.12em] text-white/58">
          {replaySubtitle}
        </div>
      </div>

      <button
        type="button"
        onClick={lockCurrentView}
        className="absolute right-3 top-14 z-30 rounded-full border border-white/10 bg-black/45 px-3 py-1 text-[11px] font-semibold text-white backdrop-blur transition hover:bg-black/60"
      >
        {cameraLocked ? 'Locked' : 'Lock'}
      </button>
      {scrollIndicator ? (
        <div className="pointer-events-none absolute left-1/2 top-6 z-30 -translate-x-1/2 rounded-full border border-white/10 bg-black/55 px-4 py-2 text-sm font-semibold text-white shadow-lg backdrop-blur">
          Frame {scrollIndicator.frame}
          {scrollIndicator.direction < 0 ? ' | reverse' : scrollIndicator.direction > 0 ? ' | forward' : ''}
        </div>
      ) : null}
      {hoveredCard ? (
        <div
          className="pointer-events-none absolute z-30 w-52 rounded-xl border border-white/10 bg-black/75 px-3 py-2 text-[11px] text-white shadow-xl backdrop-blur"
          style={{
            left: hoveredCard.x,
            top: hoveredCard.y,
          }}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="font-semibold">{hoveredCard.role}</span>
            <span className="rounded-full border border-white/10 px-2 py-0.5 text-[10px] font-semibold">
              ID {hoveredCard.id}
            </span>
          </div>
          <div className="mt-2 space-y-1 text-white/85">
            <div>
              Court:{' '}
              {hoveredCard.courtPos
                ? `${Number(hoveredCard.courtPos[0]).toFixed(2)}, ${Number(hoveredCard.courtPos[1]).toFixed(2)}`
                : 'n/a'}
            </div>
            <div>Visible: {hoveredCard.visible == null ? 'n/a' : hoveredCard.visible ? 'yes' : 'no'}</div>
            <div>
              BBox:{' '}
              {hoveredCard.bbox
                ? hoveredCard.bbox.map((v) => Math.round(Number(v) || 0)).join(', ')
                : 'n/a'}
            </div>
            <div>Pose keypoints: {hoveredCard.keypoints ?? 0}</div>
          </div>
        </div>
      ) : null}

      <div
        className="absolute bottom-4 right-4 z-20 overflow-hidden rounded-2xl border border-white/10 bg-black/35 shadow-2xl backdrop-blur"
        style={{
          width: previewMinimized ? 132 : 'min(560px, 40vw)',
          height: previewMinimized ? 42 : 'min(320px, 28vw)',
          minWidth: previewMinimized ? 132 : 360,
          minHeight: previewMinimized ? 42 : 220,
        }}
      >
        <button
          type="button"
          onClick={() => setPreviewMinimized((prev) => !prev)}
          className="absolute right-2 top-2 z-30 rounded-full border border-white/10 bg-black/55 px-2 py-1 text-[10px] font-semibold text-white backdrop-blur transition hover:bg-black/70"
        >
          {previewMinimized ? 'Expand Video' : 'Minimize Video'}
        </button>
        {previewMinimized ? (
          <div className="flex h-full w-full items-center justify-start px-3 text-[11px] font-medium text-white/85">
            Preview video minimized
          </div>
        ) : null}
        {videoFileSrc ? (
          <video
            ref={videoElRef}
            src={videoFileSrc}
            className="absolute inset-0 h-full w-full object-contain"
            muted
            playsInline
            preload="auto"
            onError={() => setPreviewVideoOk(false)}
            style={{ display: previewMinimized ? 'none' : previewVideoOk ? 'block' : 'none' }}
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
            style={{ display: previewMinimized ? 'none' : previewVideoOk && videoFileSrc ? 'none' : 'block' }}
          />
        ) : null}
        <canvas
          ref={overlayCanvasRef}
          className="absolute inset-0 h-full w-full"
          style={{ pointerEvents: 'none', display: previewMinimized ? 'none' : 'block' }}
        />
      </div>

      {!hasPoseWindow ? (
        <div className="absolute bottom-4 left-4 z-20 rounded-xl border border-amber-200 bg-amber-50/90 px-3 py-2 text-xs text-amber-900 shadow-sm dark:border-amber-900/60 dark:bg-amber-950/50 dark:text-amber-100">
          No archived pose window yet. Re-run the backend once so confirmed events are exported with YOLO bones.
        </div>
      ) : null}
      {modelError ? (
        <div className="absolute bottom-16 left-4 z-20 rounded-xl border border-amber-200 bg-amber-50/90 px-3 py-2 text-xs text-amber-900 shadow-sm dark:border-amber-900/60 dark:bg-amber-950/50 dark:text-amber-100">
          Could not load or rig `{MANNEQUIN_ASSET_LABEL}`. Showing stick-figure fallback. {modelError}
        </div>
      ) : null}
    </div>
  )
}
