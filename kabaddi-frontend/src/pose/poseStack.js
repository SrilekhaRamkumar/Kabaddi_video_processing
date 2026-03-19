// Use the package's CommonJS entry instead of the ESM entry because Vite 8 trips
// over the MediaPipe Pose export path used by the ESM bundle.
import * as poseDetection from '@tensorflow-models/pose-detection/dist/index.js'
import '@tensorflow/tfjs-backend-webgl'
import * as tf from '@tensorflow/tfjs-core'

export async function createMoveNetMultiposeDetector() {
  // Ensure TF is ready and using WebGL backend.
  try {
    await tf.setBackend('webgl')
  } catch {
    // backend may already be set
  }
  await tf.ready()

  return await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
    modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
    enableSmoothing: true,
  })
}
