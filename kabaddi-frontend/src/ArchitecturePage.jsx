import { useEffect, useRef, useState } from 'react'

const STEP_LINGER_MS = 50

const STAGES = [
  {
    id: '01',
    title: 'Input And Video Processing',
    summary:
      'This block covers how raw raid footage is loaded, configured, calibrated, and prepared for the live processing loop.',
    files: ['Court_code2.py', 'video_stream.py'],
    modules: [
      {
        stepKey: 'video_input_configuration',
        name: 'Video Input + Configuration',
        detail:
          'Loads the raid video, raid ordering, court settings, and the backend-linked configuration used for each run.',
      },
      {
        stepKey: 'input_layer',
        name: 'Input Layer',
        detail:
          'Starts from the video file and manually selected court calibration points that anchor the rest of the pipeline.',
      },
      {
        stepKey: 'yolov8_entry',
        name: 'YOLOv8 Entry',
        detail:
          'Initializes the main detector model that will observe players in each extracted frame.',
      },
      {
        stepKey: 'video_processing',
        name: 'Video Processing',
        detail:
          'Uses the threaded frame queue and frame extraction loop so detection and visualization can run continuously.',
      },
    ],
  },
  {
    id: '02',
    title: 'Detection, Tracking, Gallery, And Spatial Transform',
    summary:
      'Player observations are detected, matched across frames, stabilized, and projected onto the court representation.',
    files: ['tracking_pipeline.py', 'Court_code2.py'],
    modules: [
      {
        stepKey: 'yolov8_person_detection',
        name: 'YOLOv8 Person Detection',
        detail:
          'Detects player boxes in each frame and provides the base observations for tracking.',
      },
      {
        stepKey: 'hungarian_track_matching',
        name: 'Hungarian Track Matching',
        detail:
          'Associates current detections with existing player IDs instead of creating a new identity every frame.',
      },
      {
        stepKey: 'kalman_predict_correct',
        name: 'Kalman Predict / Correct',
        detail:
          'Smooths each player trajectory and keeps track estimates stable when detections are noisy or briefly missing.',
      },
      {
        stepKey: 'color_embed',
        name: 'Color Embed',
        detail:
          'Uses appearance cues to reduce ID switches when multiple players are close together.',
      },
      {
        stepKey: 'trackers_spawning',
        name: 'Trackers Spawning',
        detail:
          'Creates new tracked players when unmatched detections appear on the mat.',
      },
      {
        stepKey: 'track_gallery',
        name: 'Track Gallery',
        detail:
          'Maintains the gallery state for every player: bbox, foot point, display position, Kalman state, and confidence values.',
      },
      {
        stepKey: 'homography_estimation',
        name: 'Homography Estimation',
        detail:
          'Builds the court transform from calibrated line points so pixel positions can be mapped into court coordinates.',
      },
      {
        stepKey: 'perspective_projection_player_position_map',
        name: 'Perspective Projection + Player Position Map',
        detail:
          'Projects tracked players onto the 2D court map that later reasoning modules use.',
      },
    ],
  },
  {
    id: '03',
    title: 'Raider Identification And Scene Graph Reasoning',
    summary:
      'The system identifies the attacking raider, accumulates player statistics, and builds interaction proposals for AFGN reasoning.',
    files: ['raider_logic.py', 'interaction_logic.py', 'interaction_graph.py'],
    modules: [
      {
        stepKey: 'raider_stats_collection',
        name: 'Raider Stats Collection',
        detail:
          'Collects depth, motion, and court-entry cues for visible players before choosing the raider.',
      },
      {
        stepKey: 'multi_cue_raider_scoring',
        name: 'Multi Cue Raider Scoring',
        detail:
          'Scores candidates using depth rank, defender convergence, nearby players, speed, and entry prior.',
      },
      {
        stepKey: 'raider_id_assignment',
        name: 'Raider ID Assignment',
        detail:
          'Locks the best candidate as the raider once enough frames have been observed.',
      },
      {
        stepKey: 'raider_id_player_stats',
        name: 'Raider ID + Player Stats',
        detail:
          'Combines the assigned raider identity with per-player state to form the live tactical context.',
      },
      {
        stepKey: 'interaction_proposal_engine',
        name: 'Interaction Proposal Engine',
        detail:
          'Builds HHI and HLI triplets such as player-player contact proposals and player-line proximity proposals.',
      },
      {
        stepKey: 'afgn_graph_construction',
        name: 'AFGN Graph Construction',
        detail:
          'Transforms proposals into the active scene graph with pair factors, line factors, and graph relationships.',
      },
      {
        stepKey: 'scene_graph_proposals',
        name: 'Scene Graph + Proposals',
        detail:
          'Produces the structured scene understanding that temporal validation consumes next.',
      },
    ],
  },
  {
    id: '04',
    title: 'Temporal Validation And Dataset Export',
    summary:
      'Frame-level proposals are fused over time so only sustained, confident interactions become confirmed events and exportable windows.',
    files: ['temporal_events.py', 'dataset_exporter.py', 'kabaddi_afgn_reasoning.py'],
    modules: [
      {
        stepKey: 'multi_frame_accumulation',
        name: 'Multi Frame Accumulation',
        detail:
          'Aggregates the same candidate interaction over consecutive frames instead of making a one-frame decision.',
      },
      {
        stepKey: 'factor_confidence_fusion',
        name: 'Factor Confidence Fusion',
        detail:
          'Combines proposal confidence and factor confidence into a stronger temporal belief score.',
      },
      {
        stepKey: 'fused_confidence_threshold',
        name: 'Fused Confidence Threshold',
        detail:
          'Applies thresholds so only sufficiently strong temporal evidence becomes a confirmed event.',
      },
      {
        stepKey: 'confirmed_events',
        name: 'Confirmed Events',
        detail:
          'Outputs confirmed contact and line-touch events after the temporal checks pass.',
      },
      {
        stepKey: 'dataset_export',
        name: 'Dataset Export',
        detail:
          'Extracts confirmed event windows into reusable clips and metadata for classifier training and review.',
      },
      {
        stepKey: 'clip_export_metadata_json',
        name: 'Clip Export + Metadata JSON',
        detail:
          'Writes the event clip and its JSON payload into the dataset manifest structure.',
      },
    ],
  },
  {
    id: '05',
    title: 'Visual Touch Classifier',
    summary:
      'Confirmed windows are aligned, sampled, encoded, and classified by the learned touch-classifier stack.',
    files: [
      'touch_classifier_dataset.py',
      'touch_classifier_model.py',
      'train_touch_classifier.py',
      'touch_classifier_inference.py',
      'classifier_bridge.py',
    ],
    modules: [
      {
        stepKey: 'frame_buffer_alignment',
        name: 'Frame Buffer Alignment',
        detail:
          'Aligns the exported event window so the classifier sees the correct temporal slice around the interaction.',
      },
      {
        stepKey: 'temporal_frame_sampling',
        name: 'Temporal Frame Sampling',
        detail:
          'Samples the window into a fixed number of frames before model inference or training.',
      },
      {
        stepKey: 'resnet18_frame_encoder',
        name: 'ResNet18 Frame Encoder',
        detail:
          'Extracts visual features independently from each sampled frame.',
      },
      {
        stepKey: 'temporal_average_pool',
        name: 'Temporal Average Pool',
        detail:
          'Aggregates frame features into a single clip-level representation.',
      },
      {
        stepKey: 'mlp_classification_head',
        name: 'MLP Classification Head',
        detail:
          'Converts the pooled visual representation into no-touch vs valid-touch probabilities.',
      },
      {
        stepKey: 'event_confirmation',
        name: 'Event Confirmation',
        detail:
          'Bridge logic maps the classifier output back into valid, invalid, or uncertain event support.',
      },
    ],
  },
  {
    id: '07',
    title: 'Visualization And Outputs',
    summary:
      'The frontend and reporting layers present the processed scene, replay views, and final exported outputs.',
    files: ['App.jsx', 'RaidReplay3D.jsx', 'report_video.py', 'api_server.py'],
    modules: [
      {
        stepKey: 'visualization',
        name: 'Visualization',
        detail:
          'Shows 3D bbox overlays, 3D event view, 2D court map, score display, and event title/classifier cards.',
      },
      {
        stepKey: 'processed_video',
        name: 'Processed Video',
        detail:
          'Generates the tracked video with overlays from the live backend pipeline.',
      },
      {
        stepKey: 'event_report_clips',
        name: 'Event Report Clips',
        detail:
          'Exports short clips around confirmed events for review and classifier analysis.',
      },
      {
        stepKey: 'event_logs',
        name: 'Event Logs',
        detail:
          'Stores machine-readable records of confirmed events, confidences, and classifier decisions.',
      },
      {
        stepKey: 'score_card',
        name: 'Score Card',
        detail:
          'Presents the resulting raid outcomes and score progression for the match flow.',
      },
    ],
  },
]

const MODULE_SEQUENCE = STAGES.flatMap((stage) =>
  stage.modules.map((module) => ({
    stageId: stage.id,
    stepKey: module.stepKey,
    name: module.name,
  })),
)

function StageCard({ stage, currentStepKey, currentStepIndex }) {
  const completedCount = stage.modules.filter((module) => {
    const moduleIndex = MODULE_SEQUENCE.findIndex((item) => item.stepKey === module.stepKey)
    return currentStepIndex > moduleIndex
  }).length

  return (
    <article
      className="relative flex h-[500px] flex-col overflow-hidden rounded-[28px] border border-stone-200 bg-white/90 p-5 shadow-sm backdrop-blur transition-all duration-300 dark:border-stone-800 dark:bg-stone-900/70 dark:shadow-none"
    >
      <div className="absolute right-4 top-4 rounded-full border border-stone-200 bg-stone-50 px-2.5 py-1 text-[11px] font-semibold tracking-[0.2em] text-stone-500 dark:border-stone-700 dark:bg-stone-950/50 dark:text-stone-400">
        {stage.id}
      </div>
      <div className="max-w-[88%]">
        <h3 className="text-lg font-semibold tracking-tight text-stone-900 dark:text-stone-50">
          {stage.title}
        </h3>
        <p className="mt-2 text-sm leading-6 text-stone-600 dark:text-stone-300">
          {stage.summary}
        </p>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {stage.files.map((file) => (
          <span
            key={file}
            className="rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-[11px] font-medium text-stone-700 dark:border-stone-700 dark:bg-stone-950/60 dark:text-stone-200"
          >
            {file}
          </span>
        ))}
        <span className="rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-[11px] font-medium text-stone-700 dark:border-stone-700 dark:bg-stone-950/60 dark:text-stone-200">
          {completedCount}/{stage.modules.length} complete
        </span>
      </div>

      <div className="mt-4 min-h-0 flex-1 overflow-hidden">
        <div className="h-full space-y-3 overflow-y-auto pr-1">
          {stage.modules.map((module) => {
            const moduleIndex = MODULE_SEQUENCE.findIndex((item) => item.stepKey === module.stepKey)
            const isActive = currentStepKey === module.stepKey
            const isComplete = currentStepIndex > moduleIndex

            return (
              <div
                key={module.name}
                className={`rounded-2xl border px-4 py-3 transition-all duration-300 ${
                  isActive
                    ? 'border-amber-300 bg-stone-50 shadow-[0_0_0_1px_rgba(251,191,36,0.32),0_0_28px_rgba(251,191,36,0.32)] dark:border-amber-500/70 dark:bg-stone-950/45'
                    : isComplete
                      ? 'border-emerald-200 bg-stone-50 dark:border-emerald-900/40 dark:bg-stone-950/45'
                      : 'border-transparent bg-stone-50 dark:bg-stone-950/45'
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold text-stone-900 dark:text-stone-50">
                    {module.name}
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] ${
                      isActive
                        ? 'bg-amber-200 text-amber-900 dark:bg-amber-500/20 dark:text-amber-100'
                        : isComplete
                          ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-100'
                          : 'bg-stone-200 text-stone-600 dark:bg-stone-800 dark:text-stone-300'
                    }`}
                  >
                    {isActive ? 'Live' : isComplete ? 'Done' : 'Queued'}
                  </span>
                </div>
                <div className="mt-1 text-sm leading-6 text-stone-600 dark:text-stone-300">
                  {module.detail}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </article>
  )
}

function ArchitecturePage({ pipelineStep = null, showLive = false }) {
  const [displayStep, setDisplayStep] = useState(pipelineStep)
  const transitionTimerRef = useRef(null)

  useEffect(() => {
    const nextKey = pipelineStep?.step_key || null
    const shownKey = displayStep?.step_key || null

    if (nextKey === shownKey) {
      if (transitionTimerRef.current) {
        window.clearTimeout(transitionTimerRef.current)
        transitionTimerRef.current = null
      }
      if (pipelineStep && pipelineStep !== displayStep) {
        setDisplayStep(pipelineStep)
      }
      return undefined
    }

    if (!shownKey || !nextKey) {
      setDisplayStep(pipelineStep)
      return undefined
    }

    if (transitionTimerRef.current) {
      window.clearTimeout(transitionTimerRef.current)
    }

    transitionTimerRef.current = window.setTimeout(() => {
      setDisplayStep(pipelineStep)
      transitionTimerRef.current = null
    }, STEP_LINGER_MS)

    return () => {
      if (transitionTimerRef.current) {
        window.clearTimeout(transitionTimerRef.current)
        transitionTimerRef.current = null
      }
    }
  }, [pipelineStep, displayStep])

  useEffect(() => {
    return () => {
      if (transitionTimerRef.current) {
        window.clearTimeout(transitionTimerRef.current)
      }
    }
  }, [])

  const currentStepKey = displayStep?.step_key || null
  const currentStepIndex = MODULE_SEQUENCE.findIndex((item) => item.stepKey === currentStepKey)

  return (
    <main className="w-full px-3 pb-10 sm:px-4 lg:px-5">
      <section className="p-6">
        <div className="w-full text-center">
          <div className="flex flex-col items-center">
            <div className="inline-flex rounded-full border border-stone-200 bg-stone-50 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-stone-500 dark:border-stone-700 dark:bg-stone-950/60 dark:text-stone-400">
              Project Architecture
            </div>
            <h2 className="mt-4 w-full text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-50 sm:text-4xl">
              Real-time workflow from kabaddi raid input to validated output.
            </h2>
            <p className="mt-4 w-full text-sm leading-7 text-stone-600 dark:text-stone-300">
              This page follows the same structure as the architecture diagram: raw video enters the system, players are
              detected and tracked, the raider and interaction graph are inferred, events are validated over time, the
              visual classifier confirms touch evidence, and the frontend exposes the final outputs.
            </p>

            <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
              <span
                className={`rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] ${
                  showLive
                    ? 'border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-100'
                    : 'border-stone-200 bg-stone-50 text-stone-600 dark:border-stone-700 dark:bg-stone-950/60 dark:text-stone-300'
                }`}
              >
                {showLive ? 'Live Pipeline' : 'Waiting For Live Pipeline'}
              </span>
              {displayStep?.stage_id ? (
                <span className="rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-[11px] font-semibold text-amber-800 dark:border-amber-700 dark:bg-amber-950/25 dark:text-amber-100">
                  Stage {displayStep.stage_id}
                </span>
              ) : null}
              {displayStep?.module_name ? (
                <span className="rounded-full border border-stone-200 bg-white px-3 py-1 text-[11px] font-medium text-stone-700 dark:border-stone-700 dark:bg-stone-950/60 dark:text-stone-200">
                  {displayStep.module_name}
                </span>
              ) : null}
              {displayStep?.frame_idx != null ? (
                <span className="rounded-full border border-stone-200 bg-white px-3 py-1 text-[11px] font-medium text-stone-700 dark:border-stone-700 dark:bg-stone-950/60 dark:text-stone-200">
                  frame {displayStep.frame_idx}
                </span>
              ) : null}
            </div>
          </div>
        </div>
      </section>

      <section className="mt-5 overflow-hidden rounded-[32px] border border-stone-200 bg-white/90 p-4 shadow-sm backdrop-blur dark:border-stone-800 dark:bg-stone-900/70 dark:shadow-none">
        <div className="mb-3">
          <h3 className="text-lg font-semibold tracking-tight text-stone-900 dark:text-stone-50">
            Architecture Diagram
          </h3>
          <p className="mt-1 text-sm text-stone-600 dark:text-stone-300">
            Reference image for the full system architecture and module flow.
          </p>
        </div>
        <div className="overflow-hidden rounded-[24px] border border-stone-200 bg-stone-50 dark:border-stone-700 dark:bg-stone-950/50">
          <img
            src="/architecture A3.jpg"
            alt="Kabaddi project system architecture"
            className="h-auto w-full object-contain"
            loading="eager"
            decoding="async"
          />
        </div>
      </section>

      <section className="mt-5">
        <div className="mb-4">
          <h3 className="text-xl font-semibold tracking-tight text-stone-900 dark:text-stone-50">
            Module Details
          </h3>
          <p className="mt-1 text-sm text-stone-600 dark:text-stone-300">
            Detailed explanation of each module block shown in the architecture image above.
          </p>
        </div>
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
          {STAGES.map((stage) => (
            <StageCard
              key={stage.id}
              stage={stage}
              currentStepKey={currentStepKey}
              currentStepIndex={currentStepIndex}
            />
          ))}
        </div>
      </section>
    </main>
  )
}

export default ArchitecturePage
