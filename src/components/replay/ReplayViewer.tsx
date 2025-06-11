import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import {
  ReplayClient,
  ReplayData,
  ReplayEpisode,
  ReplayStep,
} from "../../libs/replayClient";
import { TrainDatasetRemote } from "../../libs/trainDataset/trainDataset.js";
import { CONNECTIONS } from "../../libs/data.js";

interface ReplayViewerParams extends Record<string, string | undefined> {
  filename: string;
}

const PosePoint: React.FC<{
  position: [number, number, number];
  opacity: number;
}> = ({ position, opacity }) => {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.015, 16, 16]} />
      <meshStandardMaterial color="red" transparent opacity={opacity} />
    </mesh>
  );
};

const PoseConnections: React.FC<{
  points: [number, number, number][];
  opacity: number;
}> = ({ points, opacity }) => {
  return (
    <>
      {CONNECTIONS.map(([fromIndex, toIndex], connectionIndex) => {
        if (!points[fromIndex] || !points[toIndex]) {
          return null;
        }

        const fromPoint = points[fromIndex];
        const toPoint = points[toIndex];

        return (
          <line key={connectionIndex}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                args={[
                  new Float32Array([
                    fromPoint[0],
                    fromPoint[1],
                    fromPoint[2],
                    toPoint[0],
                    toPoint[1],
                    toPoint[2],
                  ]),
                  3,
                ]}
              />
            </bufferGeometry>
            <lineBasicMaterial
              color="#87CEEB"
              linewidth={2}
              transparent
              opacity={opacity}
            />
          </line>
        );
      })}
    </>
  );
};

const SlsPoint: React.FC<{
  position: [number, number, number];
  opacity: number;
}> = ({ position, opacity }) => {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.02, 16, 16]} />
      <meshStandardMaterial color="#FFFF80" transparent opacity={opacity} />
    </mesh>
  );
};

const SlsTrajectory: React.FC<{
  points: [number, number, number][];
  opacity: number;
}> = ({ points, opacity }) => {
  return (
    <>
      {points.map((point, index) => {
        if (index < points.length - 1) {
          const nextPoint = points[index + 1];

          const curve = new THREE.LineCurve3(
            new THREE.Vector3(point[0], point[1], point[2]),
            new THREE.Vector3(nextPoint[0], nextPoint[1], nextPoint[2])
          );

          return (
            <mesh key={index}>
              <tubeGeometry args={[curve, 2, 0.003, 8, false]} />
              <meshStandardMaterial
                color="#FFFF80"
                transparent
                opacity={opacity}
              />
            </mesh>
          );
        }
        return null;
      })}
    </>
  );
};

const Grid: React.FC = () => {
  const size = 2;
  const divisions = 20;
  const half = size / 2;
  const step = size / divisions;
  const color = "#888888";

  const lines = [];

  for (let i = 0; i <= divisions; i++) {
    const y = -half + i * step;
    lines.push(
      <line key={`row-${i}`}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([-half, y, 0, half, y, 0]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color={color} />
      </line>
    );
  }

  for (let i = 0; i <= divisions; i++) {
    const x = -half + i * step;
    lines.push(
      <line key={`col-${i}`}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([x, -half, 0, x, half, 0]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color={color} />
      </line>
    );
  }

  return <>{lines}</>;
};

export const ReplayViewer: React.FC = () => {
  const { filename } = useParams<ReplayViewerParams>();
  const navigate = useNavigate();

  const [replayData, setReplayData] = useState<ReplayData | null>(null);
  const [poseData, setPoseData] = useState<[number, number, number][][] | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Replay state
  const [currentEpisode, setCurrentEpisode] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showSls, setShowSls] = useState(true);

  const replayClient = new ReplayClient();
  const datasetClient = new TrainDatasetRemote("http://localhost:3001");

  useEffect(() => {
    const loadReplayData = async () => {
      if (!filename) return;

      try {
        setLoading(true);
        setError(null);

        const replay = await replayClient.getReplay(filename);
        setReplayData(replay);

        // Load pose data for the subject/exercise
        const poses = await datasetClient.getPoseArray(
          replay.metadata.subject,
          replay.metadata.exercise
        );
        setPoseData(poses);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to load replay data"
        );
      } finally {
        setLoading(false);
      }
    };

    loadReplayData();
  }, [filename]);

  // Auto-play functionality
  useEffect(() => {
    if (!isPlaying || !replayData) return;

    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        const currentEpisodeData = replayData.episodes[currentEpisode];
        if (prev >= currentEpisodeData.steps.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 33); // 30 FPS for replay

    return () => clearInterval(interval);
  }, [isPlaying, replayData, currentEpisode]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (!replayData) return;

      const currentEpisodeData = replayData.episodes[currentEpisode];

      if (event.key === " ") {
        event.preventDefault();
        setIsPlaying((prev) => !prev);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        if (isPlaying) setIsPlaying(false);
        setCurrentStep((prev) =>
          Math.min(prev + 1, currentEpisodeData.steps.length - 1)
        );
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        if (isPlaying) setIsPlaying(false);
        setCurrentStep((prev) => Math.max(prev - 1, 0));
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [replayData, currentEpisode, isPlaying]);

  const getCurrentPose = (): [number, number, number][] | null => {
    if (!replayData || !poseData) return null;

    const currentEpisodeData = replayData.episodes[currentEpisode];
    const stepData = currentEpisodeData.steps[currentStep];

    if (!stepData) return null;

    // The observation should contain pose data - we need to map it to the pose format
    // For now, let's use the timestep to index into the pose data
    const poseIndex = stepData.timestep - 1; // timestep is 1-indexed
    return poseData[poseIndex] || null;
  };

  const getSlsFromObservation = (): [number, number, number][][] | null => {
    if (!replayData) return null;

    const currentEpisodeData = replayData.episodes[currentEpisode];
    const stepData = currentEpisodeData.steps[currentStep];

    if (!stepData || !stepData.observation) return null;

    const observation = stepData.observation;
    const c = replayData.metadata.env_config.c;

    // Default keypoints used in the environment (matching the order in gym_env.py)
    const keypoints = [4, 5, 6, 1, 2, 3, 14, 15, 16, 11, 12, 13, 0, 7]; // LEFT_LEG_NO_FEET + RIGHT_LEG_NO_FEET + LEFT_ARM_NO_HAND + RIGHT_ARM_NO_HAND + BACK

    if (observation.length !== keypoints.length * c * 3) {
      console.warn(
        "Observation length mismatch:",
        observation.length,
        "expected:",
        keypoints.length * c * 3
      );
      return null;
    }

    const curves: [number, number, number][][] = [];

    for (let kpIdx = 0; kpIdx < keypoints.length; kpIdx++) {
      const curve: [number, number, number][] = [];
      const startIdx = kpIdx * c * 3;

      for (let pointIdx = 0; pointIdx < c; pointIdx++) {
        const pointStartIdx = startIdx + pointIdx * 3;
        const x = observation[pointStartIdx];
        const y = observation[pointStartIdx + 1];
        const z = observation[pointStartIdx + 2];

        // Add all points from the SLS curve (no zero-padding filtering needed since curves are guaranteed to have c points)
        curve.push([x, y, z]);
      }

      curves.push(curve);
    }

    return curves;
  };

  const getActionName = (action: number): string => {
    // Map action numbers to human-readable names
    const actionNames = {
      0: "No Action",
      1: "Count Rep",
      // Add more action mappings as needed
    };
    return (
      actionNames[action as keyof typeof actionNames] || `Action ${action}`
    );
  };

  if (!filename) {
    return (
      <div className="error-container p-4">
        <h1 className="text-red-600 text-xl font-bold">Error</h1>
        <p className="text-red-500">No replay filename provided.</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="p-4">
        <h1 className="text-xl font-bold">Loading Replay...</h1>
        <p className="text-gray-600">Loading replay data for {filename}</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container p-4">
        <h1 className="text-red-600 text-xl font-bold">Error</h1>
        <p className="text-red-500">{error}</p>
        <button
          onClick={() => navigate("/replays")}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          ← Back to Replays
        </button>
      </div>
    );
  }

  if (!replayData) {
    return (
      <div className="error-container p-4">
        <h1 className="text-red-600 text-xl font-bold">Error</h1>
        <p className="text-red-500">No replay data available.</p>
      </div>
    );
  }

  const currentEpisodeData = replayData.episodes[currentEpisode];
  const currentStepData = currentEpisodeData.steps[currentStep];
  const currentPose = getCurrentPose();
  const slsCurves = showSls ? getSlsFromObservation() : null;

  return (
    <div className="p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate("/replays")}
            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          >
            ← Back to Replays
          </button>
          <h1 className="text-xl font-bold">
            {replayData.metadata.subject} - {replayData.metadata.exercise}
          </h1>
        </div>
        <div className="text-sm text-gray-600">{filename}</div>
      </div>

      {/* Episode Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div className="bg-white p-4 rounded border">
          <h3 className="font-semibold text-gray-700 mb-2">Episode Info</h3>
          <div className="space-y-1 text-sm">
            <div>
              Episode: {currentEpisodeData.episode_number} of{" "}
              {replayData.episodes.length}
            </div>
            <div>
              Total Reward: {currentEpisodeData.total_reward.toFixed(2)}
            </div>
            <div>Length: {currentEpisodeData.episode_length} steps</div>
          </div>
        </div>

        <div className="bg-white p-4 rounded border">
          <h3 className="font-semibold text-gray-700 mb-2">Current Step</h3>
          <div className="space-y-1 text-sm">
            <div>
              Step: {currentStep + 1} of {currentEpisodeData.steps.length}
            </div>
            <div>Timestep: {currentStepData?.timestep || "N/A"}</div>
            <div>
              Action:{" "}
              {currentStepData ? getActionName(currentStepData.action) : "N/A"}
            </div>
            <div
              className={`font-semibold ${
                (currentStepData?.reward || 0) >= 0
                  ? "text-green-600"
                  : "text-red-600"
              }`}
            >
              Reward: {currentStepData?.reward.toFixed(3) || "N/A"}
            </div>
          </div>
        </div>

        <div className="bg-white p-4 rounded border">
          <h3 className="font-semibold text-gray-700 mb-2">
            Environment Config
          </h3>
          <div className="space-y-1 text-sm">
            <div>c: {replayData.metadata.env_config.c}</div>
            <div>m: {replayData.metadata.env_config.m}</div>
            <div>tol: {replayData.metadata.env_config.tol}</div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mb-4 p-3 bg-blue-50 rounded border">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`px-4 py-2 rounded font-medium ${
              isPlaying
                ? "bg-red-500 text-white hover:bg-red-600"
                : "bg-green-500 text-white hover:bg-green-600"
            }`}
          >
            {isPlaying ? "Pause" : "Play"}
          </button>

          <button
            onClick={() => {
              setCurrentStep(0);
              setIsPlaying(false);
            }}
            className="px-3 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            Reset
          </button>

          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium">Episode:</label>
            <select
              value={currentEpisode}
              onChange={(e) => {
                setCurrentEpisode(parseInt(e.target.value));
                setCurrentStep(0);
                setIsPlaying(false);
              }}
              className="px-2 py-1 border rounded"
            >
              {replayData.episodes.map((_, index) => (
                <option key={index} value={index}>
                  {index + 1} (Reward:{" "}
                  {replayData.episodes[index].total_reward.toFixed(1)})
                </option>
              ))}
            </select>
          </div>

          <label className="flex items-center space-x-1 text-sm">
            <input
              type="checkbox"
              checked={showSls}
              onChange={(e) => setShowSls(e.target.checked)}
              className="rounded"
            />
            <span>Show SLS</span>
          </label>
        </div>

        <div className="text-sm text-gray-600">
          ← → Space (Step/Play) •{" "}
          {showSls ? "Yellow: SLS curves" : "Red: Original pose"}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-600">Progress:</span>
          <div className="flex-1 bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-200"
              style={{
                width: `${
                  ((currentStep + 1) / currentEpisodeData.steps.length) * 100
                }%`,
              }}
            />
          </div>
          <span className="text-sm text-gray-600">
            {Math.round(
              ((currentStep + 1) / currentEpisodeData.steps.length) * 100
            )}
            %
          </span>
        </div>
      </div>

      {/* 3D Visualization */}
      <div
        style={{
          width: "100%",
          height: "600px",
          border: "2px solid #ccc",
          borderRadius: "8px",
        }}
      >
        <Canvas
          camera={{ position: [2, 2, 2], fov: 50 }}
          onCreated={({ camera }) => {
            camera.up.set(0, 0, 1);
          }}
        >
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />

          <Grid />

          {!showSls && currentPose && (
            <group>
              {currentPose.map((point, pointIndex) => (
                <PosePoint key={pointIndex} position={point} opacity={1.0} />
              ))}
              <PoseConnections points={currentPose} opacity={1.0} />
            </group>
          )}

          {showSls && slsCurves && (
            <group>
              {slsCurves.map((curve, curveIndex) => (
                <group key={`sls-curve-${curveIndex}`}>
                  {curve.map((point, pointIndex) => (
                    <SlsPoint
                      key={`sls-${curveIndex}-${pointIndex}`}
                      position={point}
                      opacity={1.0}
                    />
                  ))}
                  <SlsTrajectory points={curve} opacity={0.8} />
                </group>
              ))}
            </group>
          )}

          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
          />
        </Canvas>
      </div>

      {/* Step Details */}
      {currentStepData && (
        <div className="mt-4 p-4 bg-gray-50 rounded">
          <h3 className="font-semibold mb-2">Step Details</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <strong>Terminated:</strong>{" "}
              {currentStepData.terminated ? "Yes" : "No"}
            </div>
            <div>
              <strong>Truncated:</strong>{" "}
              {currentStepData.truncated ? "Yes" : "No"}
            </div>
            {currentStepData.info &&
              Object.keys(currentStepData.info).length > 0 && (
                <div className="col-span-2">
                  <strong>Info:</strong>
                  <pre className="mt-1 p-2 bg-white rounded text-xs overflow-auto max-h-20">
                    {JSON.stringify(currentStepData.info, null, 2)}
                  </pre>
                </div>
              )}
          </div>
        </div>
      )}
    </div>
  );
};
