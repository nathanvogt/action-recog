import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { TrainDatasetRemote } from "../../libs/trainDataset/trainDataset.js";
import {
  CONNECTIONS,
  LEFT_LEG_NO_FEET,
  RIGHT_LEG_NO_FEET,
  LEFT_ARM_NO_HAND,
  RIGHT_ARM_NO_HAND,
  BACK,
} from "../../libs/data.js";
import { SlsBasic } from "../../libs/sls/SlsBasic.js";
import { Point } from "../../libs/sls/slsTypes.js";
import { SlsMemoized } from "../../libs/sls/SlsMemoized.js";

export const VisualizePose: React.FC = () => {
  const { subject_id, exercise_name } = useParams<{
    subject_id: string;
    exercise_name: string;
  }>();
  const navigate = useNavigate();

  const [poseData, setPoseData] = useState<[number, number, number][][] | null>(
    null
  );
  const [repTimings, setRepTimings] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const remote = new TrainDatasetRemote("http://localhost:3001");

  useEffect(() => {
    const loadPoseData = async () => {
      if (!subject_id || !exercise_name) return;

      setError(null);
      setPoseData(null);
      setRepTimings(null);

      try {
        const [poses, timings] = await Promise.all([
          remote.getPoseArray(subject_id, exercise_name),
          remote.getRepTimings(subject_id, exercise_name),
        ]);
        setPoseData(poses);
        setRepTimings(timings);
      } catch (err) {
        setError(
          err instanceof Error
            ? `Failed to load pose data: ${err.message}`
            : "Failed to load pose data"
        );
      }
    };

    loadPoseData();
  }, [subject_id, exercise_name]);

  if (!subject_id || !exercise_name) {
    return (
      <div className="error-container p-4">
        <h1 className="text-red-600 text-xl font-bold">Error</h1>
        <p className="text-red-500">
          Missing required parameters. Both subject ID and exercise name are
          required.
        </p>
        <p className="text-gray-600 text-sm mt-2">
          Expected URL format: /visualize/[subject_id]/[exercise_name]
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container p-4">
        <h1 className="text-red-600 text-xl font-bold">Error</h1>
        <p className="text-red-500">{error}</p>
        <p className="text-gray-600 text-sm mt-2">
          Subject: {subject_id}, Exercise: {exercise_name}
        </p>
      </div>
    );
  }

  if (!poseData) {
    return (
      <div className="p-4">
        <h1 className="text-xl font-bold">Loading Pose Data...</h1>
        <p className="text-gray-600">
          Loading pose data for {subject_id} - {exercise_name}
        </p>
      </div>
    );
  }

  return (
    <_VisualizePose
      subjectId={subject_id}
      exerciseName={exercise_name}
      poseData={poseData}
      repTimings={repTimings}
      navigate={navigate}
    />
  );
};

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

const MemoPoint: React.FC<{
  position: [number, number, number];
  opacity: number;
}> = ({ position, opacity }) => {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.015, 16, 16]} />
      <meshStandardMaterial color="#1E3A8A" transparent opacity={opacity} />
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

type Props = {
  exerciseName: string;
  subjectId: string;
  poseData: [number, number, number][][];
  repTimings: number[] | null;
  navigate: (path: string) => void;
};

const _VisualizePose: React.FC<Props> = ({
  exerciseName,
  subjectId,
  poseData,
  repTimings,
  navigate,
}) => {
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showHistory, setShowHistory] = useState(true);
  const [historyAnchor, setHistoryAnchor] = useState(0);
  const [showSls, setShowSls] = useState(false);
  const c = 32;
  const m = 8;
  const [slsProcessor] = useState(() => new SlsMemoized(c, m));
  const [lastProcessedRange, setLastProcessedRange] = useState<{
    start: number;
    end: number;
  } | null>(null);

  const getCurrentRep = () => {
    if (!repTimings || repTimings.length === 0) return null;

    for (let i = 0; i < repTimings.length - 1; i++) {
      if (
        currentFrameIndex >= repTimings[i] &&
        currentFrameIndex < repTimings[i + 1]
      ) {
        return i + 1;
      }
    }

    if (currentFrameIndex >= repTimings[repTimings.length - 1]) {
      return repTimings.length;
    }

    return null;
  };

  const jumpToRep = (repIndex: number) => {
    if (!repTimings || repIndex < 0 || repIndex >= repTimings.length) return;

    if (isPlaying) {
      setIsPlaying(false);
    }

    setCurrentFrameIndex(repTimings[repIndex]);
  };

  const resetHistoryAnchor = () => {
    setHistoryAnchor(currentFrameIndex);
    // The useEffect will handle resetting the processor and range
  };

  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentFrameIndex((prev) => (prev + 1) % poseData.length);
    }, 1000 / 30);

    return () => clearInterval(interval);
  }, [isPlaying, poseData.length]);

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === " ") {
        event.preventDefault();
        setIsPlaying((prev) => !prev);
      } else if (event.key === "ArrowRight") {
        if (isPlaying) {
          setIsPlaying(false);
        }
        setCurrentFrameIndex((prev) => (prev + 1) % poseData.length);
      } else if (event.key === "ArrowLeft") {
        if (isPlaying) {
          setIsPlaying(false);
        }
        setCurrentFrameIndex(
          (prev) => (prev - 1 + poseData.length) % poseData.length
        );
      } else if (event.key === "r" || event.key === "R") {
        event.preventDefault();
        resetHistoryAnchor();
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [poseData.length, isPlaying, currentFrameIndex]);

  const currentFrame: [number, number, number][] = poseData[currentFrameIndex];
  const currentRep = getCurrentRep();

  const getFramesToRender = () => {
    if (!showHistory) {
      return [{ frame: currentFrame, index: currentFrameIndex, opacity: 1.0 }];
    }

    const frames = [];
    const startIndex = Math.min(historyAnchor, currentFrameIndex);
    const endIndex = Math.max(historyAnchor, currentFrameIndex);

    for (let i = startIndex; i <= endIndex; i++) {
      const isCurrentFrame = i === currentFrameIndex;
      const opacity = isCurrentFrame ? 1.0 : 0.2;
      frames.push({
        frame: poseData[i],
        index: i,
        opacity: opacity,
      });
    }

    return frames;
  };

  const framesToRender = getFramesToRender();
  const historyWindowSize = showHistory
    ? Math.abs(currentFrameIndex - historyAnchor) + 1
    : 1;

  const [slsResult, setSlsResult] = useState<[Point[][], number] | null>(null);
  const [memoPoints, setMemoPoints] = useState<Point[][] | null>(null);

  // Effect to handle SLS processing
  useEffect(() => {
    if (!showSls) {
      setLastProcessedRange(null);
      setSlsResult(null);
      setMemoPoints(null);
      return;
    }

    const startIndex = Math.min(historyAnchor, currentFrameIndex);
    const endIndex = Math.max(historyAnchor, currentFrameIndex);

    // If anchor changed or this is the first time, reset everything
    if (!lastProcessedRange || lastProcessedRange.start !== startIndex) {
      slsProcessor.reset();
      setLastProcessedRange({ start: startIndex, end: startIndex - 1 });
      return; // Let the next effect run handle the processing
    }

    const newFrames: [number, number, number][][] = [];
    const startProcessingFrom = lastProcessedRange
      ? Math.max(lastProcessedRange.end + 1, startIndex)
      : startIndex;

    for (let i = startProcessingFrom; i <= endIndex; i++) {
      newFrames.push(poseData[i]);
    }

    if (newFrames.length === 0) {
      return;
    }

    if (endIndex - startIndex + 1 < 2) {
      setSlsResult(null);
      return;
    }

    try {
      const [result, totalError] = slsProcessor.processPoses(newFrames);
      setLastProcessedRange({ start: startIndex, end: endIndex });
      setSlsResult([result, totalError]);
      setMemoPoints(slsProcessor.getMemPoints());
    } catch (error) {
      console.error("Error computing SLS representation:", error);
      setSlsResult(null);
      setMemoPoints(null);
    }
  }, [showSls, historyAnchor, currentFrameIndex, lastProcessedRange]);

  const getSlsRepresentation = (): [Point[][], number] | [null, null] => {
    return slsResult || [null, null];
  };

  const [slsRepresentation, totalError] = getSlsRepresentation();

  return (
    <div>
      <div className="flex items-center justify-between mb-3 p-2 bg-gray-50 rounded">
        <div className="flex items-center space-x-4 text-sm">
          <button
            onClick={() => navigate("/")}
            className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
          >
            ← Main Menu
          </button>
          <span>
            <strong>{subjectId}</strong> - {exerciseName}
          </span>
          <span>
            Frame {currentFrameIndex + 1}/{poseData.length}
          </span>
          <span className={isPlaying ? "text-green-600" : "text-gray-600"}>
            {isPlaying ? "Playing" : "Paused"}
          </span>
          {currentRep && (
            <span className="text-blue-600">Rep {currentRep}</span>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between mb-3 p-2 bg-blue-50 rounded border">
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-1 text-sm">
            <input
              type="checkbox"
              checked={showHistory}
              onChange={(e) => setShowHistory(e.target.checked)}
              className="rounded"
            />
            <span>History</span>
          </label>

          <label className="flex items-center space-x-1 text-sm">
            <input
              type="checkbox"
              checked={showSls}
              onChange={(e) => setShowSls(e.target.checked)}
              className="rounded"
            />
            <span>SLS</span>
          </label>

          {showSls && totalError !== null && (
            <div className="flex items-center space-x-1 text-sm text-blue-600">
              <span>Error:</span>
              <span className="font-mono">{totalError.toFixed(4)}</span>
            </div>
          )}

          {showHistory && (
            <>
              <div className="flex items-center space-x-1 text-sm">
                <span>Anchor:</span>
                <input
                  type="number"
                  min="0"
                  max={poseData.length - 1}
                  value={historyAnchor}
                  onChange={(e) =>
                    setHistoryAnchor(
                      Math.max(
                        0,
                        Math.min(
                          poseData.length - 1,
                          parseInt(e.target.value) || 0
                        )
                      )
                    )
                  }
                  className="w-16 px-1 py-0.5 text-sm border rounded"
                />
                <button
                  onClick={resetHistoryAnchor}
                  className="px-2 py-0.5 text-xs bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Reset
                </button>
              </div>
              <span className="text-xs text-gray-600">
                ({historyWindowSize} frame{historyWindowSize !== 1 ? "s" : ""})
              </span>
            </>
          )}
        </div>

        <div className="text-xs text-gray-600">
          ← → Space R{showSls && " • SLS: yellow • Memo: dark blue"}
        </div>
      </div>

      {repTimings && repTimings.length > 1 && (
        <div className="mb-3 p-2 bg-gray-50 rounded">
          <div className="flex items-center space-x-2 text-sm">
            <span className="font-medium">Reps ({repTimings.length - 1}):</span>
            <div className="flex flex-wrap gap-1">
              {repTimings.slice(0, -1).map((startFrame, index) => {
                const endFrame = repTimings[index + 1];
                const isCurrentRep = currentRep === index + 1;
                return (
                  <button
                    key={index}
                    onClick={() => jumpToRep(index)}
                    className={`px-2 py-0.5 text-xs rounded cursor-pointer transition-colors ${
                      isCurrentRep
                        ? "bg-blue-500 text-white"
                        : "bg-gray-200 text-gray-700 hover:bg-blue-100"
                    }`}
                    title={`Frames ${startFrame}-${endFrame - 1}`}
                  >
                    {index + 1}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      <div
        style={{
          width: "100%",
          height: "800px",
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

          {framesToRender.map((frameData, frameIdx) => (
            <group key={`frame-${frameData.index}`}>
              {frameData.frame.map((point, pointIndex) => (
                <PosePoint
                  key={`${frameData.index}-${pointIndex}`}
                  position={point}
                  opacity={frameData.opacity}
                />
              ))}
              <PoseConnections
                points={frameData.frame}
                opacity={frameData.opacity}
              />
            </group>
          ))}

          {slsRepresentation && (
            <group key="sls-representation">
              {slsRepresentation.map((curve, curveIndex) => (
                <group key={`sls-curve-${curveIndex}`}>
                  {curve.map((point, pointIndex) => (
                    <SlsPoint
                      key={`sls-${curveIndex}-${pointIndex}`}
                      position={point}
                      opacity={0.6}
                    />
                  ))}
                  <SlsTrajectory points={curve} opacity={0.6} />
                </group>
              ))}
            </group>
          )}

          {memoPoints && (
            <group key="memo-points">
              {memoPoints.map((keypointMemoPoints, keypointIndex) => (
                <group key={`memo-keypoint-${keypointIndex}`}>
                  {keypointMemoPoints.map((point, pointIndex) => (
                    <MemoPoint
                      key={`memo-${keypointIndex}-${pointIndex}`}
                      position={point}
                      opacity={0.8}
                    />
                  ))}
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
    </div>
  );
};
