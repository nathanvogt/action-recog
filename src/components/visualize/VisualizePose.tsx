import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { TrainDatasetRemote } from "../../libs/trainDataset/trainDataset.js";
import { CONNECTIONS } from "../../libs/data.js";

export const VisualizePose: React.FC = () => {
  const { subject_id, exercise_name } = useParams<{
    subject_id: string;
    exercise_name: string;
  }>();

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
    />
  );
};

type Props = {
  exerciseName: string;
  subjectId: string;
  poseData: [number, number, number][][];
  repTimings: number[] | null;
};

const PosePoint: React.FC<{ position: [number, number, number] }> = ({
  position,
}) => {
  return (
    <mesh position={position}>
      <sphereGeometry args={[0.02, 16, 16]} />
      <meshStandardMaterial color="red" />
    </mesh>
  );
};

const PoseConnections: React.FC<{
  points: [number, number, number][];
}> = ({ points }) => {
  return (
    <>
      {CONNECTIONS.map(([fromIndex, toIndex], connectionIndex) => {
        // Check if both points exist in the current frame
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
            <lineBasicMaterial color="red" linewidth={2} />
          </line>
        );
      })}
    </>
  );
};

const Grid: React.FC = () => {
  const size = 2; // total width/height of the grid
  const divisions = 20; // number of squares per side
  const half = size / 2;
  const step = size / divisions;
  const color = "#888888";

  const lines = [];

  // lines parallel to X-axis (vary x, constant z = 0)
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

  // lines parallel to Y-axis (vary y, constant z = 0)
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

const _VisualizePose: React.FC<Props> = ({
  exerciseName,
  subjectId,
  poseData,
  repTimings,
}) => {
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Function to determine which rep the current frame belongs to
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

    // Check if we're in the last rep
    if (currentFrameIndex >= repTimings[repTimings.length - 1]) {
      return repTimings.length;
    }

    return null; // Frame is before the first rep
  };

  // Function to jump to the beginning of a specific rep
  const jumpToRep = (repIndex: number) => {
    if (!repTimings || repIndex < 0 || repIndex >= repTimings.length) return;

    // Pause playback if currently playing
    if (isPlaying) {
      setIsPlaying(false);
    }

    // Set frame to the start of the specified rep
    setCurrentFrameIndex(repTimings[repIndex]);
  };

  // Auto-advance frames when playing
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentFrameIndex((prev) => (prev + 1) % poseData.length);
    }, 1000 / 30); // 30 fps

    return () => clearInterval(interval);
  }, [isPlaying, poseData.length]);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === " ") {
        event.preventDefault(); // Prevent page scroll
        setIsPlaying((prev) => !prev);
      } else if (event.key === "ArrowRight") {
        if (isPlaying) {
          setIsPlaying(false); // Pause first
        }
        setCurrentFrameIndex((prev) => (prev + 1) % poseData.length);
      } else if (event.key === "ArrowLeft") {
        if (isPlaying) {
          setIsPlaying(false); // Pause first
        }
        setCurrentFrameIndex(
          (prev) => (prev - 1 + poseData.length) % poseData.length
        );
      }
    };

    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [poseData.length, isPlaying]);

  const currentFrame: [number, number, number][] = poseData[currentFrameIndex];
  const currentRep = getCurrentRep();

  return (
    <div>
      <h1>Visualize Pose</h1>
      <p>Subject ID: {subjectId}</p>
      <p>Exercise Name: {exerciseName}</p>
      <p>Pose Data: {poseData.length} frames</p>
      <p>
        Current Frame: {currentFrameIndex + 1}/{poseData.length}
      </p>
      <p>Status: {isPlaying ? "Playing" : "Paused"}</p>

      {/* Rep Timings Section */}
      {repTimings && repTimings.length > 0 ? (
        <div className="mt-4 p-3 bg-gray-100 rounded-lg">
          <h3 className="font-semibold text-lg mb-2">Rep Timings</h3>
          <p className="mb-2">
            <strong>Current Rep:</strong>{" "}
            {currentRep ? `Rep ${currentRep}` : "Not in rep"}
          </p>
          <p className="mb-2">
            <strong>Total Reps:</strong>{" "}
            {repTimings.length > 1 ? repTimings.length - 1 : repTimings.length}
          </p>
          <div className="space-y-1">
            {repTimings.length > 1 &&
              repTimings.slice(0, -1).map((startFrame, index) => {
                const endFrame = repTimings[index + 1];
                const isCurrentRep = currentRep === index + 1;
                return (
                  <div
                    key={index}
                    className={`text-sm cursor-pointer transition-colors duration-200 hover:bg-blue-50 hover:text-blue-700 px-2 py-1 rounded ${
                      isCurrentRep
                        ? "font-bold text-blue-600 bg-blue-100"
                        : "text-gray-700"
                    }`}
                    onClick={() => jumpToRep(index)}
                    title={`Click to jump to Rep ${index + 1}`}
                  >
                    Rep {index + 1}: Frames {startFrame} - {endFrame - 1}(
                    {endFrame - startFrame} frames)
                  </div>
                );
              })}
          </div>
        </div>
      ) : (
        <div className="mt-4 p-3 bg-yellow-100 rounded-lg">
          <p className="text-yellow-800">
            No rep timing data available for this exercise instance.
          </p>
        </div>
      )}

      <p className="text-sm text-gray-600 mt-4">
        Use ← → arrow keys to navigate frames • Space bar to play/pause
      </p>

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
            camera.up.set(0, 0, 1); // Z-up
          }}
        >
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />

          <Grid />

          {currentFrame.map((point, index) => (
            <PosePoint key={index} position={point} />
          ))}

          <PoseConnections points={currentFrame} />

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
