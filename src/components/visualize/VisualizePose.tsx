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
  const [error, setError] = useState<string | null>(null);

  const remote = new TrainDatasetRemote("http://localhost:3001");

  useEffect(() => {
    const loadPoseData = async () => {
      if (!subject_id || !exercise_name) return;

      setError(null);
      setPoseData(null);

      try {
        const poses = await remote.getPoseArray(subject_id, exercise_name);
        setPoseData(poses);
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
    />
  );
};

type Props = {
  exerciseName: string;
  subjectId: string;
  poseData: [number, number, number][][];
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
}) => {
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

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
      <p className="text-sm text-gray-600">
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
