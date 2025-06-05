import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { TrainDatasetRemote } from "../../libs/trainDataset/trainDataset.js";

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

const Grid: React.FC = () => {
  const gridSize = 2;
  const gridDivisions = 20;
  const gridColor = "#888888";

  const lines = [];
  const halfSize = gridSize / 2;
  const step = gridSize / gridDivisions;

  // Create grid lines parallel to X-axis
  for (let i = 0; i <= gridDivisions; i++) {
    const z = -halfSize + i * step;
    lines.push(
      <line key={`x-${i}`}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([-halfSize, 0, z, halfSize, 0, z]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color={gridColor} />
      </line>
    );
  }

  // Create grid lines parallel to Z-axis
  for (let i = 0; i <= gridDivisions; i++) {
    const x = -halfSize + i * step;
    lines.push(
      <line key={`z-${i}`}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array([x, 0, -halfSize, x, 0, halfSize]), 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color={gridColor} />
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
        <Canvas camera={{ position: [2, 2, 2], fov: 50 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />

          <Grid />

          {currentFrame.map((point, index) => (
            <PosePoint key={index} position={point} />
          ))}

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
