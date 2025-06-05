import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { TrainDatasetRemote } from "../../libs/trainDataset/trainDataset.js";

const ExerciseInstanceMenu: React.FC = () => {
  const { subject_id, exercise_name } = useParams<{
    subject_id: string;
    exercise_name: string;
  }>();

  const [cameraIds, setCameraIds] = useState<string[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Video-related state
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  const remote = new TrainDatasetRemote("http://localhost:3001");

  useEffect(() => {
    // fetch camera ids
    const fetchCameraIds = async () => {
      if (!subject_id) return;

      try {
        setError(null);
        const ids = await remote.listCameraIds(subject_id);
        setCameraIds(ids);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to fetch camera IDs"
        );
      }
    };

    fetchCameraIds();
  }, [subject_id]);

  useEffect(() => {
    // Fetch video for the first camera ID
    const fetchVideo = async () => {
      if (!subject_id || !exercise_name || !cameraIds || cameraIds.length === 0)
        return;

      const firstCameraId = cameraIds[0];

      try {
        setVideoError(null);
        const blob = await remote.getVideoBlob(
          subject_id,
          exercise_name,
          firstCameraId
        );

        // Create object URL for video element
        const url = URL.createObjectURL(blob);
        setVideoUrl(url);
      } catch (err) {
        setVideoError(
          err instanceof Error ? err.message : "Failed to load video"
        );
      }
    };

    fetchVideo();
  }, [subject_id, exercise_name, cameraIds]);

  // Handle case where params might be undefined
  if (!subject_id || !exercise_name) {
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold text-red-600">Error</h1>
        <p>
          Invalid URL parameters. Please check the subject ID and exercise name.
        </p>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Exercise Instance</h1>
      <div className="space-y-4">
        <div className="space-y-2">
          <p>
            <strong>Subject ID:</strong> {subject_id}
          </p>
          <p>
            <strong>Exercise Name:</strong> {exercise_name}
          </p>
        </div>

        <div className="border-t pt-4">
          <h2 className="text-lg font-semibold mb-2">Available Camera IDs</h2>
          {cameraIds === null ? (
            <p className="text-gray-600">Loading camera IDs...</p>
          ) : error ? (
            <p className="text-red-600">Error: {error}</p>
          ) : cameraIds.length > 0 ? (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
              {cameraIds.map((cameraId, index) => (
                <div
                  key={cameraId}
                  className={`px-3 py-2 rounded-md text-sm font-medium ${
                    index === 0
                      ? "bg-green-100 text-green-800 border-2 border-green-300"
                      : "bg-blue-100 text-blue-800"
                  }`}
                >
                  {cameraId} {index === 0 && "(Currently shown)"}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-600">
              No camera IDs found for this subject.
            </p>
          )}
        </div>

        {/* Video Section */}
        <div className="border-t pt-4">
          <h2 className="text-lg font-semibold mb-2">Exercise Video</h2>
          {cameraIds === null ? (
            <div className="flex items-center justify-center p-8 bg-gray-100 rounded-lg">
              <p className="text-gray-600">Loading cameras...</p>
            </div>
          ) : videoError ? (
            <div className="p-4 bg-red-100 border border-red-300 rounded-lg">
              <p className="text-red-600">Error loading video: {videoError}</p>
            </div>
          ) : cameraIds.length === 0 ? (
            <div className="p-4 bg-yellow-100 border border-yellow-300 rounded-lg">
              <p className="text-yellow-600">
                No cameras available for this exercise.
              </p>
            </div>
          ) : videoUrl === null ? (
            <div className="flex items-center justify-center p-8 bg-gray-100 rounded-lg">
              <p className="text-gray-600">Loading video...</p>
            </div>
          ) : (
            <div className="space-y-2">
              <p className="text-sm text-gray-600">
                Showing video from camera:{" "}
                <span className="font-medium">{cameraIds[0]}</span>
              </p>
              <video
                controls
                className="rounded-lg shadow-lg"
                preload="metadata"
                style={{ height: "300px", width: "auto" }}
              >
                <source src={videoUrl} type="video/mp4" />
                Your browser doesn't support video playback.
              </video>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ExerciseInstanceMenu;
