import React, { useState, useEffect } from "react";
import { TrainDatasetRemote } from "../libs/trainDataset/trainDataset.js";

type Props = {};

const Main: React.FC = () => {
  const [exercises, setExercises] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchExercises = async () => {
      try {
        setLoading(true);
        setError(null);

        const remote = new TrainDatasetRemote("http://localhost:3001");

        const allExercises = await remote.listAllExercises();
        setExercises(allExercises);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to fetch exercises"
        );
      } finally {
        setLoading(false);
      }
    };

    fetchExercises();
  }, []);

  if (loading) {
    return (
      <div style={{ padding: "20px" }}>
        <h2>Loading Exercises...</h2>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: "20px" }}>
        <h2>Error</h2>
        <p style={{ color: "red" }}>{error}</p>
      </div>
    );
  }

  return (
    <div style={{ padding: "20px" }}>
      <h2>All Exercises</h2>
      {exercises.length === 0 ? (
        <p>No exercises found.</p>
      ) : (
        <ul style={{ listStyle: "none", padding: 0 }}>
          {exercises.map((exercise, index) => (
            <li
              key={exercise}
              style={{
                padding: "10px",
                margin: "5px 0",
                backgroundColor: "#f5f5f5",
                borderRadius: "5px",
                border: "1px solid #ddd",
              }}
            >
              <strong>{index + 1}.</strong> {exercise}
            </li>
          ))}
        </ul>
      )}
      <p style={{ marginTop: "20px", fontSize: "14px", color: "#666" }}>
        Total exercises: {exercises.length}
      </p>
    </div>
  );
};

export default Main;
