import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { TrainDatasetRemote } from "../../libs/trainDataset/trainDataset.js";

interface ExerciseInstance {
  subject: string;
  path: string;
}

interface ExpandedExercise {
  name: string;
  instances: ExerciseInstance[];
  isLoading: boolean;
}

const ExerciseMenu: React.FC = () => {
  const [exercises, setExercises] = useState<string[]>([]);
  const [expandedExercises, setExpandedExercises] = useState<
    Record<string, ExpandedExercise>
  >({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const remote = new TrainDatasetRemote("http://localhost:3001");
  const navigate = useNavigate();

  useEffect(() => {
    const fetchExercises = async () => {
      try {
        setLoading(true);
        setError(null);

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

  const toggleExercise = async (exerciseName: string) => {
    // If already expanded, collapse it
    if (expandedExercises[exerciseName]) {
      setExpandedExercises((prev) => {
        const newState = { ...prev };
        delete newState[exerciseName];
        return newState;
      });
      return;
    }

    // Start loading state
    setExpandedExercises((prev) => ({
      ...prev,
      [exerciseName]: {
        name: exerciseName,
        instances: [],
        isLoading: true,
      },
    }));

    try {
      const instancesData = await remote.listInstances(exerciseName);
      const instances: ExerciseInstance[] = instancesData.map(
        ([subject, path]) => ({
          subject,
          path,
        })
      );

      setExpandedExercises((prev) => ({
        ...prev,
        [exerciseName]: {
          name: exerciseName,
          instances,
          isLoading: false,
        },
      }));
    } catch (err) {
      // Remove the exercise from expanded state on error
      setExpandedExercises((prev) => {
        const newState = { ...prev };
        delete newState[exerciseName];
        return newState;
      });

      console.error(`Failed to load instances for ${exerciseName}:`, err);
    }
  };

  const handleInstanceClick = (exerciseName: string, subject: string) => {
    navigate(
      `/menu/${encodeURIComponent(subject)}/${encodeURIComponent(exerciseName)}`
    );
  };

  if (loading) {
    return (
      <div className="p-5">
        <h2 className="text-2xl font-bold mb-4">Loading Exercises...</h2>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-5">
        <h2 className="text-2xl font-bold mb-4 text-red-600">Error</h2>
        <p className="text-red-500">{error}</p>
      </div>
    );
  }

  return (
    <div className="p-5 max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold mb-6 text-gray-800">Exercise Menu</h2>

      {exercises.length === 0 ? (
        <p className="text-gray-500">No exercises found.</p>
      ) : (
        <div className="space-y-2">
          {exercises.map((exercise) => {
            const isExpanded = expandedExercises[exercise];

            return (
              <div
                key={exercise}
                className="border border-gray-200 rounded-lg overflow-hidden shadow-sm"
              >
                {/* Exercise Header */}
                <button
                  onClick={() => toggleExercise(exercise)}
                  className="w-full px-4 py-3 bg-white hover:bg-gray-50 flex items-center justify-between text-left transition-colors duration-200 focus:outline-none focus:bg-gray-50"
                >
                  <span className="font-medium text-gray-800">{exercise}</span>
                  <div className="flex items-center">
                    {isExpanded?.isLoading && (
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent mr-2"></div>
                    )}
                    <svg
                      className={`w-5 h-5 text-gray-500 transition-transform duration-200 ${
                        isExpanded ? "transform rotate-180" : ""
                      }`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 9l-7 7-7-7"
                      />
                    </svg>
                  </div>
                </button>

                {/* Exercise Instances */}
                {isExpanded && !isExpanded.isLoading && (
                  <div className="bg-gray-50 border-t border-gray-200">
                    {isExpanded.instances.length === 0 ? (
                      <div className="px-4 py-3 text-gray-500 text-sm">
                        No instances found for this exercise.
                      </div>
                    ) : (
                      <div className="divide-y divide-gray-200">
                        {isExpanded.instances.map((instance, index) => (
                          <button
                            key={`${instance.subject}-${index}`}
                            onClick={() =>
                              handleInstanceClick(exercise, instance.subject)
                            }
                            className="w-full px-6 py-3 hover:bg-gray-100 transition-colors duration-150 text-left focus:outline-none focus:bg-gray-100 focus:ring-2 focus:ring-blue-500 focus:ring-inset"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-sm font-medium text-gray-700">
                                Subject: {instance.subject}
                              </span>
                              <div className="flex items-center space-x-2">
                                <span className="text-xs text-gray-500 font-mono">
                                  {instance.path.split("/").pop()}
                                </span>
                                <svg
                                  className="w-4 h-4 text-gray-400"
                                  fill="none"
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M9 5l7 7-7 7"
                                  />
                                </svg>
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="mt-6 text-sm text-gray-600 bg-gray-100 p-3 rounded-lg">
        <p>
          <strong>Total exercises:</strong> {exercises.length}
        </p>
        <p>
          <strong>Expanded:</strong> {Object.keys(expandedExercises).length}
        </p>
      </div>
    </div>
  );
};

export default ExerciseMenu;
