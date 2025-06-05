import express, { Request, Response } from "express";
import cors from "cors";
import { TrainDatasetLocal } from "../libs/trainDataset/trainDataset.js";

const app = express();
const port = process.env.PORT || 3001;

// Enable CORS for browser requests
app.use(cors());
app.use(express.json());

// Initialize the local dataset with configurable root path
const datasetRoot = process.env.DATASET_ROOT || "train";
const dataset = new TrainDatasetLocal(datasetRoot);

// Root endpoint
app.get("/api/root", (req: Request, res: Response) => {
  res.json({ root: dataset.root });
});

// Listing utilities
app.get("/api/subjects", (req: Request, res: Response) => {
  try {
    const subjects = dataset.listSubjects();
    res.json(subjects);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res
      .status(500)
      .json({ error: "Failed to list subjects", details: errorMessage });
  }
});

app.get("/api/exercises/:subject", (req: Request, res: Response) => {
  try {
    const { subject } = req.params;
    const exercises = dataset.listExercisesForSubject(subject);
    res.json(exercises);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({
      error: "Failed to list exercises for subject",
      details: errorMessage,
    });
  }
});

app.get("/api/exercises", (req: Request, res: Response) => {
  try {
    const exercises = dataset.listAllExercises();
    res.json(exercises);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res
      .status(500)
      .json({ error: "Failed to list all exercises", details: errorMessage });
  }
});

app.get("/api/instances/:exercise", (req: Request, res: Response) => {
  try {
    const { exercise } = req.params;
    const instances = dataset.listInstances(exercise);
    res.json(instances);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res
      .status(500)
      .json({ error: "Failed to list instances", details: errorMessage });
  }
});

app.get("/api/camera-ids/:subject", (req: Request, res: Response) => {
  try {
    const { subject } = req.params;
    const cameraIds = dataset.listCameraIds(subject);
    res.json(cameraIds);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res
      .status(500)
      .json({ error: "Failed to list camera IDs", details: errorMessage });
  }
});

// Loading utilities
app.get("/api/rep-annotations/:subject", (req: Request, res: Response) => {
  try {
    const { subject } = req.params;
    const annotations = dataset.loadRepAnnotations(subject);
    res.json(annotations);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({
      error: "Failed to load rep annotations",
      details: errorMessage,
    });
  }
});

app.get("/api/instance/:subject/:exercise", (req: Request, res: Response) => {
  try {
    const { subject, exercise } = req.params;
    const instance = dataset.loadInstance(subject, exercise);
    res.json(instance);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res
      .status(500)
      .json({ error: "Failed to load instance", details: errorMessage });
  }
});

// Convenience helpers
app.get("/api/poses/:subject/:exercise", (req: Request, res: Response) => {
  try {
    const { subject, exercise } = req.params;
    const poses = dataset.getPoseArray(subject, exercise);
    res.json(poses);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res
      .status(500)
      .json({ error: "Failed to get pose array", details: errorMessage });
  }
});

app.get(
  "/api/has-exercise/:subject/:exercise",
  (req: Request, res: Response) => {
    try {
      const { subject, exercise } = req.params;
      const hasExercise = dataset.subjectHasExercise(subject, exercise);
      res.json(hasExercise);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      res.status(500).json({
        error: "Failed to check if subject has exercise",
        details: errorMessage,
      });
    }
  }
);

app.get(
  "/api/rep-timings/:subject/:exercise",
  (req: Request, res: Response) => {
    try {
      const { subject, exercise } = req.params;
      const timings = dataset.getRepTimings(subject, exercise);
      res.json(timings);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      res
        .status(500)
        .json({ error: "Failed to get rep timings", details: errorMessage });
    }
  }
);

app.get(
  "/api/rep-segments/:subject/:exercise",
  (req: Request, res: Response) => {
    try {
      const { subject, exercise } = req.params;
      const segments = dataset.getRepSegments(subject, exercise);
      res.json(segments);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      res
        .status(500)
        .json({ error: "Failed to get rep segments", details: errorMessage });
    }
  }
);

app.get(
  "/api/video-blob/:subject/:exercise/:cameraId",
  (req: Request, res: Response) => {
    try {
      const { subject, exercise, cameraId } = req.params;
      const videoBlob = dataset.getVideoBlob(subject, exercise, cameraId);

      // Convert Blob to Buffer for Express response
      videoBlob
        .arrayBuffer()
        .then((arrayBuffer) => {
          const buffer = Buffer.from(arrayBuffer);

          // Set appropriate headers
          res.setHeader("Content-Type", videoBlob.type || "video/mp4");
          res.setHeader("Content-Length", buffer.length);
          res.setHeader("Accept-Ranges", "bytes");

          res.send(buffer);
        })
        .catch((error) => {
          const errorMessage =
            error instanceof Error ? error.message : "Unknown error";
          res.status(500).json({
            error: "Failed to process video blob",
            details: errorMessage,
          });
        });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      res
        .status(500)
        .json({ error: "Failed to get video blob", details: errorMessage });
    }
  }
);

// Health check endpoint
app.get("/health", (req: Request, res: Response) => {
  res.json({ status: "OK", timestamp: new Date().toISOString() });
});

app.listen(port, () => {
  console.log(`TrainDataset server running on http://localhost:${port}`);
  console.log(`Dataset root: ${dataset.root}`);
  console.log("Available endpoints:");
  console.log("  GET /api/root - Get dataset root path");
  console.log("  GET /api/subjects - List all subjects");
  console.log("  GET /api/exercises/:subject - List exercises for subject");
  console.log("  GET /api/exercises - List all exercises");
  console.log("  GET /api/instances/:exercise - List instances for exercise");
  console.log(
    "  GET /api/rep-annotations/:subject - Get rep annotations for subject"
  );
  console.log("  GET /api/instance/:subject/:exercise - Load instance data");
  console.log("  GET /api/poses/:subject/:exercise - Get pose array");
  console.log(
    "  GET /api/has-exercise/:subject/:exercise - Check if subject has exercise"
  );
  console.log("  GET /api/rep-timings/:subject/:exercise - Get rep timings");
  console.log("  GET /api/rep-segments/:subject/:exercise - Get rep segments");
  console.log("  GET /api/camera-ids/:subject - List camera IDs");
  console.log(
    "  GET /api/video-blob/:subject/:exercise/:cameraId - Get video blob"
  );
  console.log("  GET /health - Health check");
});

export default app;
