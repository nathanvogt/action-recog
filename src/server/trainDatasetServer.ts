import express, { Request, Response } from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
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

// Replay endpoints
app.get("/api/replays", (req: Request, res: Response) => {
  try {
    const modelsDir = process.env.MODELS_DIR || "models";
    const replays: Array<{
      filename: string;
      subject: string;
      exercise: string;
      path: string;
      lastModified: Date;
    }> = [];

    // Recursively search for replay files
    function findReplayFiles(dir: string) {
      if (!fs.existsSync(dir)) return;

      const items = fs.readdirSync(dir);

      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
          findReplayFiles(fullPath);
        } else if (item.startsWith("replay_") && item.endsWith(".json")) {
          // Parse filename: replay_{subject}_{exercise}_episodes.json
          const match = item.match(/^replay_(.+?)_(.+?)_episodes\.json$/);
          if (match) {
            const [, subject, exercise] = match;
            replays.push({
              filename: item,
              subject,
              exercise,
              path: fullPath,
              lastModified: stat.mtime,
            });
          }
        }
      }
    }

    findReplayFiles(modelsDir);

    // Group by subject and exercise
    const grouped: Record<
      string,
      Record<string, Array<(typeof replays)[0]>>
    > = {};
    for (const replay of replays) {
      if (!grouped[replay.subject]) {
        grouped[replay.subject] = {};
      }
      if (!grouped[replay.subject][replay.exercise]) {
        grouped[replay.subject][replay.exercise] = [];
      }
      grouped[replay.subject][replay.exercise].push(replay);
    }

    res.json(grouped);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({
      error: "Failed to list replays",
      details: errorMessage,
    });
  }
});

app.get("/api/replay/:filename", (req: Request, res: Response) => {
  try {
    const { filename } = req.params;
    const modelsDir = process.env.MODELS_DIR || "models";

    // Find the replay file
    function findReplayFile(
      dir: string,
      targetFilename: string
    ): string | null {
      if (!fs.existsSync(dir)) return null;

      const items = fs.readdirSync(dir);

      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
          const found = findReplayFile(fullPath, targetFilename);
          if (found) return found;
        } else if (item === targetFilename) {
          return fullPath;
        }
      }

      return null;
    }

    const replayPath = findReplayFile(modelsDir, filename);

    if (!replayPath) {
      return res.status(404).json({
        error: "Replay file not found",
        filename,
      });
    }

    const replayData = JSON.parse(fs.readFileSync(replayPath, "utf8"));
    res.json(replayData);
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "Unknown error";
    res.status(500).json({
      error: "Failed to load replay data",
      details: errorMessage,
    });
  }
});

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
  console.log("  GET /api/replays - List all replays");
  console.log("  GET /api/replay/:filename - Get replay data");
  console.log("  GET /health - Health check");
});

export default app;
