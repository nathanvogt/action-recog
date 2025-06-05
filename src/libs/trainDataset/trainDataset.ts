import fs from "fs";
import path from "path";
import { InstanceData, TrainDataset } from "./trainDatasetTypes";
import { AsyncMethods } from "../utilTypes";

export class TrainDatasetLocal implements TrainDataset {
  root: string;

  constructor(root: string = "train") {
    this.root = path.resolve(root);
  }

  // ------------------------------------------------------------------
  // listing utilities
  // ------------------------------------------------------------------
  listSubjects(): string[] {
    if (!fs.existsSync(this.root)) return [];
    return fs
      .readdirSync(this.root)
      .filter((d) => fs.statSync(path.join(this.root, d)).isDirectory())
      .sort();
  }

  listExercisesForSubject(subject: string): string[] {
    const p = path.join(this.root, subject, "joints3d_25");
    if (!fs.existsSync(p)) return [];
    return fs
      .readdirSync(p)
      .filter(
        (f) => f.endsWith(".json") && fs.statSync(path.join(p, f)).isFile()
      )
      .map((f) => path.parse(f).name)
      .sort();
  }

  listAllExercises(): string[] {
    const ex = new Set<string>();
    for (const s of this.listSubjects()) {
      for (const e of this.listExercisesForSubject(s)) {
        ex.add(e);
      }
    }
    return Array.from(ex).sort();
  }

  listInstances(exercise: string): Array<[string, string]> {
    const instances: Array<[string, string]> = [];
    for (const s of this.listSubjects()) {
      const p = path.join(this.root, s, "joints3d_25", `${exercise}.json`);
      if (fs.existsSync(p)) instances.push([s, p]);
    }
    return instances;
  }

  listCameraIds(subject: string): string[] {
    const p = path.join(this.root, subject, "camera_parameters");
    if (!fs.existsSync(p)) return [];
    return fs
      .readdirSync(p)
      .filter((d) => fs.statSync(path.join(p, d)).isDirectory())
      .sort();
  }

  // ------------------------------------------------------------------
  // loading utilities
  // ------------------------------------------------------------------
  loadRepAnnotations(subject: string): Record<string, number[]> | null {
    const p = path.join(this.root, subject, "rep_ann.json");
    if (!fs.existsSync(p)) return null;
    try {
      const txt = fs.readFileSync(p, "utf8");
      return JSON.parse(txt);
    } catch {
      return null;
    }
  }

  loadInstance(subject: string, exercise: string): InstanceData {
    const p = path.join(this.root, subject, "joints3d_25", `${exercise}.json`);
    const data = JSON.parse(fs.readFileSync(p, "utf8")) as Record<string, any>;

    let poses: [number, number, number][][] | null = null;
    for (const key of ["joints3d_25", "joints3d", "poses3d"]) {
      if (key in data) {
        poses = data[key];
        break;
      }
    }
    if (!poses) throw new Error(`No pose data found in ${p}`);

    const info: Record<string, any> = {};
    for (const [k, v] of Object.entries(data)) {
      if (!["joints3d_25", "joints3d", "poses3d"].includes(k)) {
        info[k] = v;
      }
    }

    let timings: number[] | null = null;
    const rep = this.loadRepAnnotations(subject);
    if (rep && exercise in rep) timings = rep[exercise];
    if (!timings) {
      timings =
        (info["rep_timings"] || info["reps"] || info["timings"]) ?? null;
    }

    const num_reps = Array.isArray(timings) ? timings.length : info["num_reps"];

    return { subject, exercise, poses, timings, num_reps, info };
  }

  // ------------------------------------------------------------------
  // convenience helpers
  // ------------------------------------------------------------------
  getPoseArray(subject: string, exercise: string): number[][][] {
    return this.loadInstance(subject, exercise).poses;
  }

  subjectHasExercise(subject: string, exercise: string): boolean {
    const p = path.join(this.root, subject, "joints3d_25", `${exercise}.json`);
    return fs.existsSync(p);
  }

  getRepTimings(subject: string, exercise: string): number[] | null {
    return this.loadInstance(subject, exercise).timings;
  }

  getRepSegments(
    subject: string,
    exercise: string
  ): Array<[number, number]> | null {
    const t = this.getRepTimings(subject, exercise);
    if (!t || t.length < 2) return null;
    const seg: Array<[number, number]> = [];
    for (let i = 0; i < t.length - 1; i++) {
      seg.push([t[i], t[i + 1]]);
    }
    return seg;
  }

  getVideoBlob(subject: string, exercise: string, cameraId: string): Blob {
    // Try common video file extensions and locations
    const possiblePaths = [
      path.join(this.root, subject, "videos", cameraId, `${exercise}.mp4`),
      path.join(this.root, subject, "videos", cameraId, `${exercise}.webm`),
      path.join(this.root, subject, "videos", cameraId, `${exercise}.mov`),
      path.join(this.root, subject, cameraId, `${exercise}.mp4`),
      path.join(this.root, subject, cameraId, `${exercise}.webm`),
      path.join(this.root, subject, cameraId, `${exercise}.mov`),
    ];

    for (const videoPath of possiblePaths) {
      if (fs.existsSync(videoPath)) {
        const videoBuffer = fs.readFileSync(videoPath);
        const mimeType = this.getMimeTypeFromPath(videoPath);
        return new Blob([videoBuffer], { type: mimeType });
      }
    }

    throw new Error(
      `Video not found for subject: ${subject}, exercise: ${exercise}, camera: ${cameraId}`
    );
  }

  private getMimeTypeFromPath(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();
    switch (ext) {
      case ".mp4":
        return "video/mp4";
      case ".webm":
        return "video/webm";
      case ".mov":
        return "video/quicktime";
      case ".avi":
        return "video/x-msvideo";
      default:
        return "video/mp4"; // default fallback
    }
  }
}

export class TrainDatasetRemote implements AsyncMethods<TrainDataset> {
  private baseUrl: string;
  private _root: Promise<string>;

  constructor(baseUrl: string = "http://localhost:3001") {
    this.baseUrl = baseUrl;
    this._root = this.fetchRoot();
  }

  private async fetchRoot(): Promise<string> {
    const response = await fetch(`${this.baseUrl}/api/root`);
    if (!response.ok) {
      throw new Error(`Failed to fetch root: ${response.statusText}`);
    }
    const data = await response.json();
    return data.root;
  }

  get root(): Promise<string> {
    return this._root;
  }

  async listSubjects(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/subjects`);
    if (!response.ok) {
      throw new Error(`Failed to list subjects: ${response.statusText}`);
    }
    return response.json();
  }

  async listExercisesForSubject(subject: string): Promise<string[]> {
    const response = await fetch(
      `${this.baseUrl}/api/exercises/${encodeURIComponent(subject)}`
    );
    if (!response.ok) {
      throw new Error(
        `Failed to list exercises for subject: ${response.statusText}`
      );
    }
    return response.json();
  }

  async listAllExercises(): Promise<string[]> {
    const response = await fetch(`${this.baseUrl}/api/exercises`);
    if (!response.ok) {
      throw new Error(`Failed to list all exercises: ${response.statusText}`);
    }
    return response.json();
  }

  async listInstances(exercise: string): Promise<Array<[string, string]>> {
    const response = await fetch(
      `${this.baseUrl}/api/instances/${encodeURIComponent(exercise)}`
    );
    if (!response.ok) {
      throw new Error(`Failed to list instances: ${response.statusText}`);
    }
    return response.json();
  }

  async listCameraIds(subject: string): Promise<string[]> {
    const response = await fetch(
      `${this.baseUrl}/api/camera-ids/${encodeURIComponent(subject)}`
    );
    if (!response.ok) {
      throw new Error(`Failed to list camera IDs: ${response.statusText}`);
    }
    return response.json();
  }

  async loadRepAnnotations(
    subject: string
  ): Promise<Record<string, number[]> | null> {
    const response = await fetch(
      `${this.baseUrl}/api/rep-annotations/${encodeURIComponent(subject)}`
    );
    if (!response.ok) {
      throw new Error(`Failed to load rep annotations: ${response.statusText}`);
    }
    return response.json();
  }

  async loadInstance(subject: string, exercise: string): Promise<InstanceData> {
    const response = await fetch(
      `${this.baseUrl}/api/instance/${encodeURIComponent(
        subject
      )}/${encodeURIComponent(exercise)}`
    );
    if (!response.ok) {
      throw new Error(`Failed to load instance: ${response.statusText}`);
    }
    return response.json();
  }

  async getPoseArray(subject: string, exercise: string): Promise<number[][][]> {
    const response = await fetch(
      `${this.baseUrl}/api/poses/${encodeURIComponent(
        subject
      )}/${encodeURIComponent(exercise)}`
    );
    if (!response.ok) {
      throw new Error(`Failed to get pose array: ${response.statusText}`);
    }
    return response.json();
  }

  async subjectHasExercise(
    subject: string,
    exercise: string
  ): Promise<boolean> {
    const response = await fetch(
      `${this.baseUrl}/api/has-exercise/${encodeURIComponent(
        subject
      )}/${encodeURIComponent(exercise)}`
    );
    if (!response.ok) {
      throw new Error(
        `Failed to check if subject has exercise: ${response.statusText}`
      );
    }
    return response.json();
  }

  async getRepTimings(
    subject: string,
    exercise: string
  ): Promise<number[] | null> {
    const response = await fetch(
      `${this.baseUrl}/api/rep-timings/${encodeURIComponent(
        subject
      )}/${encodeURIComponent(exercise)}`
    );
    if (!response.ok) {
      throw new Error(`Failed to get rep timings: ${response.statusText}`);
    }
    return response.json();
  }

  async getRepSegments(
    subject: string,
    exercise: string
  ): Promise<Array<[number, number]> | null> {
    const response = await fetch(
      `${this.baseUrl}/api/rep-segments/${encodeURIComponent(
        subject
      )}/${encodeURIComponent(exercise)}`
    );
    if (!response.ok) {
      throw new Error(`Failed to get rep segments: ${response.statusText}`);
    }
    return response.json();
  }

  async getVideoBlob(
    subject: string,
    exercise: string,
    cameraId: string
  ): Promise<Blob> {
    const response = await fetch(
      `${this.baseUrl}/api/video-blob/${encodeURIComponent(
        subject
      )}/${encodeURIComponent(exercise)}/${encodeURIComponent(cameraId)}`
    );
    if (!response.ok) {
      throw new Error(`Failed to get video blob: ${response.statusText}`);
    }
    return response.blob();
  }
}
