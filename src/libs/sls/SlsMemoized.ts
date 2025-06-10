import {
  LEFT_LEG_NO_FEET,
  RIGHT_LEG_NO_FEET,
  LEFT_ARM_NO_HAND,
  RIGHT_ARM_NO_HAND,
  BACK,
} from "../data";
import { Point, SegmentedLeastSquares } from "./slsTypes";
import { segmentedLeastSquaresFixedSegments } from "./slsFunctional";

export class SlsMemoized implements SegmentedLeastSquares {
  private c: number;
  private m: number;
  private memIndices: number[][];
  private currentFrame: number;

  constructor(c: number = 9, m: number = 4) {
    this.c = c;
    this.m = m;
    this.memIndices = [];
    this.currentFrame = 0;
  }

  processPoints(points: Point[]): Point[] {
    const [_, indices] = segmentedLeastSquaresFixedSegments(points, this.c);
    return indices.map((i) => points[i]);
  }

  processPointsWithCost(points: Point[]): [Point[], number] {
    const [cost, indices] = segmentedLeastSquaresFixedSegments(points, this.c);
    const segmentPoints = indices.map((i) => points[i]);
    return [segmentPoints, cost];
  }

  /**
   * Process points with memoization for streaming/incremental processing
   */
  processPointsWithMemo(
    pointsWindow: Point[],
    memIndices: number[]
  ): [Point[], number[], number] {
    const inputPoints = memIndices.map((i) => pointsWindow[i]);
    const [loss, indices] = segmentedLeastSquaresFixedSegments(
      inputPoints,
      this.c
    );
    const poseWindowIndices = indices.map((i) => memIndices[i]);

    // Create evenly distributed indices based on M parameter
    const evenly: number[] = [poseWindowIndices[0]];
    for (let i = 0; i < poseWindowIndices.length - 1; i++) {
      const start = poseWindowIndices[i];
      const end = poseWindowIndices[i + 1];
      const diff = end - start;
      if (diff < this.m) {
        for (let j = start; j <= end; j++) evenly.push(j);
      } else {
        const step = diff / this.m;
        for (let j = 0; j < this.m; j++) {
          evenly.push(Math.floor(start + 1 + j * step));
        }
      }
    }
    const uniq = Array.from(new Set(evenly)).sort((a, b) => a - b);
    const output = poseWindowIndices.map((i) => pointsWindow[i]);
    return [output, uniq, loss];
  }

  processPoses(poses: Point[][], keypoints?: number[]): Point[][] {
    if (!keypoints) {
      keypoints = [
        ...LEFT_LEG_NO_FEET,
        ...RIGHT_LEG_NO_FEET,
        ...LEFT_ARM_NO_HAND,
        ...RIGHT_ARM_NO_HAND,
        ...BACK,
      ];
    }

    // Initialize memoization state
    this.memIndices = keypoints.map(() =>
      Array.from({ length: this.c }, (_, i) => i)
    );
    this.currentFrame = 0;

    const lssCurves: Point[][] = keypoints.map(() => []);
    const curves: Point[][] = [];

    // Build curves for each keypoint
    for (let kp = 0; kp < poses[0].length; kp++) {
      curves[kp] = poses.map((frame) => frame[kp]);
    }

    // Process each frame
    for (let frame = 0; frame < poses.length; frame++) {
      this.currentFrame = frame;

      for (let i = 0; i < keypoints.length; i++) {
        const kp = keypoints[i];
        const curve = curves[kp];

        if (frame < this.c) {
          lssCurves[i] = [curve[0]];
        } else {
          this.memIndices[i].push(frame);
          const [points_, indices_] = this.processPointsWithMemo(
            curve,
            this.memIndices[i]
          );
          this.memIndices[i] = indices_;
          lssCurves[i] = points_;
        }
      }
    }

    return lssCurves;
  }

  /**
   * Reset the memoization state
   */
  reset(): void {
    this.memIndices = [];
    this.currentFrame = 0;
  }

  /**
   * Get the current memoization state
   */
  getMemoState(): { memIndices: number[][]; currentFrame: number } {
    return {
      memIndices: this.memIndices.map((indices) => [...indices]),
      currentFrame: this.currentFrame,
    };
  }

  getConfig(): { c: number; m: number } {
    return { c: this.c, m: this.m };
  }
}
