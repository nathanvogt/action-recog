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
  private readonly c: number;
  private readonly m: number;

  /** Global frame counter (== poses.length) */
  private frameCount = 0;
  /** Global pose buffer.  Index `i` contains the *i-th* frame ever received. */
  private poses: Point[][] = [];
  private memIndices: number[][] = [];
  private kpList: number[] = [];

  constructor(c = 9, m = 4) {
    this.c = c;
    this.m = m;
  }

  /* ─────────────────────────── pure helpers ─────────────────────────── */

  /** Plain SLS on a static set of points (unchanged). */
  processPoints(points: Point[]): Point[] {
    const [, idx] = segmentedLeastSquaresFixedSegments(points, this.c);
    return idx.map((i) => points[i]);
  }

  processPointsWithCost(points: Point[]): [Point[], number] {
    const [loss, idx] = segmentedLeastSquaresFixedSegments(points, this.c);
    return [idx.map((i) => points[i]), loss];
  }

  /**
   * The **memoised** variant used internally for streaming.
   * Returns the new segment points, the *updated* memo-indices and the loss.
   */
  private processPointsWithMemo(
    pointsWindow: Point[],
    mem: number[]
  ): [Point[], number[], number] {
    const inputPoints = mem.map((i) => pointsWindow[i]);
    const [loss, segIdx] = segmentedLeastSquaresFixedSegments(
      inputPoints,
      this.c
    );
    const poseWindowIdx = segIdx.map((i) => mem[i]);

    /* build evenly-spaced indices between successive endpoints */
    const even: number[] = [poseWindowIdx[0]];
    for (let k = 0; k < poseWindowIdx.length - 1; k++) {
      const start = poseWindowIdx[k];
      const end = poseWindowIdx[k + 1];
      const gap = end - start;

      if (gap <= this.m) {
        for (let j = start; j < end; j++) even.push(j);
      } else {
        const step = gap / this.m;
        for (let j = 0; j < this.m; j++) {
          even.push(Math.floor(start + 1 + j * step));
        }
      }
    }

    const uniq = Array.from(new Set(even)).sort((a, b) => a - b);
    const segPoints = poseWindowIdx.map((i) => pointsWindow[i]);
    return [segPoints, uniq, loss];
  }

  /* ─────────────────────── streaming public API ─────────────────────── */

  /**
   * Incrementally process *new* poses.
   *
   * @param newPoses  – array of (new) frames; each frame is a 16-length keypoint array.
   * @param keypoints – which keypoints to approximate; if omitted we use the default body list.
   * @returns         – `[lssCurves, totalLoss]`
   *
   *   * `lssCurves[i]` is the updated SLS curve for the *i-th* requested keypoint.
   *   * `totalLoss`    is the sum of approximation errors for this batch.
   */
  processPoses(
    newPoses: Point[][],
    keypoints: number[] = [
      ...LEFT_LEG_NO_FEET,
      ...RIGHT_LEG_NO_FEET,
      ...LEFT_ARM_NO_HAND,
      ...RIGHT_ARM_NO_HAND,
      ...BACK,
    ]
  ): [Point[][], number] {
    this.kpList = keypoints;
    if (newPoses.length === 0) return [[], 0];

    /* ── initialise state (first call OR changed keypoints) ── */
    if (
      this.memIndices.length === 0 ||
      this.memIndices.length !== keypoints.length
    ) {
      this.memIndices = keypoints.map(() => []);
      this.frameCount = 0;
      this.poses = [];
    }

    /* ── append the new frames to the global buffer ── */
    this.poses.push(...newPoses);
    const startFrame = this.frameCount;
    this.frameCount += newPoses.length;

    /* ── update memo lists for every keypoint with the *new* frame indices ── */
    for (let kpIdx = 0; kpIdx < keypoints.length; kpIdx++) {
      const kpMemo = this.memIndices[kpIdx];
      for (let i = 0; i < newPoses.length; i++) {
        kpMemo.push(startFrame + i);
      }
    }

    /* ── run SLS with memo on the full curve of each keypoint ── */
    const lssCurves: Point[][] = [];
    let totalLoss = 0;

    for (let idx = 0; idx < keypoints.length; idx++) {
      const kp = keypoints[idx]; // actual keypoint id in a frame
      const curve = this.poses.map((f) => f[kp]); // whole trajectory so far
      const memo = this.memIndices[idx];
      if (curve.length <= this.c) {
        // still warming up – nothing to do yet
        lssCurves[idx] = curve.length ? [curve[0]] : [];
        continue;
      }

      const [segPts, newMemo, loss] = this.processPointsWithMemo(curve, memo);
      this.memIndices[idx] = newMemo; // persist for next call
      lssCurves[idx] = segPts;
      totalLoss += loss;
    }

    return [lssCurves, totalLoss];
  }

  /* ─────────────────────────── misc helpers ─────────────────────────── */

  /** Forget all state (useful between independent sequences). */
  reset(): void {
    this.frameCount = 0;
    this.poses = [];
    this.memIndices = [];
  }

  /** Introspection helper – *copies* of the mutable state. */
  getMemoState() {
    return {
      currentFrame: this.frameCount,
      memIndices: this.memIndices.map((a) => [...a]),
    };
  }

  getMemPoints(): Point[][] {
    if (this.kpList.length === 0) return [];

    return this.memIndices.map((frameIdxArr, kpIdx) =>
      frameIdxArr.map((frameIdx) => this.poses[frameIdx][this.kpList[kpIdx]])
    );
  }

  getConfig() {
    return { c: this.c, m: this.m };
  }
}
