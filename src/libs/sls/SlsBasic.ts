import {
  LEFT_LEG_NO_FEET,
  RIGHT_LEG_NO_FEET,
  LEFT_ARM_NO_HAND,
  RIGHT_ARM_NO_HAND,
  BACK,
} from "../data";
import { Point, SegmentedLeastSquares } from "./slsTypes";
import {
  shortestDistance,
  segmentedLeastSquaresFixedSegments,
} from "./slsFunctional";

export class SlsBasic implements SegmentedLeastSquares {
  private c: number;
  private m: number;

  constructor(c: number = 9, m: number = 4) {
    this.c = c;
    this.m = m;
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

    const curves: Point[][] = [];
    for (let kp = 0; kp < poses[0].length; kp++) {
      curves[kp] = poses.map((frame) => frame[kp]);
    }

    const result: Point[][] = [];
    for (let i = 0; i < keypoints.length; i++) {
      const kp = keypoints[i];
      const curve = curves[kp];
      result[i] = this.processPoints(curve);
    }

    return result;
  }

  getConfig(): { c: number; m: number } {
    return { c: this.c, m: this.m };
  }
}
