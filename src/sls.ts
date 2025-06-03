import { LEFT_LEG_NO_FEET, RIGHT_LEG_NO_FEET, LEFT_ARM_NO_HAND, RIGHT_ARM_NO_HAND, BACK } from "./data";
export type Point = [number, number, number];

export function shortestDistance(p1: Point, p2: Point, p: Point): number {
  const lineVec: Point = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
  const pointVec: Point = [p[0] - p1[0], p[1] - p1[1], p[2] - p1[2]];

  const cross: Point = [
    lineVec[1] * pointVec[2] - lineVec[2] * pointVec[1],
    lineVec[2] * pointVec[0] - lineVec[0] * pointVec[2],
    lineVec[0] * pointVec[1] - lineVec[1] * pointVec[0],
  ];

  const norm = Math.sqrt(lineVec[0]**2 + lineVec[1]**2 + lineVec[2]**2) || 1e-5;
  let distance = Math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2) / norm;

  const p1Dist = Math.sqrt(
    (p[0]-p1[0])**2 + (p[1]-p1[1])**2 + (p[2]-p1[2])**2
  );
  const p2Dist = Math.sqrt(
    (p[0]-p2[0])**2 + (p[1]-p2[1])**2 + (p[2]-p2[2])**2
  );
  const closer = p1Dist <= p2Dist ? p1 : p2;
  const farther = p1Dist <= p2Dist ? p2 : p1;
  const vec: Point = [farther[0]-closer[0], farther[1]-closer[1], farther[2]-closer[2]];
  const dot = vec[0]*pointVec[0] + vec[1]*pointVec[1] + vec[2]*pointVec[2];

  if (dot < 0) {
    distance = Math.min(p1Dist, p2Dist);
  }

  return distance;
}

export function segmentedLeastSquaresFixedSegments(points: Point[], numSegments: number): [number, number[]] {
  const n = points.length;
  if (numSegments >= n) return [0, []];

  const E: number[][] = Array.from({length: n}, () => Array(n).fill(0));
  const dp: number[][] = Array.from({length: n}, () => Array(numSegments+1).fill(Infinity));
  const result: number[][] = Array.from({length: n}, () => Array(numSegments+1).fill(0));

  for (let j=0; j<n; j++) {
    for (let i=0; i<=j; i++) {
      let sum = 0;
      for (let k=i; k<=j; k++) {
        sum += shortestDistance(points[i], points[j], points[k]);
      }
      E[i][j] = sum;
    }
  }

  for (let j=0; j<n; j++) dp[j][1] = E[0][j];
  for (let j=1; j<n; j++) dp[j][numSegments] = Infinity;
  for (let j=0; j<n; j++) result[j][1] = 0;

  for (let k=2; k<=numSegments; k++) {
    for (let j=k-1; j<n; j++) {
      let minCost = Infinity;
      let minIdx = 0;
      for (let i=k-1; i<=j; i++) {
        const cost = E[i][j] + dp[i-1][k-1];
        if (cost < minCost) {
          minCost = cost;
          minIdx = i;
        }
      }
      dp[j][k] = minCost;
      result[j][k] = minIdx;
    }
  }

  const indices: number[] = [];
  let k = numSegments - 1;
  let idx = n - 1;
  while (k > 0) {
    indices.push(idx);
    idx = result[idx][k] - 1;
    k -= 1;
  }
  indices.push(0);
  indices.reverse();
  return [dp[n-1][numSegments], indices];
}

export const DEFAULT_C = 9;
export const DEFAULT_M = 4;

export function slsPoints(points: Point[], c: number = DEFAULT_C): Point[] {
  const [_, indices] = segmentedLeastSquaresFixedSegments(points, c);
  return indices.map(i => points[i]);
}

export function createSlsWithMemo(m: number = DEFAULT_M) {
  const M = m;
  return function(pointsWindow: Point[], memIndices: number[], c: number = DEFAULT_C): [Point[], number[], number] {
    const inputPoints = memIndices.map(i => pointsWindow[i]);
    const [loss, indices] = segmentedLeastSquaresFixedSegments(inputPoints, c);
    const poseWindowIndices = indices.map(i => memIndices[i]);

    const evenly: number[] = [poseWindowIndices[0]];
    for (let i=0; i<poseWindowIndices.length-1; i++) {
      const start = poseWindowIndices[i];
      const end = poseWindowIndices[i+1];
      const diff = end - start;
      if (diff < M) {
        for (let j=start; j<=end; j++) evenly.push(j);
      } else {
        const step = diff / M;
        for (let j=0; j<M; j++) {
          evenly.push(Math.floor(start + 1 + j*step));
        }
      }
    }
    const uniq = Array.from(new Set(evenly)).sort((a,b)=>a-b);
    const output = poseWindowIndices.map(i => pointsWindow[i]);
    return [output, uniq, loss];
  }
}

export function processPoses(
  poses: Point[][],
  c: number,
  m: number,
  keypoints?: number[]
): Point[][] {
  if (!keypoints) {
    keypoints = [
      ...LEFT_LEG_NO_FEET,
      ...RIGHT_LEG_NO_FEET,
      ...LEFT_ARM_NO_HAND,
      ...RIGHT_ARM_NO_HAND,
      ...BACK,
    ];
  }
  const slsFuncs = keypoints.map(() => createSlsWithMemo(m));
  const lssCurves: Point[][] = keypoints.map(() => []);
  const memIndices: number[][] = keypoints.map(() => Array.from({length:c}, (_,i)=>i));
  const curves: Point[][] = [];
  for (let kp=0; kp<poses[0].length; kp++) {
    curves[kp] = poses.map(frame => frame[kp]);
  }
  for (let frame=0; frame<poses.length; frame++) {
    for (let i=0; i<keypoints.length; i++) {
      const kp = keypoints[i];
      const curve = curves[kp];
      if (frame < c) {
        lssCurves[i] = [curve[0]];
      } else {
        memIndices[i].push(frame);
        const [points_, indices_] = slsFuncs[i](curve, memIndices[i], c);
        memIndices[i] = indices_;
        lssCurves[i] = points_;
      }
    }
  }
  return lssCurves;
}
