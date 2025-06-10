// test_sls_typescript.ts
import fs from "fs";
import { SlsMemoized } from "../libs/sls/SlsMemoized";
import { TrainDatasetLocal } from "../libs/trainDataset/trainDataset";
import {
  LEFT_LEG_NO_FEET,
  RIGHT_LEG_NO_FEET,
  LEFT_ARM_NO_HAND,
  RIGHT_ARM_NO_HAND,
  BACK,
} from "../libs/data.js";

const DEFAULT_KEYPOINTS = [
  ...LEFT_LEG_NO_FEET,
  ...RIGHT_LEG_NO_FEET,
  ...LEFT_ARM_NO_HAND,
  ...RIGHT_ARM_NO_HAND,
  ...BACK,
];

function testSlsTypeScript(
  subject: string,
  exercise: string,
  c: number = 9,
  m: number = 4,
  maxFrames: number = 50
) {
  // Load dataset
  const dataset = new TrainDatasetLocal("train");
  const poses = dataset.getPoseArray(subject, exercise);

  // Limit frames for testing
  const limitedPoses = poses.slice(0, maxFrames);

  // Initialize SLS
  const sls = new SlsMemoized(c, m);

  const results: any[] = [];

  // Process poses one by one
  for (let frameIdx = 0; frameIdx < limitedPoses.length; frameIdx++) {
    const pose = limitedPoses[frameIdx];

    // Process single pose (wrap in array since TS version expects array)
    const [lssCurves, totalLoss] = sls.processPoses([pose], DEFAULT_KEYPOINTS);

    // Get memo state
    const memoState = sls.getMemoState();

    const frameResult = {
      frame: frameIdx,
      lss_curves: lssCurves,
      total_loss: totalLoss,
      memo_state: memoState,
      num_keypoints: DEFAULT_KEYPOINTS.length,
    };

    results.push(frameResult);

    // Print progress to stderr
    console.error(
      `TypeScript - Frame ${frameIdx}: loss=${totalLoss.toFixed(
        6
      )}, curves_count=${lssCurves.length}`
    );
  }

  // Output results to stdout as JSON
  const output = {
    implementation: "typescript",
    subject,
    exercise,
    c,
    m,
    keypoints: DEFAULT_KEYPOINTS,
    total_frames: limitedPoses.length,
    results,
  };

  console.log(JSON.stringify(output, null, 2));
}

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length !== 2) {
  console.error("Usage: ts-node test_sls_typescript.ts <subject> <exercise>");
  process.exit(1);
}

const [subject, exercise] = args;
testSlsTypeScript(subject, exercise);
