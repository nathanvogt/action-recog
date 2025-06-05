export const KP_TO_NAME: Record<number, string> = {
  0: "center_hip",
  1: "right_hip",
  2: "right_knee",
  3: "right_ankle",
  4: "left_hip",
  5: "left_knee",
  6: "left_ankle",
  7: "center_spine",
  8: "neck",
  9: "tip_of_neck",
  10: "head",
  11: "right_shoulder",
  12: "right_elbow",
  13: "right_wrist",
  14: "left_shoulder",
  15: "left_elbow",
  16: "left_wrist",
  17: "right_middle_foot",
  18: "right_tip_foot",
  19: "left_middle_foot",
  20: "left_tip_foot",
  21: "right_hand_1",
  22: "right_hand_2",
  23: "left_hand_1",
  24: "left_hand_2",
};

export const LEFT_LEG_NO_FEET = [4, 5, 6];
export const RIGHT_LEG_NO_FEET = [1, 2, 3];
export const LEFT_ARM_NO_HAND = [14, 15, 16];
export const RIGHT_ARM_NO_HAND = [11, 12, 13];
export const LEFT_HAND = [22, 24];
export const RIGHT_HAND = [21, 23];
export const LEFT_FOOT = [19, 20];
export const RIGHT_FOOT = [17, 18];
export const BACK = [0, 7];

// Connections between keypoints for pose visualization
// Each sub-array contains two keypoint indices that should be connected
export const CONNECTIONS: [number, number][] = [
  // Spine/torso
  [0, 7], // center_hip -> center_spine
  [7, 8], // center_spine -> neck
  [8, 9], // neck -> tip_of_neck
  [9, 10], // tip_of_neck -> head

  // Hip connections
  [0, 1], // center_hip -> right_hip
  [0, 4], // center_hip -> left_hip

  // Right leg
  [1, 2], // right_hip -> right_knee
  [2, 3], // right_knee -> right_ankle

  // Left leg
  [4, 5], // left_hip -> left_knee
  [5, 6], // left_knee -> left_ankle

  // Shoulder connections
  [8, 11], // neck -> right_shoulder
  [8, 14], // neck -> left_shoulder

  // Right arm
  [11, 12], // right_shoulder -> right_elbow
  [12, 13], // right_elbow -> right_wrist

  // Left arm
  [14, 15], // left_shoulder -> left_elbow
  [15, 16], // left_elbow -> left_wrist

  // Right hand connections
  [13, 21], // right_wrist -> right_hand_1
  [13, 22], // right_wrist -> right_hand_2
  //   [21, 22], // right_hand_1 -> right_hand_2

  // Left hand connections
  [16, 23], // left_wrist -> left_hand_1
  [16, 24], // left_wrist -> left_hand_2
  //   [23, 24], // left_hand_1 -> left_hand_2

  // Right foot connections
  [3, 17], // right_ankle -> right_middle_foot
  [17, 18], // right_middle_foot -> right_tip_foot

  // Left foot connections
  [6, 19], // left_ankle -> left_middle_foot
  [19, 20], // left_middle_foot -> left_tip_foot
];
