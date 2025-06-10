export type Point = [number, number, number];

export interface SegmentedLeastSquares {
  /**
   * Process a sequence of points and return the optimal segment points
   * @param points Array of 3D points to segment
   * @returns Array of points representing segment boundaries
   */
  processPoints(points: Point[]): Point[];

  /**
   * Process a sequence of points and return both the segment points and the total cost
   * @param points Array of 3D points to segment
   * @returns Tuple of [segment points, total cost]
   */
  processPointsWithCost(points: Point[]): [Point[], number];

  /**
   * Process a sequence of poses (frames of multiple keypoints)
   * @param poses Array of frames, where each frame is an array of keypoints
   * @param keypoints Optional array of keypoint indices to process
   * @returns Tuple of [segmented curves (one for each processed keypoint), total error]
   */
  processPoses(poses: Point[][], keypoints?: number[]): [Point[][], number];

  /**
   * Get the current configuration
   */
  getConfig(): { c: number; m: number };
}
