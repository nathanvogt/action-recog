// Types and interfaces
export { Point, SegmentedLeastSquares } from "./slsTypes";

// Class implementations
export { SlsBasic } from "./SlsBasic";
export { SlsMemoized } from "./SlsMemoized";

// Functional implementation (for backward compatibility)
export * from "./slsFunctional";

// Import for factory function
import { SegmentedLeastSquares } from "./slsTypes";
import { SlsBasic } from "./SlsBasic";
import { SlsMemoized } from "./SlsMemoized";

// Factory function for easy instantiation
export function createSls(options?: {
  memoized?: boolean;
  c?: number;
  m?: number;
}): SegmentedLeastSquares {
  const { memoized = false, c = 9, m = 4 } = options || {};

  if (memoized) {
    return new SlsMemoized(c, m);
  } else {
    return new SlsBasic(c, m);
  }
}
