export interface InstanceData {
  subject: string;
  exercise: string;
  poses: number[][][];
  timings: number[] | null;
  num_reps: number | undefined;
  info: Record<string, any>;
}

export interface TrainDataset {
  readonly root: string;

  // Listing utilities
  listSubjects(): string[];
  listExercisesForSubject(subject: string): string[];
  listAllExercises(): string[];
  listInstances(exercise: string): Array<[string, string]>;

  // Loading utilities
  loadRepAnnotations(subject: string): Record<string, number[]> | null;
  loadInstance(subject: string, exercise: string): InstanceData;

  // Convenience helpers
  getPoseArray(subject: string, exercise: string): number[][][];
  subjectHasExercise(subject: string, exercise: string): boolean;
  getRepTimings(subject: string, exercise: string): number[] | null;
  getRepSegments(
    subject: string,
    exercise: string
  ): Array<[number, number]> | null;
}
