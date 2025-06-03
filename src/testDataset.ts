import { TrainDataset } from './trainDataset';

function main() {
  const dataset = new TrainDataset();
  console.log('Dataset root:', dataset.root);

  const subjects = dataset.listSubjects();
  console.log('Subjects:', subjects);

  const allExercises = dataset.listAllExercises();
  console.log('All exercises:', allExercises);

  if (subjects.length) {
    const subj = subjects[0];
    const exercises = dataset.listExercisesForSubject(subj);
    console.log(`Exercises for ${subj}:`, exercises);

    if (exercises.length) {
      const ex = exercises[0];
      if (dataset.subjectHasExercise(subj, ex)) {
        const instance = dataset.loadInstance(subj, ex);
        console.log('Loaded instance:', {
          subject: instance.subject,
          exercise: instance.exercise,
          posesShape: [instance.poses.length, instance.poses[0]?.length ?? 0],
          numReps: instance.num_reps,
          hasTimings: instance.timings !== null,
        });
      }
    }
  }
}

main();
