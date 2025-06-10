import { TrainDatasetRemote } from "../libs/trainDataset/trainDataset";

async function testRemoteDataset() {
  console.log("Testing TrainDatasetRemote...");

  // Create remote dataset instance
  const remote = new TrainDatasetRemote("http://localhost:3001");

  try {
    // Test basic functionality
    console.log("\n1. Getting dataset root...");
    const root = await remote.root;
    console.log("Root:", root);

    console.log("\n2. Listing subjects...");
    const subjects = await remote.listSubjects();
    console.log("Subjects:", subjects);

    console.log("\n3. Listing all exercises...");
    const exercises = await remote.listAllExercises();
    console.log("Exercises:", exercises);

    if (subjects.length > 0) {
      const firstSubject = subjects[0];
      console.log(`\n4. Listing exercises for subject "${firstSubject}"...`);
      const subjectExercises = await remote.listExercisesForSubject(
        firstSubject
      );
      console.log("Subject exercises:", subjectExercises);

      if (subjectExercises.length > 0) {
        const firstExercise = subjectExercises[0];
        console.log(
          `\n5. Checking if subject "${firstSubject}" has exercise "${firstExercise}"...`
        );
        const hasExercise = await remote.subjectHasExercise(
          firstSubject,
          firstExercise
        );
        console.log("Has exercise:", hasExercise);

        if (hasExercise) {
          console.log(
            `\n6. Loading instance for "${firstSubject}" - "${firstExercise}"...`
          );
          const instance = await remote.loadInstance(
            firstSubject,
            firstExercise
          );
          console.log("Instance info:", {
            subject: instance.subject,
            exercise: instance.exercise,
            posesLength: instance.poses.length,
            timings: instance.timings,
            numReps: instance.num_reps,
          });

          console.log(
            `\n7. Getting pose array for "${firstSubject}" - "${firstExercise}"...`
          );
          const poses = await remote.getPoseArray(firstSubject, firstExercise);
          console.log("Poses shape:", poses.length, "frames");

          console.log(
            `\n8. Getting rep timings for "${firstSubject}" - "${firstExercise}"...`
          );
          const timings = await remote.getRepTimings(
            firstSubject,
            firstExercise
          );
          console.log("Rep timings:", timings);

          console.log(
            `\n9. Getting rep segments for "${firstSubject}" - "${firstExercise}"...`
          );
          const segments = await remote.getRepSegments(
            firstSubject,
            firstExercise
          );
          console.log("Rep segments:", segments);
        }
      }
    }

    if (exercises.length > 0) {
      const firstExercise = exercises[0];
      console.log(`\n10. Listing instances for exercise "${firstExercise}"...`);
      const instances = await remote.listInstances(firstExercise);
      console.log(
        "Instances:",
        instances.map(([subject, path]) => ({ subject, path }))
      );
    }

    console.log("\n✅ All tests passed!");
  } catch (error) {
    console.error("\n❌ Test failed:", error);
    console.error("\nMake sure the server is running on http://localhost:3001");
    console.error("Run: npm run server");
  }
}

// Run the test
testRemoteDataset().catch(console.error);
