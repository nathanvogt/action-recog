// Test script to demonstrate TrainDataset functionality
import fs from 'fs';
import { TrainDataset } from './trainDataset';

function main() {
  console.log('='.repeat(60));
  console.log('TrainDataset Functionality Test');
  console.log('='.repeat(60));

  const dataset = new TrainDataset();
  console.log(`Dataset root directory: ${dataset.root}`);
  console.log(`Root directory exists: ${fs.existsSync(dataset.root)}`);
  console.log();

  // 1. List subjects
  console.log('1. Available Subjects:');
  console.log('-'.repeat(30));
  const subjects = dataset.listSubjects();
  if (subjects.length) {
    subjects.forEach((subject, idx) => {
      console.log(`   ${idx + 1}. ${subject}`);
    });
  } else {
    console.log('   No subjects found (dataset directory may not exist)');
  }
  console.log(`   Total subjects: ${subjects.length}`);
  console.log();

  // 2. List all exercises
  console.log('2. All Available Exercises:');
  console.log('-'.repeat(30));
  const allExercises = dataset.listAllExercises();
  if (allExercises.length) {
    allExercises.forEach((ex, idx) => {
      console.log(`   ${idx + 1}. ${ex}`);
    });
  } else {
    console.log('   No exercises found');
  }
  console.log(`   Total exercises: ${allExercises.length}`);
  console.log();

  // 3. Exercises by subject
  if (subjects.length) {
    console.log('3. Exercises by Subject:');
    console.log('-'.repeat(30));
    for (const subject of subjects) {
      const exercises = dataset.listExercisesForSubject(subject);
      console.log(`   ${subject}: ${exercises.length} exercises`);
      if (exercises.length) {
        exercises.slice(0, 3).forEach((ex) => console.log(`      - ${ex}`));
        if (exercises.length > 3) {
          console.log(`      ... and ${exercises.length - 3} more`);
        }
      }
      console.log();
    }
  }

  // 4. Instances for a specific exercise
  if (allExercises.length) {
    const testExercise = allExercises[0];
    console.log(`4. Instances of '${testExercise}':`);
    console.log('-'.repeat(30));
    const instances = dataset.listInstances(testExercise);
    if (instances.length) {
      for (const [subject, p] of instances) {
        console.log(`   Subject: ${subject}`);
        console.log(`   Path: ${p}`);
        console.log(`   File exists: ${fs.existsSync(p)}`);

        const hasExercise = dataset.subjectHasExercise(subject, testExercise);
        console.log(`   Has exercise (verification): ${hasExercise}`);
        console.log();
      }
    } else {
      console.log(`   No instances found for '${testExercise}'`);
    }
    console.log();
  }

  // 5. Load an instance
  if (subjects.length && allExercises.length) {
    const testSubject = subjects[0];
    const testExercise = allExercises[0];
    if (dataset.subjectHasExercise(testSubject, testExercise)) {
      console.log(`5. Loading Instance: ${testSubject} - ${testExercise}`);
      console.log('-'.repeat(30));
      try {
        const instance = dataset.loadInstance(testSubject, testExercise);
        console.log(`   Subject: ${instance.subject}`);
        console.log(`   Exercise: ${instance.exercise}`);
        console.log(
          `   Poses shape: [${instance.poses.length}, ${instance.poses[0]?.length ?? 0}]`
        );
        console.log(`   Number of reps: ${instance.num_reps}`);
        console.log(`   Has timings: ${instance.timings !== null}`);
        console.log(`   Additional info keys: ${Object.keys(instance.info)}`);

        const poses = dataset.getPoseArray(testSubject, testExercise);
        console.log(
          `   Pose array shape (direct): [${poses.length}, ${poses[0]?.length ?? 0}]`
        );
      } catch (err) {
        console.log(`   Error loading instance: ${err}`);
      }
      console.log();
    }
  }

  console.log('='.repeat(60));
  console.log('Test completed!');
  console.log('='.repeat(60));
}

main();
