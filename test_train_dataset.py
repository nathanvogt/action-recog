#!/usr/bin/env python3
"""
Test script to demonstrate TrainDataset functionality
"""

from train_dataset import TrainDataset
import os


def main():
    print("=" * 60)
    print("TrainDataset Functionality Test")
    print("=" * 60)

    # Initialize the dataset
    dataset = TrainDataset()
    print(f"Dataset root directory: {dataset.root}")
    print(f"Root directory exists: {os.path.exists(dataset.root)}")
    print()

    # Test listing subjects
    print("1. Available Subjects:")
    print("-" * 30)
    subjects = dataset.list_subjects()
    if subjects:
        for i, subject in enumerate(subjects, 1):
            print(f"   {i}. {subject}")
    else:
        print("   No subjects found (dataset directory may not exist)")
    print(f"   Total subjects: {len(subjects)}")
    print()

    # Test listing all exercises
    print("2. All Available Exercises:")
    print("-" * 30)
    all_exercises = dataset.list_all_exercises()
    if all_exercises:
        for i, exercise in enumerate(all_exercises, 1):
            print(f"   {i}. {exercise}")
    else:
        print("   No exercises found")
    print(f"   Total exercises: {len(all_exercises)}")
    print()

    # Test exercises for each subject
    if subjects:
        print("3. Exercises by Subject:")
        print("-" * 30)
        for subject in subjects:
            exercises = dataset.list_exercises_for_subject(subject)
            print(f"   {subject}: {len(exercises)} exercises")
            if exercises:
                for exercise in exercises[:3]:  # Show first 3
                    print(f"      - {exercise}")
                if len(exercises) > 3:
                    print(f"      ... and {len(exercises) - 3} more")
            print()

    # Test instances for a specific exercise
    if all_exercises:
        test_exercise = all_exercises[0]  # Use first exercise
        print(f"4. Instances of '{test_exercise}':")
        print("-" * 30)
        instances = dataset.list_instances(test_exercise)
        if instances:
            for subject, path in instances:
                print(f"   Subject: {subject}")
                print(f"   Path: {path}")
                print(f"   File exists: {os.path.exists(path)}")

                # Test if subject has this exercise
                has_exercise = dataset.subject_has_exercise(subject, test_exercise)
                print(f"   Has exercise (verification): {has_exercise}")
                print()
        else:
            print(f"   No instances found for '{test_exercise}'")
        print()

    # Test loading an instance (if available)
    if subjects and all_exercises:
        test_subject = subjects[0]
        test_exercise = all_exercises[0]

        if dataset.subject_has_exercise(test_subject, test_exercise):
            print(f"5. Loading Instance: {test_subject} - {test_exercise}")
            print("-" * 30)
            try:
                instance_data = dataset.load_instance(test_subject, test_exercise)
                print(f"   Subject: {instance_data['subject']}")
                print(f"   Exercise: {instance_data['exercise']}")
                print(f"   Poses shape: {instance_data['poses'].shape}")
                print(f"   Number of reps: {instance_data['num_reps']}")
                print(f"   Has timings: {instance_data['timings'] is not None}")
                print(f"   Additional info keys: {list(instance_data['info'].keys())}")

                # Test getting just the pose array
                poses = dataset.get_pose_array(test_subject, test_exercise)
                print(f"   Pose array shape (direct): {poses.shape}")
                print(f"   Pose array dtype: {poses.dtype}")

            except Exception as e:
                print(f"   Error loading instance: {e}")
            print()

    print("=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
