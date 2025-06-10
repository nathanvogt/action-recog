#!/usr/bin/env python3
"""Test script to demonstrate TrainDataset functionality"""

import os
import sys
from train_dataset import TrainDatasetLocal


def main():
    print("=" * 60)
    print("TrainDataset Functionality Test")
    print("=" * 60)

    dataset = TrainDatasetLocal()
    print(f"Dataset root directory: {dataset.root}")
    print(f"Root directory exists: {os.path.exists(dataset.root)}")
    print()

    # 1. List subjects
    print("1. Available Subjects:")
    print("-" * 30)
    subjects = dataset.list_subjects()
    if subjects:
        for idx, subject in enumerate(subjects, 1):
            print(f"   {idx}. {subject}")
    else:
        print("   No subjects found (dataset directory may not exist)")
    print(f"   Total subjects: {len(subjects)}")
    print()

    # 2. List all exercises
    print("2. All Available Exercises:")
    print("-" * 30)
    all_exercises = dataset.list_all_exercises()
    if all_exercises:
        for idx, exercise in enumerate(all_exercises, 1):
            print(f"   {idx}. {exercise}")
    else:
        print("   No exercises found")
    print(f"   Total exercises: {len(all_exercises)}")
    print()

    # 3. Exercises by subject
    if subjects:
        print("3. Exercises by Subject:")
        print("-" * 30)
        for subject in subjects:
            exercises = dataset.list_exercises_for_subject(subject)
            print(f"   {subject}: {len(exercises)} exercises")
            if exercises:
                # Show first 3 exercises
                for exercise in exercises[:3]:
                    print(f"      - {exercise}")
                if len(exercises) > 3:
                    print(f"      ... and {len(exercises) - 3} more")
            print()

    # 4. Instances for a specific exercise
    if all_exercises:
        test_exercise = all_exercises[0]
        print(f"4. Instances of '{test_exercise}':")
        print("-" * 30)
        instances = dataset.list_instances(test_exercise)
        if instances:
            for subject, path in instances:
                print(f"   Subject: {subject}")
                print(f"   Path: {path}")
                print(f"   File exists: {os.path.exists(path)}")

                has_exercise = dataset.subject_has_exercise(subject, test_exercise)
                print(f"   Has exercise (verification): {has_exercise}")
                print()
        else:
            print(f"   No instances found for '{test_exercise}'")
        print()

    # 5. Load an instance
    if subjects and all_exercises:
        test_subject = subjects[0]
        test_exercise = all_exercises[0]
        if dataset.subject_has_exercise(test_subject, test_exercise):
            print(f"5. Loading Instance: {test_subject} - {test_exercise}")
            print("-" * 30)
            try:
                instance = dataset.load_instance(test_subject, test_exercise)
                print(f"   Subject: {instance.subject}")
                print(f"   Exercise: {instance.exercise}")

                # Check poses shape
                poses_length = len(instance.poses)
                first_pose_length = (
                    len(instance.poses[0]) if len(instance.poses) > 0 else 0
                )
                print(f"   Poses shape: [{poses_length}, {first_pose_length}]")
                print(f"   Number of reps: {instance.num_reps}")
                print(f"   Has timings: {instance.timings is not None}")

                # Print timing details if available
                if instance.timings is not None:
                    print("   Rep timings (frame indices):")
                    for idx, timing in enumerate(instance.timings, 1):
                        if isinstance(timing, list) and len(timing) == 2:
                            frame_count = timing[1] - timing[0] + 1
                            print(
                                f"      Rep {idx}: frames {timing[0]} - {timing[1]} ({frame_count} frames)"
                            )
                        elif isinstance(timing, (int, float)):
                            print(f"      Rep {idx}: frame {timing}")
                        else:
                            print(f"      Rep {idx}: {timing}")

                print(f"   Additional info keys: {list(instance.info.keys())}")

                # Test direct pose array access
                poses = dataset.get_pose_array(test_subject, test_exercise)
                poses_length = len(poses)
                first_pose_length = len(poses[0]) if len(poses) > 0 else 0
                print(
                    f"   Pose array shape (direct): [{poses_length}, {first_pose_length}]"
                )

                # Test rep segments if available
                segments = dataset.get_rep_segments(test_subject, test_exercise)
                if segments:
                    print(f"   Rep segments: {len(segments)} segments")
                    for idx, (start, end) in enumerate(segments[:3], 1):
                        print(f"      Segment {idx}: {start} - {end}")
                    if len(segments) > 3:
                        print(f"      ... and {len(segments) - 3} more segments")

            except Exception as err:
                print(f"   Error loading instance: {err}")
            print()

    # 6. Test camera functionality if subjects exist
    if subjects:
        test_subject = subjects[0]
        print(f"6. Camera Information for '{test_subject}':")
        print("-" * 30)
        try:
            camera_ids = dataset.list_camera_ids(test_subject)
            if camera_ids:
                print(f"   Available cameras: {camera_ids}")
                for camera_id in camera_ids[:3]:  # Show first 3 cameras
                    print(f"      - {camera_id}")
                if len(camera_ids) > 3:
                    print(f"      ... and {len(camera_ids) - 3} more")
            else:
                print("   No camera information found")
        except Exception as err:
            print(f"   Error accessing camera info: {err}")
        print()

    # 7. Test rep annotations
    if subjects:
        test_subject = subjects[0]
        print(f"7. Repetition Annotations for '{test_subject}':")
        print("-" * 30)
        try:
            rep_annotations = dataset.load_rep_annotations(test_subject)
            if rep_annotations:
                print(f"   Found annotations for {len(rep_annotations)} exercises:")
                for exercise, timings in list(rep_annotations.items())[:3]:
                    print(
                        f"      {exercise}: {len(timings) if isinstance(timings, list) else 'N/A'} timing markers"
                    )
                if len(rep_annotations) > 3:
                    print(f"      ... and {len(rep_annotations) - 3} more exercises")
            else:
                print("   No repetition annotations found")
        except Exception as err:
            print(f"   Error loading rep annotations: {err}")
        print()

    print("=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
