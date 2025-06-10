import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


class InstanceData:
    """Data class representing an instance of exercise data."""

    def __init__(
        self,
        subject: str,
        exercise: str,
        poses: np.ndarray,
        timings: Optional[List[float]],
        num_reps: Optional[int],
        info: Dict[str, Any],
    ):
        self.subject = subject
        self.exercise = exercise
        self.poses = poses  # numpy array [frame, kp, 3d point]
        self.timings = timings
        self.num_reps = num_reps
        self.info = info


class TrainDatasetLocal:
    """Local implementation of the training dataset interface."""

    def __init__(self, root: str = "train"):
        """Initialize the dataset with the root directory path.

        Args:
            root: Root directory path for the dataset (defaults to "train")
        """
        self.root = os.path.abspath(root)

    # ------------------------------------------------------------------
    # listing utilities
    # ------------------------------------------------------------------

    def list_subjects(self) -> List[str]:
        """List all subject directories in the dataset.

        Returns:
            Sorted list of subject directory names
        """
        if not os.path.exists(self.root):
            return []

        subjects = []
        for item in os.listdir(self.root):
            item_path = os.path.join(self.root, item)
            if os.path.isdir(item_path):
                subjects.append(item)

        return sorted(subjects)

    def list_exercises_for_subject(self, subject: str) -> List[str]:
        """List all exercises available for a given subject.

        Args:
            subject: Subject name

        Returns:
            Sorted list of exercise names (without .json extension)
        """
        joints_path = os.path.join(self.root, subject, "joints3d_25")
        if not os.path.exists(joints_path):
            return []

        exercises = []
        for file in os.listdir(joints_path):
            file_path = os.path.join(joints_path, file)
            if file.endswith(".json") and os.path.isfile(file_path):
                # Remove .json extension
                exercise_name = os.path.splitext(file)[0]
                exercises.append(exercise_name)

        return sorted(exercises)

    def list_all_exercises(self) -> List[str]:
        """List all unique exercises across all subjects.

        Returns:
            Sorted list of unique exercise names
        """
        exercises = set()
        for subject in self.list_subjects():
            for exercise in self.list_exercises_for_subject(subject):
                exercises.add(exercise)

        return sorted(list(exercises))

    def list_instances(self, exercise: str) -> List[Tuple[str, str]]:
        """List all instances (subject, file_path pairs) for a given exercise.

        Args:
            exercise: Exercise name

        Returns:
            List of tuples containing (subject, file_path)
        """
        instances = []
        for subject in self.list_subjects():
            file_path = os.path.join(
                self.root, subject, "joints3d_25", f"{exercise}.json"
            )
            if os.path.exists(file_path):
                instances.append((subject, file_path))

        return instances

    def list_camera_ids(self, subject: str) -> List[str]:
        """List all camera IDs available for a given subject.

        Args:
            subject: Subject name

        Returns:
            Sorted list of camera ID directory names
        """
        camera_path = os.path.join(self.root, subject, "camera_parameters")
        if not os.path.exists(camera_path):
            return []

        camera_ids = []
        for item in os.listdir(camera_path):
            item_path = os.path.join(camera_path, item)
            if os.path.isdir(item_path):
                camera_ids.append(item)

        return sorted(camera_ids)

    # ------------------------------------------------------------------
    # loading utilities
    # ------------------------------------------------------------------

    def load_rep_annotations(self, subject: str) -> Optional[Dict[str, List[float]]]:
        """Load repetition annotations for a subject.

        Args:
            subject: Subject name

        Returns:
            Dictionary mapping exercise names to repetition timings, or None if not found
        """
        annotations_path = os.path.join(self.root, subject, "rep_ann.json")
        if not os.path.exists(annotations_path):
            return None

        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def load_instance(self, subject: str, exercise: str) -> InstanceData:
        """Load instance data for a specific subject and exercise.

        Args:
            subject: Subject name
            exercise: Exercise name

        Returns:
            InstanceData object containing poses, timings, and metadata

        Raises:
            FileNotFoundError: If the exercise file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
            ValueError: If no pose data is found in the file
        """
        file_path = os.path.join(self.root, subject, "joints3d_25", f"{exercise}.json")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Look for pose data under different possible keys
        poses = None
        for key in ["joints3d_25", "joints3d", "poses3d"]:
            if key in data:
                poses = data[key]
                break

        if poses is None:
            raise ValueError(f"No pose data found in {file_path}")

        # Convert poses to numpy array for faster operations
        poses = np.array(poses)

        # Extract metadata (everything except pose data)
        info = {}
        for key, value in data.items():
            if key not in ["joints3d_25", "joints3d", "poses3d"]:
                info[key] = value

        # Load timings from rep annotations first, then fallback to instance data
        timings = None
        rep_annotations = self.load_rep_annotations(subject)
        if rep_annotations and exercise in rep_annotations:
            timings = rep_annotations[exercise]

        if timings is None:
            # Look for timings in instance data under various keys
            timings = info.get("rep_timings")
            if timings is None:
                timings = info.get("reps")
            if timings is None:
                timings = info.get("timings")

        # Get number of reps
        num_reps = len(timings) if isinstance(timings, list) else info.get("num_reps")

        return InstanceData(
            subject=subject,
            exercise=exercise,
            poses=poses,
            timings=timings,
            num_reps=num_reps,
            info=info,
        )

    # ------------------------------------------------------------------
    # convenience helpers
    # ------------------------------------------------------------------

    def get_pose_array(self, subject: str, exercise: str) -> np.ndarray:
        """Get the pose array for a specific subject and exercise.

        Args:
            subject: Subject name
            exercise: Exercise name

        Returns:
            3D pose numpy array with shape [frame, keypoint, 3] containing (x, y, z) coordinates
        """
        return self.load_instance(subject, exercise).poses

    def subject_has_exercise(self, subject: str, exercise: str) -> bool:
        """Check if a subject has data for a specific exercise.

        Args:
            subject: Subject name
            exercise: Exercise name

        Returns:
            True if the exercise file exists for the subject
        """
        file_path = os.path.join(self.root, subject, "joints3d_25", f"{exercise}.json")
        return os.path.exists(file_path)

    def get_rep_timings(self, subject: str, exercise: str) -> Optional[List[float]]:
        """Get repetition timings for a specific subject and exercise.

        Args:
            subject: Subject name
            exercise: Exercise name

        Returns:
            List of repetition timing markers, or None if not available
        """
        return self.load_instance(subject, exercise).timings

    def get_rep_segments(
        self, subject: str, exercise: str
    ) -> Optional[List[Tuple[float, float]]]:
        """Get repetition segments as start-end pairs.

        Args:
            subject: Subject name
            exercise: Exercise name

        Returns:
            List of (start_time, end_time) tuples for each repetition, or None if not available
        """
        timings = self.get_rep_timings(subject, exercise)
        if not timings or len(timings) < 2:
            return None

        segments = []
        for i in range(len(timings) - 1):
            segments.append((timings[i], timings[i + 1]))

        return segments

    def get_video_blob(self, subject: str, exercise: str, camera_id: str) -> bytes:
        """Get video data as bytes for a specific subject, exercise, and camera.

        Args:
            subject: Subject name
            exercise: Exercise name
            camera_id: Camera identifier

        Returns:
            Video file content as bytes

        Raises:
            FileNotFoundError: If no video file is found for the given parameters
        """
        # Try common video file extensions and locations
        possible_paths = [
            os.path.join(self.root, subject, "videos", camera_id, f"{exercise}.mp4"),
            os.path.join(self.root, subject, "videos", camera_id, f"{exercise}.webm"),
            os.path.join(self.root, subject, "videos", camera_id, f"{exercise}.mov"),
            os.path.join(self.root, subject, camera_id, f"{exercise}.mp4"),
            os.path.join(self.root, subject, camera_id, f"{exercise}.webm"),
            os.path.join(self.root, subject, camera_id, f"{exercise}.mov"),
        ]

        for video_path in possible_paths:
            if os.path.exists(video_path):
                with open(video_path, "rb") as f:
                    return f.read()

        raise FileNotFoundError(
            f"Video not found for subject: {subject}, exercise: {exercise}, camera: {camera_id}"
        )

    def get_mime_type_from_path(self, file_path: str) -> str:
        """Get MIME type based on file extension.

        Args:
            file_path: Path to the video file

        Returns:
            MIME type string
        """
        ext = os.path.splitext(file_path)[1].lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
        }
        return mime_types.get(ext, "video/mp4")  # Default fallback
