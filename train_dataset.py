import os
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class TrainDataset:
    """Utility class to navigate the local training dataset.

    The dataset folder is excluded from version control, so this class
    attempts to hide the exact file layout.  It expects a directory
    structure similar to::

        train/
            s01/
                rep_ann.json
                joints3d_25/
                    squat.json
                    ...
            s02/
                rep_ann.json
                joints3d_25/
                    pushup.json
                    ...

    Each JSON file is expected to contain a key ``"joints3d_25"`` (or a
    similar one) that stores the pose array.  Additional metadata such as
    repetition timings will be returned if present. Repetition timing
    information is primarily loaded from the rep_ann.json file in each
    subject's directory.
    """

    def __init__(self, root: str = "train") -> None:
        self.root = os.fspath(root)

    # ------------------------------------------------------------------
    # listing utilities
    # ------------------------------------------------------------------
    def list_subjects(self) -> List[str]:
        """Return all available subject IDs."""
        if not os.path.isdir(self.root):
            return []
        return sorted(
            d
            for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        )

    def list_exercises_for_subject(self, subject: str) -> List[str]:
        """Return exercise names performed by ``subject``."""
        path = os.path.join(self.root, subject, "joints3d_25")
        if not os.path.isdir(path):
            return []
        return sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(path)
            if f.endswith(".json") and os.path.isfile(os.path.join(path, f))
        )

    def list_all_exercises(self) -> List[str]:
        """Return a sorted list of all unique exercises in the dataset."""
        exercises = set()
        for subject in self.list_subjects():
            exercises.update(self.list_exercises_for_subject(subject))
        return sorted(exercises)

    def list_instances(self, exercise: str) -> List[Tuple[str, str]]:
        """Return ``(subject, path)`` tuples for every instance of ``exercise``."""
        instances: List[Tuple[str, str]] = []
        for subject in self.list_subjects():
            path = os.path.join(self.root, subject, "joints3d_25", f"{exercise}.json")
            if os.path.isfile(path):
                instances.append((subject, path))
        return instances

    # ------------------------------------------------------------------
    # loading utilities
    # ------------------------------------------------------------------
    def load_rep_annotations(self, subject: str) -> Optional[Dict[str, List[int]]]:
        """Load the repetition annotations for a given subject.

        Returns a dictionary mapping exercise names to lists of frame numbers
        representing repetition timing annotations, or None if the file doesn't exist.
        """
        rep_ann_path = os.path.join(self.root, subject, "rep_ann.json")
        if not os.path.isfile(rep_ann_path):
            return None

        try:
            with open(rep_ann_path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, IOError):
            return None

    def load_instance(self, subject: str, exercise: str) -> Dict[str, Any]:
        """Load the full JSON for ``subject`` performing ``exercise``."""
        path = os.path.join(self.root, subject, "joints3d_25", f"{exercise}.json")
        with open(path) as fh:
            data = json.load(fh)

        poses = None
        for key in ("joints3d_25", "joints3d", "poses3d"):
            if key in data:
                poses = np.asarray(data[key])
                break
        if poses is None:
            raise KeyError(f"No pose data found in {path}")

        info = {
            k: v
            for k, v in data.items()
            if k not in {"joints3d_25", "joints3d", "poses3d"}
        }

        # First, try to get timings from rep_ann.json
        timings = None
        rep_annotations = self.load_rep_annotations(subject)
        if rep_annotations and exercise in rep_annotations:
            timings = rep_annotations[exercise]

        # Fall back to timings from the individual exercise file if not found in rep_ann.json
        if timings is None:
            timings = info.get("rep_timings") or info.get("reps") or info.get("timings")

        num_reps = len(timings) if isinstance(timings, list) else info.get("num_reps")

        return {
            "subject": subject,
            "exercise": exercise,
            "poses": poses,
            "timings": timings,
            "num_reps": num_reps,
            "info": info,
        }

    # ------------------------------------------------------------------
    # convenience helpers
    # ------------------------------------------------------------------
    def get_pose_array(self, subject: str, exercise: str) -> np.ndarray:
        """Return only the pose array for ``subject`` and ``exercise``."""
        return self.load_instance(subject, exercise)["poses"]

    def subject_has_exercise(self, subject: str, exercise: str) -> bool:
        """Check whether ``subject`` contains data for ``exercise``."""
        path = os.path.join(self.root, subject, "joints3d_25", f"{exercise}.json")
        return os.path.isfile(path)

    def get_rep_timings(self, subject: str, exercise: str) -> Optional[List[int]]:
        """Get repetition timings for a specific subject and exercise.

        Returns the list of frame numbers representing repetition boundaries,
        or None if no timing information is available.
        """
        instance_data = self.load_instance(subject, exercise)
        return instance_data["timings"]

    def get_rep_segments(
        self, subject: str, exercise: str
    ) -> Optional[List[Tuple[int, int]]]:
        """Get repetition segments as (start, end) frame pairs.

        Returns a list of (start_frame, end_frame) tuples representing individual
        repetitions, or None if no timing information is available.
        """
        timings = self.get_rep_timings(subject, exercise)
        if timings is None or len(timings) < 2:
            return None

        return [(timings[i], timings[i + 1]) for i in range(len(timings) - 1)]
