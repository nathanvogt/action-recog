import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import ExerciseMenu from "./menu/ExerciseMenu";
import ExerciseInstanceMenu from "./menu/ExerciseInstanceMenu";
import { VisualizePose } from "./visualize/VisualizePose";

const Main: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/menu" element={<ExerciseMenu />} />
        <Route
          path="/menu/:subject_id/:exercise_name"
          element={<ExerciseInstanceMenu />}
        />
        <Route
          path="/visualize/:subject_id/:exercise_name"
          element={<VisualizePose />}
        />
        <Route path="/" element={<Navigate to="/menu" replace />} />
      </Routes>
    </Router>
  );
};

export default Main;
