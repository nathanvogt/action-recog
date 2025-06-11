import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ReplayClient, ReplayList, ReplayFile } from "../../libs/replayClient";

interface ReplayMenuProps {
  onSelectReplay?: (filename: string) => void;
}

export const ReplayMenu: React.FC<ReplayMenuProps> = ({ onSelectReplay }) => {
  const [replays, setReplays] = useState<ReplayList>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSubjects, setExpandedSubjects] = useState<Set<string>>(
    new Set()
  );
  const navigate = useNavigate();

  const replayClient = new ReplayClient();

  useEffect(() => {
    loadReplays();
  }, []);

  const loadReplays = async () => {
    try {
      setLoading(true);
      const replayData = await replayClient.listReplays();
      setReplays(replayData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load replays");
    } finally {
      setLoading(false);
    }
  };

  const toggleSubject = (subject: string) => {
    const newExpanded = new Set(expandedSubjects);
    if (newExpanded.has(subject)) {
      newExpanded.delete(subject);
    } else {
      newExpanded.add(subject);
    }
    setExpandedSubjects(newExpanded);
  };

  const handleSelectReplay = (filename: string) => {
    if (onSelectReplay) {
      onSelectReplay(filename);
    } else {
      // Navigate to replay viewer
      navigate(`/replay/${filename}`);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <div className="replay-menu loading">
        <h2>Model Replays</h2>
        <p>Loading replays...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="replay-menu error">
        <h2>Model Replays</h2>
        <p className="error-message">Error: {error}</p>
        <button onClick={loadReplays}>Retry</button>
      </div>
    );
  }

  const subjects = Object.keys(replays);

  if (subjects.length === 0) {
    return (
      <div className="replay-menu empty">
        <h2>Model Replays</h2>
        <p>
          No replay files found. Run evaluations with --save-replay to generate
          replays.
        </p>
        <button onClick={loadReplays}>Refresh</button>
      </div>
    );
  }

  return (
    <div className="replay-menu">
      <div className="replay-menu-header">
        <h2>Model Replays</h2>
        <div className="header-actions">
          <button className="nav-button" onClick={() => navigate("/menu")}>
            Back to Menu
          </button>
          <button className="refresh-button" onClick={loadReplays}>
            Refresh
          </button>
        </div>
      </div>

      <div className="replay-tree">
        {subjects.map((subject) => (
          <div key={subject} className="subject-group">
            <div
              className="subject-header"
              onClick={() => toggleSubject(subject)}
            >
              <span
                className={`expand-icon ${
                  expandedSubjects.has(subject) ? "expanded" : ""
                }`}
              >
                ▶
              </span>
              <span className="subject-name">{subject}</span>
              <span className="exercise-count">
                ({Object.keys(replays[subject]).length} exercises)
              </span>
            </div>

            {expandedSubjects.has(subject) && (
              <div className="exercise-list">
                {Object.entries(replays[subject]).map(
                  ([exercise, replayFiles]) => (
                    <div key={exercise} className="exercise-group">
                      <div className="exercise-header">
                        <span className="exercise-name">{exercise}</span>
                        <span className="replay-count">
                          ({replayFiles.length} replays)
                        </span>
                      </div>

                      <div className="replay-files">
                        {replayFiles.map((replayFile: ReplayFile) => (
                          <div
                            key={replayFile.filename}
                            className="replay-file"
                            onClick={() =>
                              handleSelectReplay(replayFile.filename)
                            }
                          >
                            <div className="replay-info">
                              <span className="replay-filename">
                                {replayFile.filename}
                              </span>
                              <span className="replay-date">
                                {formatDate(replayFile.lastModified)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <style>{`
        .replay-menu {
          padding: 20px;
          max-width: 800px;
          margin: 0 auto;
          background: #f8f9fa;
          border-radius: 8px;
          min-height: 80vh;
        }
        
        .replay-menu-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
          padding-bottom: 12px;
          border-bottom: 2px solid #dee2e6;
        }
        
        .replay-menu h2 {
          margin: 0;
          color: #343a40;
          font-size: 24px;
        }
        
        .header-actions {
          display: flex;
          gap: 10px;
        }
        
        .nav-button, .refresh-button {
          padding: 8px 16px;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: all 0.2s;
        }
        
        .nav-button {
          background: #6c757d;
          color: white;
        }
        
        .nav-button:hover {
          background: #5a6268;
        }
        
        .refresh-button {
          background: #007bff;
          color: white;
        }
        
        .refresh-button:hover {
          background: #0056b3;
        }
        
        .replay-tree {
          max-height: 70vh;
          overflow-y: auto;
        }
        
        .subject-group {
          margin-bottom: 12px;
        }
        
        .subject-header {
          display: flex;
          align-items: center;
          padding: 12px;
          background: #e9ecef;
          border-radius: 6px;
          cursor: pointer;
          user-select: none;
          transition: background-color 0.2s;
        }
        
        .subject-header:hover {
          background: #dee2e6;
        }
        
        .expand-icon {
          margin-right: 10px;
          transition: transform 0.2s;
          font-size: 14px;
          color: #495057;
        }
        
        .expand-icon.expanded {
          transform: rotate(90deg);
        }
        
        .subject-name {
          font-weight: 600;
          color: #495057;
          font-size: 16px;
        }
        
        .exercise-count, .replay-count {
          margin-left: auto;
          font-size: 12px;
          color: #6c757d;
          background: #fff;
          padding: 2px 8px;
          border-radius: 12px;
        }
        
        .exercise-list {
          padding-left: 24px;
          margin-top: 8px;
        }
        
        .exercise-group {
          margin-bottom: 10px;
        }
        
        .exercise-header {
          display: flex;
          align-items: center;
          padding: 8px 12px;
          background: #f8f9fa;
          border-radius: 6px;
          border-left: 4px solid #007bff;
        }
        
        .exercise-name {
          font-weight: 500;
          color: #495057;
          font-size: 15px;
        }
        
        .replay-files {
          padding-left: 20px;
          margin-top: 6px;
        }
        
        .replay-file {
          padding: 10px 12px;
          background: white;
          border: 1px solid #dee2e6;
          border-radius: 6px;
          margin-bottom: 4px;
          cursor: pointer;
          transition: all 0.2s;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .replay-file:hover {
          background: #e3f2fd;
          border-color: #007bff;
          transform: translateY(-1px);
          box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }
        
        .replay-info {
          display: flex;
          flex-direction: column;
        }
        
        .replay-filename {
          font-size: 14px;
          color: #495057;
          font-weight: 500;
        }
        
        .replay-date {
          font-size: 12px;
          color: #6c757d;
          margin-top: 4px;
        }
        
        .error-message {
          color: #dc3545;
          margin: 12px 0;
          font-weight: 500;
        }
        
        .loading, .error, .empty {
          text-align: center;
          padding: 60px 20px;
        }
        
        .loading p, .empty p {
          color: #6c757d;
          font-size: 16px;
        }
        
        .error button, .empty button {
          margin-top: 16px;
          padding: 10px 20px;
          background: #007bff;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
        }
        
        .error button:hover, .empty button:hover {
          background: #0056b3;
        }
      `}</style>
    </div>
  );
};
