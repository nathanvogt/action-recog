export interface ReplayStep {
  timestep: number;
  observation: number[];
  action: number;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: any;
}

export interface ReplayEpisode {
  episode_number: number;
  subject: string;
  exercise: string;
  model_path: string;
  env_config: {
    c: number;
    m: number;
    tol: number;
    dataset_root: string;
  };
  initial_info: any;
  steps: ReplayStep[];
  total_reward: number;
  episode_length: number;
}

export interface ReplayData {
  metadata: {
    subject: string;
    exercise: string;
    model_path: string;
    n_episodes: number;
    env_config: {
      c: number;
      m: number;
      tol: number;
      dataset_root: string;
    };
  };
  episodes: ReplayEpisode[];
}

export interface ReplayFile {
  filename: string;
  subject: string;
  exercise: string;
  path: string;
  lastModified: string;
}

export type ReplayList = Record<string, Record<string, ReplayFile[]>>;

export class ReplayClient {
  private baseURL: string;

  constructor(baseURL: string = "http://localhost:3001") {
    this.baseURL = baseURL;
  }

  /**
   * Get list of all available replays, grouped by subject and exercise
   */
  async listReplays(): Promise<ReplayList> {
    const response = await fetch(`${this.baseURL}/api/replays`);
    if (!response.ok) {
      throw new Error(`Failed to fetch replays: ${response.statusText}`);
    }
    return await response.json();
  }

  /**
   * Get replay data for a specific file
   */
  async getReplay(filename: string): Promise<ReplayData> {
    const response = await fetch(`${this.baseURL}/api/replay/${filename}`);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch replay ${filename}: ${response.statusText}`
      );
    }
    return await response.json();
  }

  /**
   * Get pose data for a specific subject and exercise to use with replay visualization
   */
  async getPoses(subject: string, exercise: string): Promise<number[][]> {
    const response = await fetch(
      `${this.baseURL}/api/poses/${subject}/${exercise}`
    );
    if (!response.ok) {
      throw new Error(
        `Failed to fetch poses for ${subject}/${exercise}: ${response.statusText}`
      );
    }
    return await response.json();
  }
}
