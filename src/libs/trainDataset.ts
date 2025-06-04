import fs from 'fs';
import path from 'path';

export interface InstanceData {
  subject: string;
  exercise: string;
  poses: number[][][];
  timings: number[] | null;
  num_reps: number | undefined;
  info: Record<string, any>;
}

export class TrainDataset {
  root: string;

  constructor(root: string = 'train') {
    this.root = path.resolve(root);
  }

  // ------------------------------------------------------------------
  // listing utilities
  // ------------------------------------------------------------------
  listSubjects(): string[] {
    if (!fs.existsSync(this.root)) return [];
    return fs
      .readdirSync(this.root)
      .filter((d) => fs.statSync(path.join(this.root, d)).isDirectory())
      .sort();
  }

  listExercisesForSubject(subject: string): string[] {
    const p = path.join(this.root, subject, 'joints3d_25');
    if (!fs.existsSync(p)) return [];
    return fs
      .readdirSync(p)
      .filter((f) => f.endsWith('.json') && fs.statSync(path.join(p, f)).isFile())
      .map((f) => path.parse(f).name)
      .sort();
  }

  listAllExercises(): string[] {
    const ex = new Set<string>();
    for (const s of this.listSubjects()) {
      for (const e of this.listExercisesForSubject(s)) {
        ex.add(e);
      }
    }
    return Array.from(ex).sort();
  }

  listInstances(exercise: string): Array<[string, string]> {
    const instances: Array<[string, string]> = [];
    for (const s of this.listSubjects()) {
      const p = path.join(this.root, s, 'joints3d_25', `${exercise}.json`);
      if (fs.existsSync(p)) instances.push([s, p]);
    }
    return instances;
  }

  // ------------------------------------------------------------------
  // loading utilities
  // ------------------------------------------------------------------
  loadRepAnnotations(subject: string): Record<string, number[]> | null {
    const p = path.join(this.root, subject, 'rep_ann.json');
    if (!fs.existsSync(p)) return null;
    try {
      const txt = fs.readFileSync(p, 'utf8');
      return JSON.parse(txt);
    } catch {
      return null;
    }
  }

  loadInstance(subject: string, exercise: string): InstanceData {
    const p = path.join(this.root, subject, 'joints3d_25', `${exercise}.json`);
    const data = JSON.parse(fs.readFileSync(p, 'utf8')) as Record<string, any>;

    let poses: number[][][] | null = null;
    for (const key of ['joints3d_25', 'joints3d', 'poses3d']) {
      if (key in data) {
        poses = data[key];
        break;
      }
    }
    if (!poses) throw new Error(`No pose data found in ${p}`);

    const info: Record<string, any> = {};
    for (const [k, v] of Object.entries(data)) {
      if (!['joints3d_25', 'joints3d', 'poses3d'].includes(k)) {
        info[k] = v;
      }
    }

    let timings: number[] | null = null;
    const rep = this.loadRepAnnotations(subject);
    if (rep && exercise in rep) timings = rep[exercise];
    if (!timings) {
      timings = (info['rep_timings'] || info['reps'] || info['timings']) ?? null;
    }

    const num_reps = Array.isArray(timings) ? timings.length : info['num_reps'];

    return { subject, exercise, poses, timings, num_reps, info };
  }

  // ------------------------------------------------------------------
  // convenience helpers
  // ------------------------------------------------------------------
  getPoseArray(subject: string, exercise: string): number[][][] {
    return this.loadInstance(subject, exercise).poses;
  }

  subjectHasExercise(subject: string, exercise: string): boolean {
    const p = path.join(this.root, subject, 'joints3d_25', `${exercise}.json`);
    return fs.existsSync(p);
  }

  getRepTimings(subject: string, exercise: string): number[] | null {
    return this.loadInstance(subject, exercise).timings;
  }

  getRepSegments(subject: string, exercise: string): Array<[number, number]> | null {
    const t = this.getRepTimings(subject, exercise);
    if (!t || t.length < 2) return null;
    const seg: Array<[number, number]> = [];
    for (let i = 0; i < t.length - 1; i++) {
      seg.push([t[i], t[i + 1]]);
    }
    return seg;
  }
}
