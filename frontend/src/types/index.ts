// ─── Settings ──────────────────────────────────────────────────────────────

export type StainType = "KI67" | "ER" | "PR";
export type Resolution = "40x" | "20x" | "10x";
export type SourceMode = "file" | "camera";

export interface Settings {
  stain: StainType;
  resolution: Resolution;
  probThresh: number;
  nopost: boolean;
  slim: boolean;
  usePil: boolean;
  dedupRadius: number;
  everyNSec: number;
  outFps: number;
  interpSteps: number;
  maxWorkers: number;
}

export const DEFAULT_SETTINGS: Settings = {
  stain: "KI67",
  resolution: "40x",
  probThresh: 0.5,
  nopost: true,
  slim: false,
  usePil: true,
  dedupRadius: 20,
  everyNSec: 1.0,
  outFps: 5,
  interpSteps: 1,
  maxWorkers: 4,
};

// ─── Cell / Scoring ────────────────────────────────────────────────────────

export interface Cell {
  centroid: [number, number]; // [y, x]
  positive: boolean;
}

export interface Ki67Score {
  numTotal: number;
  numPos: number;
  numNeg: number;
  percentPos: number;
  cells: Cell[];
}

export interface IntensityDistribution {
  0: number;
  1: number;
  2: number;
  3: number;
}

export interface ErPrScore extends Ki67Score {
  hScore: number;
  hScoreLabel: string;
  allredScore: number;
  allredLabel: string;
  allredProportion: number;
  allredIntensity: number;
  intensityDistribution: IntensityDistribution;
}

export type ScoringResult = Ki67Score | ErPrScore;

export function isErPrScore(s: ScoringResult): s is ErPrScore {
  return "hScore" in s;
}

// ─── API Response ──────────────────────────────────────────────────────────

export interface TuroQuantResponse {
  images: Record<string, string>; // channel → base64
  scoring?: Record<string, unknown>;
  notes?: string[];
}

// ─── Analysis Results ──────────────────────────────────────────────────────

export interface FrameResult {
  images: Record<string, string>; // base64
  scoring: ScoringResult;
  overlay?: string; // base64
  cells: Cell[];
  notes: string[];
}

export interface VideoResult {
  frameResults: Record<string, FrameResult>;
  dedupScoring?: DedupScoring;
  channelVideos?: Record<string, string>; // channel → blob URL
}

export interface DedupScoring {
  numTotal: number;
  numPos: number;
  numNeg: number;
  percentPos: number;
  rawDetections: number;
  note: string;
  // ER/PR extras
  hScore?: number;
  hScoreLabel?: string;
  allredScore?: number;
  allredLabel?: string;
  allredProportion?: number;
  allredIntensity?: number;
  intensityDistribution?: IntensityDistribution;
}

// ─── App State ─────────────────────────────────────────────────────────────

export type FileType = "image" | "video" | "tiff";

export type AnalysisStatus =
  | "idle"
  | "uploading"
  | "extracting"
  | "processing"
  | "scoring"
  | "deduplicating"
  | "stitching"
  | "done"
  | "error";

export interface AnalysisState {
  status: AnalysisStatus;
  progress: number; // 0-100
  progressText: string;
  fileType?: FileType;
  // Single image results
  result?: FrameResult;
  // Video / multi-page results
  videoResult?: VideoResult;
  error?: string;
}

export interface CameraState {
  isStreaming: boolean;
  batchFrames: ImageData[];
  result?: FrameResult;
  batchResult?: DedupScoring & { overlay?: string };
}
