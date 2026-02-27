export const IMAGE_FORMATS = [
  "png", "jpg", "jpeg", "bmp", "gif",
] as const;

export const TIFF_FORMATS = ["tif", "tiff"] as const;

export const MICROSCOPY_FORMATS = [
  "svs", "ndpi", "scn", "czi", "lif", "mrxs",
  "vms", "vmu", "qptiff",
] as const;

export const VIDEO_FORMATS = [
  "mp4", "avi", "mov", "mkv", "webm", "mpeg",
] as const;

export const ALL_FORMATS = [
  ...IMAGE_FORMATS,
  ...TIFF_FORMATS,
  ...MICROSCOPY_FORMATS,
  ...VIDEO_FORMATS,
] as const;

export const ACCEPT_STRING = ALL_FORMATS.map((f) => `.${f}`).join(",");

export const STAINS = ["KI67", "ER", "PR"] as const;
export const RESOLUTIONS = ["40x", "20x", "10x"] as const;

// Scoring thresholds (from scoring.py)
export const MIN_CELL_AREA = 15;
export const INTENSITY_THRESH_MODERATE = 50;
export const INTENSITY_THRESH_STRONG = 100;
export const MAX_IMAGE_DIM = 3000;

export const MIN_DIM_MAP: Record<string, number> = {
  "40x": 512,
  "20x": 256,
  "10x": 128,
};

// Colors matching the Seg image conventions
export const POS_COLOR = { r: 235, g: 60, b: 55 };
export const NEG_COLOR = { r: 50, g: 110, b: 230 };

export const DEEPLIIF_API_URL = "/api/infer";
export const HEALTH_API_URL = "/api/health";
