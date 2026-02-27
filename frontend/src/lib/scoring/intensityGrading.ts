import {
  INTENSITY_THRESH_MODERATE,
  INTENSITY_THRESH_STRONG,
} from "../constants";

/**
 * Grade a single cell's max intensity into 1+ (weak), 2+ (moderate), 3+ (strong).
 * Thresholds from scoring.py: <50 = 1, <100 = 2, >=100 = 3.
 */
export function intensityGrade(maxIntensity: number): 1 | 2 | 3 {
  if (maxIntensity < INTENSITY_THRESH_MODERATE) return 1;
  if (maxIntensity < INTENSITY_THRESH_STRONG) return 2;
  return 3;
}

/**
 * Allred proportion score (0-5) from percent positive.
 */
export function allredProportionScore(pctPos: number): number {
  if (pctPos === 0) return 0;
  if (pctPos <= 1) return 1;
  if (pctPos <= 10) return 2;
  if (pctPos <= 33) return 3;
  if (pctPos <= 66) return 4;
  return 5;
}

/**
 * H-score clinical interpretation.
 */
export function hScoreInterpretation(h: number): string {
  if (h <= 100) return "Low";
  if (h <= 200) return "Intermediate";
  return "High";
}

/**
 * Allred score clinical interpretation.
 */
export function allredInterpretation(score: number): string {
  if (score <= 2) return "Negative";
  if (score <= 4) return "Weakly Positive";
  if (score <= 6) return "Moderately Positive";
  return "Strongly Positive";
}
