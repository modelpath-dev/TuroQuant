import type { ScoringResult, StainType } from "@/types";
import { computeKi67Score } from "./ki67Scorer";
import { computeErPrScore } from "./erprScorer";

/**
 * Score a segmentation image. Dispatches to KI67 or ER/PR scorer based on stain.
 */
export function scoreImage(
  segImageData: ImageData,
  markerImageData: ImageData | null,
  stain: StainType,
): ScoringResult {
  if ((stain === "ER" || stain === "PR") && markerImageData) {
    return computeErPrScore(segImageData, markerImageData);
  }
  return computeKi67Score(segImageData);
}

export { computeKi67Score } from "./ki67Scorer";
export { computeErPrScore, computeGlobalErPrScore } from "./erprScorer";
export { findCells } from "./cellDetection";
export { labelConnectedComponents } from "./connectedComponents";
export { regionProps } from "./regionProps";
export {
  intensityGrade,
  allredProportionScore,
  hScoreInterpretation,
  allredInterpretation,
} from "./intensityGrading";
