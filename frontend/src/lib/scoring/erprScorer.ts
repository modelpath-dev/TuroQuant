import type { ErPrScore, IntensityDistribution } from "@/types";
import { findCells } from "./cellDetection";
import { labelConnectedComponents } from "./connectedComponents";
import { regionProps } from "./regionProps";
import {
  intensityGrade,
  allredProportionScore,
  hScoreInterpretation,
  allredInterpretation,
} from "./intensityGrading";

/**
 * Build a single-channel max-intensity image from RGB marker data.
 * Equivalent to np.max(marker_rgb, axis=2).
 */
function markerToIntensity(markerData: ImageData): Uint8Array {
  const { data, width, height } = markerData;
  const out = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const offset = i * 4;
    out[i] = Math.max(data[offset], data[offset + 1], data[offset + 2]);
  }
  return out;
}

/**
 * Compute ER/PR scoring from Seg + Marker images.
 * Returns KI67 metrics plus H-score, Allred score, and intensity distribution.
 */
export function computeErPrScore(
  segImageData: ImageData,
  markerImageData: ImageData,
): ErPrScore {
  const { width, height } = segImageData;
  const { posMask, negMask } = findCells(segImageData);
  const markerIntensity = markerToIntensity(markerImageData);

  const posLabeled = labelConnectedComponents(posMask, width, height);
  const negLabeled = labelConnectedComponents(negMask, width, height);

  const posProps = regionProps(posLabeled, markerIntensity);
  const negProps = regionProps(negLabeled);

  const nPos = posProps.length;
  const nNeg = negProps.length;
  const nTotal = nPos + nNeg;

  if (nTotal === 0) {
    return {
      numTotal: 0,
      numPos: 0,
      numNeg: 0,
      percentPos: 0,
      cells: [],
      hScore: 0,
      hScoreLabel: "Low",
      allredScore: 0,
      allredLabel: "Negative",
      allredProportion: 0,
      allredIntensity: 0,
      intensityDistribution: { 0: 0, 1: 0, 2: 0, 3: 0 },
    };
  }

  const pctPos = Math.round((nPos / nTotal) * 1000) / 10;

  // Intensity grading
  const dist: IntensityDistribution = { 0: nNeg, 1: 0, 2: 0, 3: 0 };
  for (const prop of posProps) {
    const grade = intensityGrade(prop.maxIntensity!);
    dist[grade]++;
  }

  // H-score = (1*count1 + 2*count2 + 3*count3) / total * 100
  const hScore = Math.round(
    ((1 * dist[1] + 2 * dist[2] + 3 * dist[3]) / nTotal) * 100,
  );

  // Allred
  const ps = allredProportionScore(pctPos);
  let is_ = 0;
  if (nPos > 0) {
    // Most common intensity grade among positive cells
    if (dist[3] >= dist[2] && dist[3] >= dist[1]) is_ = 3;
    else if (dist[2] >= dist[1]) is_ = 2;
    else is_ = 1;
  }
  const allred = ps + is_;

  const cells = [
    ...posProps.map((p) => ({ centroid: p.centroid, positive: true as const })),
    ...negProps.map((p) => ({ centroid: p.centroid, positive: false as const })),
  ];

  return {
    numTotal: nTotal,
    numPos: nPos,
    numNeg: nNeg,
    percentPos: pctPos,
    cells,
    hScore,
    hScoreLabel: hScoreInterpretation(hScore),
    allredScore: allred,
    allredLabel: allredInterpretation(allred),
    allredProportion: ps,
    allredIntensity: is_,
    intensityDistribution: dist,
  };
}

/**
 * Compute global ER/PR scores from aggregated intensity distribution.
 * Used after video deduplication. Port of compute_global_erpr_score.
 */
export function computeGlobalErPrScore(
  totalDist: IntensityDistribution,
  dedupPos: number,
  dedupNeg: number,
): Partial<ErPrScore> {
  const rawPosTotal = totalDist[1] + totalDist[2] + totalDist[3];
  const nTotal = dedupPos + dedupNeg;

  if (nTotal === 0 || rawPosTotal === 0) {
    return {
      hScore: 0,
      hScoreLabel: "Low",
      allredScore: 0,
      allredLabel: "Negative",
      allredProportion: 0,
      allredIntensity: 0,
      intensityDistribution: { 0: dedupNeg, 1: 0, 2: 0, 3: 0 },
    };
  }

  // Proportionally distribute intensity grades across deduplicated cells
  const prop1 = totalDist[1] / rawPosTotal;
  const prop2 = totalDist[2] / rawPosTotal;
  const prop3 = totalDist[3] / rawPosTotal;

  const dedupDist: IntensityDistribution = {
    0: dedupNeg,
    1: Math.round(dedupPos * prop1),
    2: Math.round(dedupPos * prop2),
    3: Math.round(dedupPos * prop3),
  };

  // Fix rounding to ensure sum equals dedupPos
  const sum123 = dedupDist[1] + dedupDist[2] + dedupDist[3];
  if (sum123 !== dedupPos) {
    const maxGrade = [1, 2, 3].reduce((a, b) =>
      dedupDist[b as 1 | 2 | 3] >= dedupDist[a as 1 | 2 | 3] ? b : a,
    ) as 1 | 2 | 3;
    dedupDist[maxGrade] += dedupPos - sum123;
  }

  const hScore = Math.round(
    ((1 * dedupDist[1] + 2 * dedupDist[2] + 3 * dedupDist[3]) / nTotal) * 100,
  );

  const pctPos = Math.round((dedupPos / nTotal) * 1000) / 10;
  const ps = allredProportionScore(pctPos);
  let is_ = 0;
  if (dedupPos > 0) {
    if (dedupDist[3] >= dedupDist[2] && dedupDist[3] >= dedupDist[1]) is_ = 3;
    else if (dedupDist[2] >= dedupDist[1]) is_ = 2;
    else is_ = 1;
  }

  return {
    hScore,
    hScoreLabel: hScoreInterpretation(hScore),
    allredScore: ps + is_,
    allredLabel: allredInterpretation(ps + is_),
    allredProportion: ps,
    allredIntensity: is_,
    intensityDistribution: dedupDist,
  };
}
