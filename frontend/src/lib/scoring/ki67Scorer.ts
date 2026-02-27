import type { Ki67Score } from "@/types";
import { findCells } from "./cellDetection";
import { labelConnectedComponents } from "./connectedComponents";
import { regionProps } from "./regionProps";

/**
 * Compute KI67 scoring from a Seg image.
 * Returns total, positive, negative cell counts and percent positive.
 */
export function computeKi67Score(segImageData: ImageData): Ki67Score {
  const { width, height } = segImageData;
  const { posMask, negMask } = findCells(segImageData);

  const posLabeled = labelConnectedComponents(posMask, width, height);
  const negLabeled = labelConnectedComponents(negMask, width, height);

  const posProps = regionProps(posLabeled);
  const negProps = regionProps(negLabeled);

  const nPos = posProps.length;
  const nNeg = negProps.length;
  const nTotal = nPos + nNeg;
  const pctPos =
    nTotal > 0 ? Math.round((nPos / nTotal) * 1000) / 10 : 0;

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
  };
}
