/**
 * Simple 2D phase correlation for estimating shift between two grayscale images.
 * Simplified port of cv2.phaseCorrelate — uses DFT via the browser.
 *
 * Returns [dx, dy] pixel shift.
 */
export function phaseCorrelate(
  prev: Float32Array,
  curr: Float32Array,
  width: number,
  height: number,
): [number, number] {
  try {
    // Use cross-correlation via spatial domain (simpler, avoids full FFT impl)
    // Scan a small search window around center for the peak correlation
    const maxShift = 50; // max pixels of shift to detect

    let bestDx = 0;
    let bestDy = 0;
    let bestCorr = -Infinity;

    for (let dy = -maxShift; dy <= maxShift; dy += 2) {
      for (let dx = -maxShift; dx <= maxShift; dx += 2) {
        let sum = 0;
        let count = 0;

        // Sample every 4th pixel for speed
        for (let y = Math.max(0, -dy); y < Math.min(height, height - dy); y += 4) {
          for (let x = Math.max(0, -dx); x < Math.min(width, width - dx); x += 4) {
            const i1 = y * width + x;
            const i2 = (y + dy) * width + (x + dx);
            sum += prev[i1] * curr[i2];
            count++;
          }
        }

        const corr = count > 0 ? sum / count : 0;
        if (corr > bestCorr) {
          bestCorr = corr;
          bestDx = dx;
          bestDy = dy;
        }
      }
    }

    // Refine with single-pixel steps around the best coarse result
    const refinedDx = bestDx;
    const refinedDy = bestDy;
    bestCorr = -Infinity;

    for (let dy = refinedDy - 2; dy <= refinedDy + 2; dy++) {
      for (let dx = refinedDx - 2; dx <= refinedDx + 2; dx++) {
        let sum = 0;
        let count = 0;

        for (let y = Math.max(0, -dy); y < Math.min(height, height - dy); y += 2) {
          for (let x = Math.max(0, -dx); x < Math.min(width, width - dx); x += 2) {
            const i1 = y * width + x;
            const i2 = (y + dy) * width + (x + dx);
            sum += prev[i1] * curr[i2];
            count++;
          }
        }

        const corr = count > 0 ? sum / count : 0;
        if (corr > bestCorr) {
          bestCorr = corr;
          bestDx = dx;
          bestDy = dy;
        }
      }
    }

    return [bestDx, bestDy];
  } catch {
    return [0, 0];
  }
}
