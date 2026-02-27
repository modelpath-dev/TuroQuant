/**
 * Interpolate between consecutive frames for smooth video output.
 * Port of interpolate_frames from app.py.
 */
export function interpolateFrames(
  frames: ImageData[],
  steps: number,
): ImageData[] {
  if (steps === 0 || frames.length < 2) return frames;

  const out: ImageData[] = [];

  for (let i = 0; i < frames.length - 1; i++) {
    out.push(frames[i]);
    const a = frames[i];
    const b = frames[i + 1];
    const w = a.width;
    const h = a.height;

    for (let s = 1; s <= steps; s++) {
      const t = s / (steps + 1);
      const interp = new ImageData(w, h);
      for (let p = 0; p < w * h * 4; p++) {
        interp.data[p] = Math.round((1 - t) * a.data[p] + t * b.data[p]);
      }
      out.push(interp);
    }
  }
  out.push(frames[frames.length - 1]);

  return out;
}
