import type { LabelResult } from "./types";

/**
 * 8-connected component labeling using union-find (two-pass algorithm).
 * Equivalent to skimage.measure.label(mask).
 */
export function labelConnectedComponents(
  mask: Uint8Array,
  width: number,
  height: number,
): LabelResult {
  const labels = new Int32Array(width * height);
  const parent: number[] = [0]; // index 0 unused (background)
  let nextLabel = 1;

  function find(x: number): number {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]; // path compression
      x = parent[x];
    }
    return x;
  }

  function union(a: number, b: number): void {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  }

  // First pass: assign provisional labels
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (!mask[idx]) continue;

      // 8-connected neighbors already visited: N, NW, NE, W
      const neighbors: number[] = [];
      if (y > 0) {
        const n = labels[(y - 1) * width + x];
        if (n) neighbors.push(n);
        if (x > 0) {
          const nw = labels[(y - 1) * width + (x - 1)];
          if (nw) neighbors.push(nw);
        }
        if (x < width - 1) {
          const ne = labels[(y - 1) * width + (x + 1)];
          if (ne) neighbors.push(ne);
        }
      }
      if (x > 0) {
        const w = labels[y * width + (x - 1)];
        if (w) neighbors.push(w);
      }

      if (neighbors.length === 0) {
        labels[idx] = nextLabel;
        parent[nextLabel] = nextLabel;
        nextLabel++;
      } else {
        let minLabel = neighbors[0];
        for (let i = 1; i < neighbors.length; i++) {
          if (neighbors[i] < minLabel) minLabel = neighbors[i];
        }
        labels[idx] = minLabel;
        for (const n of neighbors) {
          union(n, minLabel);
        }
      }
    }
  }

  // Second pass: resolve labels to sequential IDs
  const labelMap = new Map<number, number>();
  let finalLabel = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 0) continue;
    const root = find(labels[i]);
    if (!labelMap.has(root)) {
      labelMap.set(root, ++finalLabel);
    }
    labels[i] = labelMap.get(root)!;
  }

  return { labels, numLabels: finalLabel, width, height };
}
