/**
 * Parse the Seg image to find positive (red) and negative (blue) cell masks.
 * Matches Python thresholds: R>150,G<100,B<100 = positive; B>150,R<100,G<100 = negative
 */
export function findCells(segImageData: ImageData): {
  posMask: Uint8Array;
  negMask: Uint8Array;
} {
  const { data, width, height } = segImageData;
  const size = width * height;
  const posMask = new Uint8Array(size);
  const negMask = new Uint8Array(size);

  for (let i = 0; i < size; i++) {
    const offset = i * 4;
    const r = data[offset];
    const g = data[offset + 1];
    const b = data[offset + 2];

    if (r > 150 && g < 100 && b < 100) {
      posMask[i] = 1;
    }
    if (b > 150 && r < 100 && g < 100) {
      negMask[i] = 1;
    }
  }

  return { posMask, negMask };
}
