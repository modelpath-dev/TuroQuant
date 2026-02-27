/* eslint-disable @typescript-eslint/no-explicit-any */
import UTIF from "utif";

/**
 * Parse a multi-page TIFF file and return each page as ImageData.
 */
export async function parseTiffPages(
  buffer: ArrayBuffer,
): Promise<ImageData[]> {
  const ifds = UTIF.decode(buffer);
  const pages: ImageData[] = [];

  for (const ifd of ifds) {
    UTIF.decodeImage(buffer, ifd);
    const rgba = UTIF.toRGBA8(ifd);
    const w = (ifd as any).width as number;
    const h = (ifd as any).height as number;
    const imageData = new ImageData(new Uint8ClampedArray(rgba), w, h);
    pages.push(imageData);
  }

  return pages;
}
