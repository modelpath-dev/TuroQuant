import JSZip from "jszip";

/**
 * Build a ZIP file from analysis results.
 */
export async function buildResultsZip(
  images: Record<string, string>,
  scoring: Record<string, unknown>,
  overlay?: string,
): Promise<Blob> {
  const zip = new JSZip();

  for (const [name, b64] of Object.entries(images)) {
    const raw = b64.replace(/^data:image\/\w+;base64,/, "");
    zip.file(`${name}.png`, raw, { base64: true });
  }

  if (overlay) {
    const raw = overlay.replace(/^data:image\/\w+;base64,/, "");
    zip.file("overlay.png", raw, { base64: true });
  }

  // Exclude cells array to keep JSON readable
  const clean = Object.fromEntries(
    Object.entries(scoring).filter(([k]) => k !== "cells"),
  );
  zip.file("scoring.json", JSON.stringify(clean, null, 2));

  return zip.generateAsync({ type: "blob" });
}

/**
 * Build a ZIP with multiple frames (for video/TIF results).
 */
export async function buildFramesZip(
  allResults: Record<string, { images: Record<string, string>; scoring: Record<string, unknown> }>,
  dedupScoring?: Record<string, unknown>,
): Promise<Blob> {
  const zip = new JSZip();

  for (const [frameName, res] of Object.entries(allResults)) {
    const folder = zip.folder(frameName)!;
    for (const [ch, b64] of Object.entries(res.images)) {
      const raw = b64.replace(/^data:image\/\w+;base64,/, "");
      folder.file(`${ch}.png`, raw, { base64: true });
    }
    const clean = Object.fromEntries(
      Object.entries(res.scoring).filter(([k]) => k !== "cells"),
    );
    folder.file("scoring.json", JSON.stringify(clean, null, 2));
  }

  if (dedupScoring) {
    zip.file("scoring_summary.json", JSON.stringify(dedupScoring, null, 2));
  }

  return zip.generateAsync({ type: "blob" });
}
