import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 60;

const API_URL =
  process.env.TUROQUANT_API_URL || "https://deepliif.org/api/infer";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const img = formData.get("img") as File | null;

    if (!img) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 });
    }

    const resolution = (formData.get("resolution") as string) || "40x";
    const probThresh = formData.get("prob_thresh") as string;
    const nopost = formData.get("nopost") as string;
    const slim = formData.get("slim") as string;
    const pil = formData.get("pil") as string;

    // Build query params
    const params = new URLSearchParams({ resolution });
    if (probThresh) {
      const scaled = Math.round(parseFloat(probThresh) * 254);
      params.set("prob_thresh", String(scaled));
    }
    if (nopost === "true") params.set("nopost", "true");
    if (slim === "true") params.set("slim", "true");
    if (pil === "true") params.set("pil", "true");

    // Forward to TuroQuant API
    const apiFormData = new FormData();
    apiFormData.append("img", img, img.name || "image.png");

    const response = await fetch(`${API_URL}?${params}`, {
      method: "POST",
      body: apiFormData,
    });

    if (!response.ok) {
      const rawText = await response.text().catch(() => "");
      // Strip HTML from error responses so users see a clean message
      const text = rawText.replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();

      if (response.status === 500 && nopost !== "true") {
        return NextResponse.json(
          {
            error:
              "Server postprocessing failed (HTTP 500). Try with 'Skip postprocessing' enabled.",
            retryWithNopost: true,
          },
          { status: 500 },
        );
      }

      const friendlyMsg =
        response.status === 500
          ? "The server encountered an internal error. The image may be too large or the server is overloaded — please try again with a smaller image or try later."
          : `TuroQuant API error (HTTP ${response.status}): ${text || "Unknown error"}`;

      return NextResponse.json(
        { error: friendlyMsg, retryable: response.status >= 500 },
        { status: response.status },
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
