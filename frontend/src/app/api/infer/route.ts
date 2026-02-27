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

    // Forward to TuroQuant API with server-side retries and timeout
    const imgBytes = await img.arrayBuffer();
    const maxRetries = 3;
    let lastStatus = 0;
    let lastErrorText = "";

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      if (attempt > 0) {
        // If first attempt failed without nopost, enable it and retry
        if (attempt === 1 && lastStatus === 500 && nopost !== "true") {
          params.set("nopost", "true");
        }
        await new Promise((r) => setTimeout(r, 1500 * attempt));
      }

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 50000);

      try {
        const apiFormData = new FormData();
        apiFormData.append(
          "img",
          new Blob([imgBytes], { type: img.type || "image/png" }),
          img.name || "image.png",
        );

        const response = await fetch(`${API_URL}?${params}`, {
          method: "POST",
          body: apiFormData,
          signal: controller.signal,
        });

        clearTimeout(timeout);

        if (response.ok) {
          const data = await response.json();
          return NextResponse.json(data);
        }

        lastStatus = response.status;
        const rawText = await response.text().catch(() => "");
        lastErrorText = rawText.replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();

        // Only retry on 500+ errors
        if (response.status < 500) {
          return NextResponse.json(
            { error: `TuroQuant API error (HTTP ${response.status}): ${lastErrorText || "Unknown error"}` },
            { status: response.status },
          );
        }
      } catch (err) {
        clearTimeout(timeout);
        const msg = err instanceof Error ? err.message : String(err);
        lastErrorText = msg.includes("abort") ? "Request timed out" : msg;
        lastStatus = 500;
      }
    }

    // All retries exhausted
    const friendlyMsg =
      "The server is currently unavailable or overloaded. All retry attempts failed — please try again in a few minutes.";
    return NextResponse.json(
      { error: friendlyMsg },
      { status: lastStatus || 500 },
    );
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
