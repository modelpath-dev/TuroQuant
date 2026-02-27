import { NextResponse } from "next/server";

export async function GET() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const res = await fetch("https://deepliif.org", {
      method: "GET",
      signal: controller.signal,
    });

    clearTimeout(timeout);
    return NextResponse.json({ ok: res.status < 500 });
  } catch {
    return NextResponse.json({ ok: false });
  }
}
