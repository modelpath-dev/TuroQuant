import { HEALTH_API_URL } from "../constants";

/**
 * Check if the TuroQuant server is reachable.
 */
export async function checkServerHealth(): Promise<boolean> {
  try {
    const res = await fetch(HEALTH_API_URL);
    const data = await res.json();
    return data.ok === true;
  } catch {
    return false;
  }
}
