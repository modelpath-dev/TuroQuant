"use client";

import { useState, useCallback } from "react";
import { checkServerHealth } from "@/lib/api/serverHealth";

export function useServerHealth() {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);
  const [checking, setChecking] = useState(false);

  const check = useCallback(async () => {
    setChecking(true);
    const ok = await checkServerHealth();
    setIsHealthy(ok);
    setChecking(false);
    return ok;
  }, []);

  return { isHealthy, checking, check };
}
