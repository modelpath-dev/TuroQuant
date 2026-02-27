"use client";

import { Badge } from "@/components/ui/badge";

interface ServerStatusBadgeProps {
  isHealthy: boolean | null;
  checking: boolean;
}

export function ServerStatusBadge({ isHealthy, checking }: ServerStatusBadgeProps) {
  if (checking) {
    return (
      <Badge variant="outline" className="gap-1.5 text-xs">
        <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
        Checking...
      </Badge>
    );
  }

  if (isHealthy === null) {
    return (
      <Badge variant="outline" className="gap-1.5 text-xs">
        <span className="h-2 w-2 rounded-full bg-gray-400" />
        Unknown
      </Badge>
    );
  }

  return (
    <Badge variant="outline" className={`gap-1.5 text-xs ${isHealthy ? "border-green-300" : "border-red-300"}`}>
      <span className={`h-2 w-2 rounded-full ${isHealthy ? "bg-green-500" : "bg-red-500"}`} />
      {isHealthy ? "Server Online" : "Server Offline"}
    </Badge>
  );
}
