"use client";

import { Microscope } from "lucide-react";
import { ServerStatusBadge } from "@/components/common/ServerStatusBadge";

interface HeaderProps {
  isHealthy: boolean | null;
  checking: boolean;
}

export function Header({ isHealthy, checking }: HeaderProps) {
  return (
    <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between h-14 px-4 sm:px-6">
        <div className="flex items-center gap-2.5">
          <Microscope className="h-6 w-6 text-primary" />
          <h1 className="text-lg font-semibold tracking-tight">TuroQuant</h1>
          <span className="text-xs text-muted-foreground hidden sm:inline">
            IHC Quantification Pipeline
          </span>
        </div>
        <ServerStatusBadge isHealthy={isHealthy} checking={checking} />
      </div>
    </header>
  );
}
