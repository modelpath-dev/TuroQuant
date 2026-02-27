"use client";

import Image from "next/image";
import { ServerStatusBadge } from "@/components/common/ServerStatusBadge";
import { ThemeToggle } from "@/components/common/ThemeToggle";

interface HeaderProps {
  isHealthy: boolean | null;
  checking: boolean;
}

export function Header({ isHealthy, checking }: HeaderProps) {
  return (
    <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto flex items-center justify-between h-16 sm:h-18 px-4 sm:px-6">
        <div className="flex items-center gap-3">
          <Image
            src="/turocrates.svg"
            alt="TuroQuant logo"
            width={120}
            height={67}
            className="dark:invert h-10 w-auto sm:h-12"
            priority
          />
          <div className="flex flex-col">
            <h1 className="text-lg sm:text-xl font-bold tracking-tight leading-tight">TuroQuant</h1>
            <span className="text-[10px] sm:text-xs text-muted-foreground leading-tight hidden sm:block">
              IHC Quantification Pipeline
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <ServerStatusBadge isHealthy={isHealthy} checking={checking} />
        </div>
      </div>
    </header>
  );
}
