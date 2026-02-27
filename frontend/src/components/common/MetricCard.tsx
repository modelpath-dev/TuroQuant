"use client";

import { Card, CardContent } from "@/components/ui/card";

interface MetricCardProps {
  label: string;
  value: string | number;
  sublabel?: string;
  className?: string;
}

export function MetricCard({ label, value, sublabel, className = "" }: MetricCardProps) {
  return (
    <Card className={`text-center ${className}`}>
      <CardContent className="pt-4 pb-3 px-3">
        <p className="text-2xl font-bold tracking-tight">{value}</p>
        <p className="text-xs text-muted-foreground mt-1">{label}</p>
        {sublabel && (
          <p className="text-[10px] text-muted-foreground/70 mt-0.5">{sublabel}</p>
        )}
      </CardContent>
    </Card>
  );
}
