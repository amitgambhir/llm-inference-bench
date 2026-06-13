"use client";

interface ModeBadgeProps {
  mode: string;
}

const MODE_CONFIG: Record<string, { label: string; className: string }> = {
  estimate_only: {
    label: "⚠ Estimate Only",
    className: "bg-yellow-100 text-yellow-800 border border-yellow-300",
  },
  partially_validated: {
    label: "◑ Partially Validated",
    className: "bg-blue-100 text-blue-800 border border-blue-300",
  },
  validated_by_benchmark: {
    label: "✓ Validated by Benchmark",
    className: "bg-green-100 text-green-800 border border-green-300",
  },
};

export function ModeBadge({ mode }: ModeBadgeProps) {
  const cfg = MODE_CONFIG[mode] ?? MODE_CONFIG.estimate_only;
  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${cfg.className}`}>
      {cfg.label}
    </span>
  );
}
