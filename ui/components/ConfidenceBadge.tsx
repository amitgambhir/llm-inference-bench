"use client";

interface ConfidenceBadgeProps {
  level: string;
}

const CONF_CONFIG: Record<string, { label: string; className: string; band: string }> = {
  high: {
    label: "HIGH",
    className: "bg-green-100 text-green-800 border border-green-300",
    band: "±10%",
  },
  medium: {
    label: "MEDIUM",
    className: "bg-yellow-100 text-yellow-800 border border-yellow-300",
    band: "±25%",
  },
  low: {
    label: "LOW",
    className: "bg-red-100 text-red-800 border border-red-300",
    band: "±50%",
  },
};

export function ConfidenceBadge({ level }: ConfidenceBadgeProps) {
  const cfg = CONF_CONFIG[level] ?? CONF_CONFIG.low;
  return (
    <span
      className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${cfg.className}`}
      title={`Confidence: ${cfg.label}. Range: ${cfg.band}`}
    >
      {cfg.label}
      <span className="opacity-60 text-xs">{cfg.band}</span>
    </span>
  );
}
