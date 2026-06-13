"use client";

import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ReplicaRangeChartProps {
  replicas: number;
  replicasLow: number;
  replicasHigh: number;
  confidence: string;
}

const CONF_COLOR: Record<string, string> = {
  high: "#16a34a",
  medium: "#d97706",
  low: "#dc2626",
};

export function ReplicaRangeChart({
  replicas,
  replicasLow,
  replicasHigh,
  confidence,
}: ReplicaRangeChartProps) {
  const color = CONF_COLOR[confidence] ?? "#6b7280";

  const data = [
    { name: "Low", value: replicasLow, fill: "#d1fae5" },
    { name: "Recommended", value: replicas, fill: color },
    { name: "High", value: replicasHigh, fill: "#fee2e2" },
  ];

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
          <Tooltip
            formatter={(v: number) => [`${v} replicas`, ""]}
            labelStyle={{ fontWeight: 600 }}
          />
          <ReferenceLine y={replicas} stroke={color} strokeDasharray="4 2" />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.fill} stroke={color} strokeWidth={1} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-center text-xs text-gray-500 mt-1">
        Replica range: {replicasLow} – <strong>{replicas}</strong> – {replicasHigh}
      </p>
    </div>
  );
}
