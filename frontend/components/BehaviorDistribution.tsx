"use client";

import {

  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer

} from "recharts";

const COLORS = [

  "#22c55e", // STABLE

  "#ef4444", // INCONSISTENT

  "#eab308"  // UNCERTAIN

];

export default function BehaviorDistribution({

  segments

}: any) {

  const counts: Record<string, number> = {};

  segments.forEach((s: any) => {

    const label = s.behavior_label;

    counts[label] = (counts[label] || 0) + 1;

  });

  const data = Object.keys(counts).map((key) => ({

    name: key,

    value: counts[key]

  }));

  return (

    <div
      className="
      bg-white/5
      backdrop-blur-xl
      p-8
      rounded-3xl
      border
      border-white/10
      shadow-2xl
    "
    >

      <h3 className="text-3xl font-semibold mb-8">
        Behavioral State Distribution
      </h3>

      <div className="h-[400px]">

        <ResponsiveContainer width="100%" height="100%">

          <PieChart>

            <Pie

              data={data}

              cx="50%"

              cy="50%"

              outerRadius={140}

              dataKey="value"

              label
            >

              {data.map((entry, index) => (

                <Cell

                  key={`cell-${index}`}

                  fill={COLORS[index % COLORS.length]}
                />

              ))}

            </Pie>

            <Tooltip />

          </PieChart>

        </ResponsiveContainer>

      </div>

    </div>
  );
}