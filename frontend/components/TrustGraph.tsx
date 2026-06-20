"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid
} from "recharts";

interface Props {
  segments: any[];
}

export default function TrustGraph({ segments }: Props) {

  const data = segments.map((s, i) => ({
    segment: i,
    trust: s.trust,
  }));

  return (

    <div className="bg-black p-6 rounded-2xl border border-zinc-700">

      <h3 className="text-2xl font-semibold mb-6">
        Trust Evolution Timeline
      </h3>

      <div className="w-full h-[350px]">

        <ResponsiveContainer width="100%" height="100%">

          <LineChart data={data}>

            <CartesianGrid strokeDasharray="3 3" />

            <XAxis dataKey="segment" />

            <YAxis domain={[0, 1]} />

            <Tooltip />

            <Line
              type="monotone"
              dataKey="trust"
              stroke="#00ff99"
              strokeWidth={3}
              dot={false}
            />

          </LineChart>

        </ResponsiveContainer>

      </div>

    </div>
  );
}