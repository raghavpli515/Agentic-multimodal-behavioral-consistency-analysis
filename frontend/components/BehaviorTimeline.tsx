"use client";

interface Props {
  segments: any[];
}

export default function BehaviorTimeline({
  segments
}: Props) {

  const getColor = (label: string) => {

    if (label === "STABLE") {
      return "bg-green-500";
    }

    if (label === "UNCERTAIN") {
      return "bg-yellow-500";
    }

    return "bg-red-500";
  };

  return (

    <div className="bg-black p-6 rounded-2xl border border-zinc-700">

      <h3 className="text-2xl font-semibold mb-6">
        Behavioral State Timeline
      </h3>

      <div className="flex gap-1 overflow-x-auto">

        {segments.map((s: any, idx: number) => (

          <div
            key={idx}
            className={`
              min-w-[28px]
              h-20
              rounded-md
              ${getColor(s.behavior_label)}
            `}
            title={`
              Segment ${idx}
              | ${s.behavior_label}
            `}
          />

        ))}

      </div>

      {/* LEGEND */}
      <div className="flex gap-6 mt-6 text-sm">

        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded" />
          STABLE
        </div>

        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-yellow-500 rounded" />
          UNCERTAIN
        </div>

        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-500 rounded" />
          INCONSISTENT
        </div>

      </div>

    </div>
  );
}