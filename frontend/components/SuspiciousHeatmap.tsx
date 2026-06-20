"use client";

interface Props {
  segments: any[];
}

export default function SuspiciousHeatmap({
  segments
}: Props) {

  const getIntensity = (
    trust: number,
    entropy: number
  ) => {

    // Higher entropy + lower trust
    const score =
      (1 - trust) * 0.7 +
      entropy * 0.3;

    if (score > 1.0) {
      return "bg-red-600";
    }

    if (score > 0.75) {
      return "bg-orange-500";
    }

    if (score > 0.5) {
      return "bg-yellow-500";
    }

    return "bg-green-500";
  };

  return (

    <div className="bg-black p-6 rounded-2xl border border-zinc-700">

      <h3 className="text-2xl font-semibold mb-6">
        Suspicious Segment Heatmap
      </h3>

      <div className="flex gap-1 overflow-x-auto">

        {segments.map((s: any, idx: number) => (

          <div
            key={idx}
            className={`
              min-w-[32px]
              h-24
              rounded-md
              ${getIntensity(
                s.trust,
                s.entropy
              )}
            `}
            title={`
                    Segment ${idx}

                    Trust: ${s.trust.toFixed(2)}

                    Entropy: ${s.entropy.toFixed(2)}

                    Behavior: ${s.behavior_label}
            `}
          />

        ))}

      </div>

      {/* LEGEND */}
      <div className="flex gap-6 mt-6 text-sm flex-wrap">

        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 rounded" />
          Stable
        </div>

        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-yellow-500 rounded" />
          Moderate Risk
        </div>

        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-orange-500 rounded" />
          High Uncertainty
        </div>

        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-red-600 rounded" />
          Suspicious Segment
        </div>

      </div>

    </div>
  );
}