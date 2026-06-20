"use client";

interface Props {
  segments: any[];
}

const EMOTIONS: Record<number, string> = {
  0: "ANGRY",
  1: "HAPPY",
  2: "SAD",
  3: "NEUTRAL",
  4: "FEAR"
};

export default function ModalAgreement({
  segments
}: Props) {

  return (

    <div className="bg-black p-6 rounded-2xl border border-zinc-700">

      <h3 className="text-2xl font-semibold mb-6">
        Cross-Modal Agreement Analysis
      </h3>

      <div className="overflow-x-auto">

        <table className="w-full border-collapse">

          <thead>

            <tr className="border-b border-zinc-700 text-left">

              <th className="p-3">Segment</th>
              <th className="p-3">Audio</th>
              <th className="p-3">Text</th>
              <th className="p-3">Video</th>
              <th className="p-3">Fusion</th>
              <th className="p-3">Agreement</th>

            </tr>

          </thead>

          <tbody>

            {segments.map((s: any, idx: number) => {

              const modalPreds = s.modal_predictions || {};

              const preds = [
                modalPreds.audio,
                modalPreds.text,
                modalPreds.video
              ];

              const agreement =
                new Set(preds).size === 1;

              return (

                <tr
                  key={idx}
                  className="border-b border-zinc-800"
                >

                  <td className="p-3">{idx}</td>

                  <td className="p-3">
                    {EMOTIONS[modalPreds.audio]}
                  </td>

                  <td className="p-3">
                    {EMOTIONS[modalPreds.text]}
                  </td>

                  <td className="p-3">
                    {EMOTIONS[modalPreds.video]}
                  </td>

                  <td className="p-3 font-bold">
                    {EMOTIONS[s.prediction]}
                  </td>

                  <td className="p-3">

                    {agreement ? (

                      <span className="text-green-400">
                        CONSISTENT
                      </span>

                    ) : (

                      <span className="text-red-400">
                        CONFLICT
                      </span>

                    )}

                  </td>

                </tr>

              );
            })}

          </tbody>

        </table>

      </div>

    </div>
  );
}