"use client";

interface Props {
  analysis: any;
}

export default function AgentCards({
  analysis
}: Props) {

  const patterns = analysis.patterns;

  const persistence = analysis.persistence;

  const graph = analysis.graph_analysis;

  return (

    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">

      {/* FLIP RATE */}
      <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">

        <h3 className="text-lg text-gray-400 mb-2">
          Flip Rate
        </h3>

        <p className="text-4xl font-bold text-red-400">
          {patterns.flip_rate.toFixed(2)}
        </p>

      </div>

      {/* INSTABILITY */}
      <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">

        <h3 className="text-lg text-gray-400 mb-2">
          Instability
        </h3>

        <p className="text-4xl font-bold text-orange-400">
          {patterns.instability.toFixed(2)}
        </p>

      </div>

      {/* STABILITY SCORE */}
      <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">

        <h3 className="text-lg text-gray-400 mb-2">
          Stability Score
        </h3>

        <p className="text-4xl font-bold text-green-400">
          {graph.stability_score.toFixed(2)}
        </p>

      </div>

      {/* ESCALATION */}
      <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">

        <h3 className="text-lg text-gray-400 mb-2">
          Escalation Detection
        </h3>

        <p className="text-2xl font-bold">

          {patterns.escalation_detected ? (
            <span className="text-red-400">
              DETECTED
            </span>
          ) : (
            <span className="text-green-400">
              NOT DETECTED
            </span>
          )}

        </p>

      </div>

      {/* RECOVERY */}
      <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">

        <h3 className="text-lg text-gray-400 mb-2">
          Behavioral Recovery
        </h3>

        <p className="text-2xl font-bold">

          {patterns.behavioral_recovery ? (
            <span className="text-green-400">
              RECOVERY OBSERVED
            </span>
          ) : (
            <span className="text-red-400">
              NO RECOVERY
            </span>
          )}

        </p>

      </div>

      {/* MODAL CONFLICT */}
      <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">

        <h3 className="text-lg text-gray-400 mb-2">
          Modal Conflicts
        </h3>

        <p className="text-4xl font-bold text-yellow-400">
          {patterns.modal_conflicts.length}
        </p>

      </div>

      {/* PERSISTENCE */}
      <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">

        <h3 className="text-lg text-gray-400 mb-2">
          Persistent Uncertain Runs
        </h3>

        <p className="text-4xl font-bold text-orange-400">
          {persistence.persistent_uncertain}
        </p>

      </div>

      {/* INCONSISTENT RUNS */}
      <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">

        <h3 className="text-lg text-gray-400 mb-2">
          Persistent Inconsistent Runs
        </h3>

        <p className="text-4xl font-bold text-red-400">
          {persistence.persistent_inconsistent}
        </p>

      </div>

      {/* DOMINANT STATE */}
      <div className="bg-zinc-900 p-6 rounded-2xl border border-zinc-800">

        <h3 className="text-lg text-gray-400 mb-2">
          Dominant Behavioral State
        </h3>

        <p className="text-3xl font-bold text-cyan-400">
          {graph.dominant_state}
        </p>

      </div>

    </div>
  );
}