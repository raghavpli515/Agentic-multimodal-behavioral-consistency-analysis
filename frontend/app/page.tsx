
"use client";

import { motion } from "framer-motion";
import { useState } from "react";

import { analyzeVideo } from "@/lib/api";

import TrustGraph from "@/components/TrustGraph";
import BehaviorTimeline from "@/components/BehaviorTimeline";
import ModalAgreement from "@/components/ModalAgreement";
import SuspiciousHeatmap from "@/components/SuspiciousHeatmap";
import AgentCards from "@/components/AgentCards";
import AILoader from "@/components/AILoader";
import BehaviorDistribution from "@/components/BehaviorDistribution";
import { exportBehaviorReport } from "@/lib/exportReport";

export default function Home() {

  const [file, setFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);

  const [result, setResult] = useState<any>(null);

  // =========================================
  // DYNAMIC TRUST COLORING
  // =========================================

  const getTrustColor = (trust: string) => {

    switch (trust) {

      case "HIGH":
        return "text-green-400";

      case "MEDIUM":
        return "text-yellow-400";

      case "LOW":
        return "text-red-500";

      default:
        return "text-cyan-400";
    }
  };

  const handleAnalyze = async () => {

    if (!file) return;

    try {

      setLoading(true);

      const data = await analyzeVideo(file);

      setResult(data);

    } catch (err) {

      console.error(err);

      alert("Analysis failed");

    } finally {

      setLoading(false);
    }
  };

  return (

    <main
      className="
      min-h-screen
      bg-gradient-to-br
      from-black
      via-zinc-950
      to-black
      text-white
      px-8
      py-12
      relative
      overflow-hidden
    "
    >

      {/* BACKGROUND GLOWS */}

      <div
        className="
        absolute
        top-0
        left-0
        w-[500px]
        h-[500px]
        bg-cyan-500/10
        rounded-full
        blur-3xl
      "
      />

      <div
        className="
        absolute
        bottom-0
        right-0
        w-[500px]
        h-[500px]
        bg-purple-500/10
        rounded-full
        blur-3xl
      "
      />

      {loading && <AILoader />}

      {/* HERO */}

      <motion.section

        initial={{ opacity: 0, y: 40 }}

        animate={{ opacity: 1, y: 0 }}

        transition={{ duration: 1 }}

        className="text-center mb-16 relative z-10"
      >

        <h1
          className="
          text-7xl
          font-extrabold
          mb-6
          bg-gradient-to-r
          from-cyan-400
          to-purple-500
          bg-clip-text
          text-transparent
        "
        >
          Multimodal Trust AI
        </h1>

        <p className="text-gray-400 max-w-3xl mx-auto text-lg leading-8">
          Behavioral consistency analysis using multimodal reasoning,
          trust-aware inference, temporal intelligence,
          graph reasoning, memory-aware agents,
          and narrative behavioral analytics.
        </p>

        {/* AI STATUS BADGE */}

        <div
          className="
          inline-flex
          items-center
          gap-3
          mt-8
          px-6
          py-3
          rounded-full
          bg-cyan-500/10
          border
          border-cyan-500/20
          backdrop-blur-xl
        "
        >

          <div
            className="
            w-3
            h-3
            rounded-full
            bg-cyan-400
            animate-pulse
          "
          />

          <span className="text-cyan-300 font-medium">
            Agentic Behavioral Intelligence Active
          </span>

        </div>

      </motion.section>

      {/* UPLOAD SECTION */}

      <motion.section

        whileHover={{
          scale: 1.01
        }}

        className="
        max-w-3xl
        mx-auto
        bg-white/5
        backdrop-blur-xl
        p-8
        rounded-3xl
        border
        border-white/10
        shadow-2xl
        relative
        z-10
      "
      >

        <h2 className="text-3xl font-semibold mb-6">
          Upload Interview Video
        </h2>

        <input
          type="file"
          accept="video/*"
          onChange={(e) => {

            if (e.target.files?.[0]) {
              setFile(e.target.files[0]);
            }

          }}
          className="mb-6"
        />

        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="
          bg-gradient-to-r
          from-cyan-500
          to-purple-500
          px-6
          py-3
          rounded-xl
          font-semibold
          hover:scale-105
          transition
        "
        >
          {loading ? "Analyzing..." : "Start Analysis"}
        </button>

      </motion.section>

      {/* RESULTS */}

      {result && (

        <section className="mt-16 max-w-6xl mx-auto">

          <div className="bg-white/5 backdrop-blur-xl p-8 rounded-3xl border border-white/10">

            <h2 className="text-4xl font-bold mb-8">
              Behavioral Intelligence Report
            </h2>

            <div className="mb-8 flex justify-end">

              <button

                onClick={() => exportBehaviorReport(result)}

                className="
                  bg-cyan-400
                  text-black
                  px-6
                  py-3
                  rounded-xl
                  font-semibold
                  hover:bg-cyan-300
                  transition
                "
              >

                Download Behavioral Report

              </button>

            </div>

            {/* OVERALL TRUST */}

            <div className="mb-10 bg-zinc-900/70 border border-cyan-500/30 p-8 rounded-3xl">

              <h3 className="text-2xl font-semibold mb-3">
                Overall Trust Level
              </h3>

              <p
                className={`
                  text-5xl
                  font-bold
                  ${getTrustColor(result.overall_trust)}
                `}
              >
                {result.overall_trust}
              </p>

              {/* TRUST STATUS BADGE */}

              <div className="mt-4">

                <span
                  className={`
                    px-4
                    py-2
                    rounded-full
                    text-sm
                    font-semibold
                    border

                    ${
                      result.overall_trust === "HIGH"
                        ? "border-green-500 text-green-400"

                        : result.overall_trust === "MEDIUM"
                        ? "border-yellow-500 text-yellow-400"

                        : "border-red-500 text-red-400"
                    }
                  `}
                >
                  Autonomous Behavioral Assessment
                </span>

              </div>

            </div>

            {/* TRUST GRAPH */}

            <div className="mb-10">

              <TrustGraph segments={result.segments} />

            </div>

            {/* TIMELINE */}

            <div className="mb-10">

              <BehaviorTimeline
                segments={result.segments}
              />

            </div>

            <div className="mb-10">

                <BehaviorDistribution
                    segments={result.segments}
                />

            </div>

            {/* MODAL AGREEMENT */}

            <div className="mb-10">

              <ModalAgreement
                segments={result.segments}
              />

            </div>

            {/* HEATMAP */}

            <div className="mb-10">

              <SuspiciousHeatmap
                segments={result.segments}
              />

            </div>

            {/* AGENTIC CARDS */}

            <div className="mb-10">

              <h2 className="text-4xl font-bold mb-6">
                Agentic Behavioral Intelligence
              </h2>

              <AgentCards
                analysis={result.agent_analysis}
              />

            </div>

            {/* NARRATIVE REPORT */}

            <div className="mb-10">

              <h3 className="text-2xl font-semibold mb-4">
                Agentic Narrative Reasoning
              </h3>

              <div className="bg-black p-6 rounded-2xl border border-cyan-500/30 whitespace-pre-wrap leading-8 text-gray-300">
                {result.agent_analysis.narrative_report}
              </div>

            </div>

            {/* PATTERN CARDS */}

            <div className="mb-10">

              <h3 className="text-2xl font-semibold mb-4">
                Detected Behavioral Patterns
              </h3>

              <div className="grid grid-cols-2 gap-6">

                <div className="bg-black p-6 rounded-2xl border border-zinc-700">
                  <p className="text-gray-400 mb-2">Flip Rate</p>
                  <p className="text-3xl font-bold text-cyan-400">
                    {result.agent_analysis.patterns.flip_rate.toFixed(2)}
                  </p>
                </div>

                <div className="bg-black p-6 rounded-2xl border border-zinc-700">
                  <p className="text-gray-400 mb-2">Instability</p>
                  <p className="text-3xl font-bold text-cyan-400">
                    {result.agent_analysis.patterns.instability.toFixed(2)}
                  </p>
                </div>

                <div className="bg-black p-6 rounded-2xl border border-zinc-700">
                  <p className="text-gray-400 mb-2">Behavior Pattern</p>
                  <p className="text-xl font-semibold text-white">
                    {
                      result.agent_analysis.graph_analysis
                        .behavior_pattern
                    }
                  </p>
                </div>

                <div className="bg-black p-6 rounded-2xl border border-zinc-700">
                  <p className="text-gray-400 mb-2">Dominant State</p>
                  <p className="text-3xl font-bold text-cyan-400">
                    {
                      result.agent_analysis.graph_analysis
                        .dominant_state
                    }
                  </p>
                </div>

              </div>

            </div>

          </div>

        </section>
      )}
    
    {/* FOOTER */}

    <footer
      className="
      mt-24
      border-t
      border-white/10
      pt-10
      pb-6
      relative
      z-10
    "
    >

      <div className="max-w-7xl mx-auto">

        {/* TOP */}

        <div className="
          flex
          flex-col
          md:flex-row
          justify-between
          items-center
          gap-8
        ">

          {/* BRAND */}

          <div>

            <h2
              className="
              text-2xl
              font-bold
              bg-gradient-to-r
              from-cyan-400
              to-purple-500
              bg-clip-text
              text-transparent
            "
            >
              Multimodal Trust AI
            </h2>

            <p className="text-gray-400 mt-2 max-w-md leading-7">

              Trust-aware multimodal behavioral intelligence
              using temporal reasoning, graph analytics,
              agentic AI, and forensic behavioral modeling.

            </p>

          </div>

          {/* SYSTEM STATUS */}

          <div
            className="
            flex
            items-center
            gap-3
            px-5
            py-3
            rounded-2xl
            bg-white/5
            border
            border-cyan-500/20
            backdrop-blur-xl
          "
          >

            <div
              className="
              w-3
              h-3
              rounded-full
              bg-cyan-400
              animate-pulse
            "
            />

            <span className="text-cyan-300 font-medium">
              Behavioral Intelligence Engine Online
            </span>

          </div>

        </div>

        {/* CAPABILITIES */}

        <div
          className="
          mt-10
          flex
          flex-wrap
          gap-4
        "
        >

          {[
            "Multimodal Fusion",
            "Trust Analytics",
            "Temporal Intelligence",
            "Graph Reasoning",
            "Agentic AI",
            "Behavioral Memory",
            "Cross-Modal Reasoning",
            "Explainable AI"
          ].map((item, idx) => (

            <span

              key={idx}

              className="
                px-4
                py-2
                rounded-full
                bg-white/5
                border
                border-white/10
                text-sm
                text-gray-300
              "
            >
              {item}
            </span>

          ))}

        </div>

        {/* COPYRIGHT */}

        <div
          className="
          mt-10
          text-center
          text-gray-500
          text-sm
        "
        >

          AI Behavioral Intelligence Platform •
          Research-Oriented Multimodal Trust Analytics System

        </div>

      </div>

    </footer>


    </main>
  );
}
