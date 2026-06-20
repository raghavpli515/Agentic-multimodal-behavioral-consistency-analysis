"use client";

import { motion } from "framer-motion";

const messages = [

  "Analyzing multimodal behavioral signals...",

  "Extracting temporal trust patterns...",

  "Running graph-based reasoning...",

  "Evaluating behavioral consistency...",

  "Performing cross-modal conflict analysis...",

  "Generating agentic narrative intelligence..."

];

export default function AILoader() {

  return (

    <div className="
      fixed
      inset-0
      bg-black/90
      backdrop-blur-xl
      z-50
      flex
      flex-col
      items-center
      justify-center
      overflow-hidden
    ">

      {/* BACKGROUND GLOW */}

      <div className="
        absolute
        w-[400px]
        h-[400px]
        bg-cyan-500/20
        rounded-full
        blur-3xl
      " />

      {/* MAIN LOADER */}

      <motion.div

        animate={{
          rotate: 360
        }}

        transition={{
          repeat: Infinity,
          duration: 2,
          ease: "linear"
        }}

        className="
          w-28
          h-28
          rounded-full
          border-4
          border-cyan-400
          border-t-transparent
          mb-12
          relative
          z-10
        "
      />

      {/* TITLE */}

      <motion.h2

        initial={{ opacity: 0.5 }}

        animate={{
          opacity: [0.5, 1, 0.5]
        }}

        transition={{
          repeat: Infinity,
          duration: 2
        }}

        className="
          text-4xl
          font-bold
          text-cyan-400
          mb-8
          relative
          z-10
        "
      >
        AI Behavioral Engine
      </motion.h2>

      {/* SCANNING MESSAGES */}

      <div className="space-y-4 relative z-10">

        {messages.map((msg, idx) => (

          <motion.p

            key={idx}

            initial={{
              opacity: 0,
              x: -20
            }}

            animate={{
              opacity: [0.3, 1, 0.3],
              x: 0
            }}

            transition={{
              repeat: Infinity,
              duration: 2,
              delay: idx * 0.4
            }}

            className="
              text-gray-300
              text-lg
              tracking-wide
            "
          >
            {msg}
          </motion.p>

        ))}

      </div>

    </div>
  );
}