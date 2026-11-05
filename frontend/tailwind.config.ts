import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
      colors: {
        oled: "#050505",
        surface: {
          DEFAULT: "#0a0a0c",
          elevated: "#121216",
          card: "#16161c",
        },
        brand: {
          50: "#eef8ff",
          100: "#d8eeff",
          200: "#b9e2ff",
          300: "#88cfff",
          400: "#50b2ff",
          500: "#268fff",
          600: "#0c6dfa",
          700: "#0654e5",
          800: "#0b43b9",
          900: "#0f3c91",
        },
        accent: {
          cyan: "#22d3ee",
          emerald: "#10b981",
          amber: "#f59e0b",
          rose: "#f43f5e",
          purple: "#a855f7",
        },
      },
      transitionTimingFunction: {
        "fluid": "cubic-bezier(0.32, 0.72, 0, 1)",
        "spring": "cubic-bezier(0.16, 1, 0.3, 1)",
      },
      animation: {
        "pulse-subtle": "pulseSubtle 4s cubic-bezier(0.32, 0.72, 0, 1) infinite",
        "mesh": "meshSpin 20s linear infinite",
      },
      keyframes: {
        pulseSubtle: {
          "0%, 100%": { opacity: "0.4", transform: "scale(1)" },
          "50%": { opacity: "0.8", transform: "scale(1.05)" },
        },
        meshSpin: {
          "0%": { transform: "rotate(0deg) scale(1)" },
          "50%": { transform: "rotate(180deg) scale(1.1)" },
          "100%": { transform: "rotate(360deg) scale(1)" },
        },
      },
      borderRadius: {
        "bezel-outer": "2rem",
        "bezel-inner": "calc(2rem - 0.5rem)",
      },
    },
  },
  plugins: [],
};

export default config;
