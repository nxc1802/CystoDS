"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowUpRight, Activity, Cpu } from "lucide-react";

const NAV_LINKS = [
  { name: "Overview", href: "/" },
  { name: "Live Demo", href: "/demo", highlight: true },
  { name: "Methodology", href: "/research" },
  { name: "Benchmark Results", href: "/results" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Close mobile menu upon route change
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  // Lock scroll when mobile modal is open
  useEffect(() => {
    if (mobileOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [mobileOpen]);

  return (
    <>
      <header className="fixed top-0 left-0 right-0 z-50 flex justify-center px-4 pt-4 md:pt-6 pointer-events-none">
        <nav
          className={`pointer-events-auto flex items-center justify-between gap-3 md:gap-8 rounded-full border border-white/10 bg-black/60 backdrop-blur-2xl px-4 py-2.5 md:px-6 md:py-3 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
            scrolled
              ? "shadow-[0_12px_40px_rgba(0,0,0,0.8)] border-white/15 bg-black/80"
              : "shadow-[0_8px_30px_rgba(0,0,0,0.4)]"
          }`}
        >
          {/* Brand Logo & Clinical Pill */}
          <Link
            href="/"
            className="flex items-center gap-2.5 group focus:outline-none"
          >
            <div className="relative w-8 h-8 rounded-full bg-gradient-to-tr from-brand-600 to-accent-cyan p-0.5 flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:scale-105">
              <div className="w-full h-full bg-[#08080a] rounded-full flex items-center justify-center">
                <Activity
                  className="w-4 h-4 text-brand-400 group-hover:text-white transition-colors duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]"
                  strokeWidth={1.25}
                />
              </div>
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-semibold tracking-tight text-white flex items-center gap-1.5">
                CystoDS
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded-full bg-brand-500/10 text-brand-400 border border-brand-500/20">
                  v3.1
                </span>
              </span>
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <div className="hidden md:flex items-center gap-1">
            {NAV_LINKS.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`relative px-3.5 py-1.5 rounded-full text-xs font-medium tracking-wide transition-all duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                    isActive
                      ? "text-white bg-white/10 shadow-[inset_0_1px_1px_rgba(255,255,255,0.15)]"
                      : "text-neutral-400 hover:text-white hover:bg-white/[0.04]"
                  }`}
                >
                  {link.name}
                </Link>
              );
            })}
          </div>

          {/* CTA Button-in-Button */}
          <div className="flex items-center gap-2">
            <Link
              href="/demo"
              className="hidden sm:inline-flex items-center gap-2 pl-4 pr-1.5 py-1.5 rounded-full bg-white text-black text-xs font-medium tracking-tight hover:bg-neutral-100 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98] group"
            >
              <span>Try Live Demo</span>
              <div className="w-6 h-6 rounded-full bg-black/10 flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
                <ArrowUpRight className="w-3.5 h-3.5 text-black" strokeWidth={1.25} />
              </div>
            </Link>

            {/* Mobile Hamburger Morph to 'X' */}
            <button
              type="button"
              onClick={() => setMobileOpen(!mobileOpen)}
              aria-label="Toggle navigation menu"
              className="md:hidden relative w-9 h-9 rounded-full border border-white/10 bg-white/[0.05] flex items-center justify-center text-white focus:outline-none"
            >
              <div className="w-4 h-3.5 flex flex-col justify-between relative">
                <span
                  className={`w-full h-0.5 bg-white rounded-full transition-all duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] origin-center ${
                    mobileOpen ? "translate-y-[6px] rotate-45" : ""
                  }`}
                />
                <span
                  className={`w-full h-0.5 bg-white rounded-full transition-all duration-200 ease-[cubic-bezier(0.32,0.72,0,1)] ${
                    mobileOpen ? "opacity-0 scale-x-0" : "opacity-100"
                  }`}
                />
                <span
                  className={`w-full h-0.5 bg-white rounded-full transition-all duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] origin-center ${
                    mobileOpen ? "-translate-y-[6px] -rotate-45" : ""
                  }`}
                />
              </div>
            </button>
          </div>
        </nav>
      </header>

      {/* Mobile Full-Screen Glass Modal Expansion */}
      <div
        className={`fixed inset-0 z-40 md:hidden transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
          mobileOpen
            ? "opacity-100 pointer-events-auto backdrop-blur-3xl bg-black/85"
            : "opacity-0 pointer-events-none"
        }`}
      >
        <div className="flex flex-col justify-between h-full px-6 pt-28 pb-10">
          <div className="flex flex-col gap-4">
            <span className="text-[10px] uppercase font-mono tracking-[0.2em] text-neutral-400 pl-2">
              Navigation Menu
            </span>
            <div className="flex flex-col gap-2">
              {NAV_LINKS.map((link, idx) => {
                const isActive = pathname === link.href;
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    style={{
                      transitionDelay: mobileOpen ? `${(idx + 1) * 60}ms` : "0ms",
                    }}
                    className={`text-2xl font-medium tracking-tight px-4 py-3 rounded-2xl border transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] flex items-center justify-between ${
                      mobileOpen
                        ? "translate-y-0 opacity-100"
                        : "translate-y-6 opacity-0"
                    } ${
                      isActive
                        ? "text-white border-white/20 bg-white/10"
                        : "text-neutral-300 border-white/5 bg-white/[0.02]"
                    }`}
                  >
                    <span>{link.name}</span>
                    <div className="w-7 h-7 rounded-full bg-white/5 flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
                      <ArrowUpRight
                        className="w-4 h-4 text-neutral-400 group-hover:text-white"
                        strokeWidth={1.25}
                      />
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>

          {/* Mobile Modal Footer Pill */}
          <div
            style={{
              transitionDelay: mobileOpen ? "320ms" : "0ms",
            }}
            className={`flex flex-col gap-3 transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] ${
              mobileOpen ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0"
            }`}
          >
            <div className="p-4 rounded-2xl border border-white/10 bg-white/[0.03] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-brand-400" strokeWidth={1.25} />
                <span className="text-xs text-neutral-300">Model Checkpoint</span>
              </div>
              <span className="text-xs font-mono text-neutral-400">
                6c103665...
              </span>
            </div>

            <Link
              href="/demo"
              onClick={() => setMobileOpen(false)}
              className="w-full py-3.5 pl-6 pr-3 rounded-full bg-white text-black font-semibold text-sm flex items-center justify-between active:scale-[0.98] transition-all duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group"
            >
              <span>Launch Live Inference</span>
              <div className="w-8 h-8 rounded-full bg-black/10 flex items-center justify-center transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:translate-x-0.5 group-hover:-translate-y-0.5">
                <ArrowUpRight className="w-4 h-4 text-black" strokeWidth={1.25} />
              </div>
            </Link>
          </div>
        </div>
      </div>
    </>
  );
}
