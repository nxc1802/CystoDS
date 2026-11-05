import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import "./globals.css";
import Navbar from "@/components/common/Navbar";
import Footer from "@/components/common/Footer";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
  display: "swap",
});

const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CystoDS — Three-Stage Hierarchical AI for Bladder Cancer Endoscopy",
  description:
    "An AI research platform powered by the 3S-HFT hierarchical deep learning model for real-time bladder cystoscopy diagnostic assistance, lesion localization, and clinical urgency stratification.",
  keywords: [
    "CystoDS",
    "Bladder Cancer",
    "Cystoscopy",
    "Hierarchical Fine-Tuning",
    "Deep Learning",
    "Medical AI",
    "Swin Transformer",
  ],
  authors: [{ name: "CystoDS Research Team" }],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: "#050505",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} font-sans antialiased bg-oled text-foreground min-h-[100dvh] flex flex-col selection:bg-brand-500/30 selection:text-white`}
      >
        <div className="fixed inset-0 bg-radial-mesh pointer-events-none z-0 opacity-70" />
        <Navbar />
        <main className="flex-1 relative z-10">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
