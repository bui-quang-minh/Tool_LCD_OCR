import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { ThemeProvider } from "@/components/ThemeProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "LCD OCR Inspector",
  description: "Torque wrench LCD reading detection and validation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`} suppressHydrationWarning>
      <body className="bg-white dark:bg-zinc-950 text-zinc-900 dark:text-zinc-100 antialiased font-sans">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
