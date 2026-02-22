"use client";

export default function ClientLayoutWrapper({ children }: { children: React.ReactNode }) {
  return <main className="min-h-screen">{children}</main>;
}
