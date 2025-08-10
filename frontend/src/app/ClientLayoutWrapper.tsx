"use client";

import { usePathname } from "next/navigation";
import Header from "@/components/Header";
import Footer from "@/components/Footer";

export default function ClientLayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  const noLayoutPages = ["/login", "/register", "/policyholder"];
  const showLayout = !noLayoutPages.includes(pathname ?? "");

  return (
    <>
      {showLayout && <Header />}
      <main>{children}</main>
      {showLayout && <Footer />}
    </>
  );
}
