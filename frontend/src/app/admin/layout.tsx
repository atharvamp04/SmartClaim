"use client"

import { SidebarProvider } from "@/components/ui/sidebar"

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider
      style={{
        "--sidebar-width": "16rem",
      } as React.CSSProperties}
    >
      <div className="flex min-h-screen w-full bg-gray-50">
        {children}
      </div>
    </SidebarProvider>
  )
}