"use client"

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
  SidebarRail,
} from "@/components/ui/sidebar"
import { ClipboardList, Home, CheckCircle2, LogOut } from "lucide-react"
import Link from "next/link"

const adminItems = [
  {
    title: "Dashboard",
    url: "/admin",
    icon: Home,
  },
  {
    title: "Claims",
    url: "/admin/claims",
    icon: ClipboardList,
  },
  {
    title: "Verified Claims",
    url: "/admin/verified",
    icon: CheckCircle2,
  },
]

export function AppSidebar() {
  return (
    <Sidebar collapsible="offcanvas" side="left" variant="sidebar">
      <SidebarHeader>
        <div className="text-lg font-semibold px-4 pt-3">Admin Panel</div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Management</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {adminItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <Link href={item.url} className="flex items-center gap-3">
                      <item.icon size={18} />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild>
              <button className="flex items-center gap-2 text-red-500 hover:text-red-600">
                <LogOut size={18} /> Logout
              </button>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>

      {/* Adds slim clickable rail on desktop */}
      <SidebarRail />
    </Sidebar>
  )
}
