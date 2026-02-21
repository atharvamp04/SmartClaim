"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
    CheckCircle, XCircle, AlertTriangle, Clock, Search,
    Bell, FileDown, LogOut, Loader2,
    RefreshCw, ChevronDown, ChevronUp, ShieldCheck,
    FileText, TrendingUp, Menu, ArrowRight, Activity
} from "lucide-react";
import {
    Sidebar,
    SidebarContent,
    SidebarFooter,
    SidebarGroup,
    SidebarGroupLabel,
    SidebarGroupContent,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarProvider,
} from "@/components/ui/sidebar";
import { useSidebar } from "@/components/ui/sidebar";

const API_BASE = "http://127.0.0.1:8000/api/detection";

type NotificationType = "success" | "error" | "warning" | "info" | "pending";

interface Notification {
    claim_id: number;
    claim_number: string;
    type: NotificationType;
    title: string;
    message: string;
    status: string;
    updated_at: string;
    action_available: boolean;
    rejection_reason: string | null;
    claim_amount: number;
}

interface Claim {
    id: number;
    claim_number: string;
    status: string;
    claim_amount: number;
    risk_level: string;
    fraud_detected: boolean;
    accident_date: string | null;
    submitted_at: string | null;
    updated_at: string | null;
    reviewed_by: string | null;
    reviewed_at: string | null;
    admin_notes: string | null;
    rejection_reason: string | null;
    has_report: boolean;
}

const formatINR = (amount: number) =>
    new Intl.NumberFormat("en-IN", {
        style: "currency", currency: "INR",
        minimumFractionDigits: 0, maximumFractionDigits: 0,
    }).format(amount);

const typeConfig: Record<NotificationType, {
    accent: string; iconBg: string; icon: React.ElementType;
    iconColor: string; badgeBg: string; badgeText: string;
}> = {
    success: { accent: "border-l-[3px] border-l-emerald-500", iconBg: "bg-emerald-50", icon: CheckCircle, iconColor: "text-emerald-600", badgeBg: "bg-emerald-50", badgeText: "text-emerald-700" },
    error: { accent: "border-l-[3px] border-l-red-400", iconBg: "bg-red-50", icon: XCircle, iconColor: "text-red-500", badgeBg: "bg-red-50", badgeText: "text-red-700" },
    warning: { accent: "border-l-[3px] border-l-amber-400", iconBg: "bg-amber-50", icon: AlertTriangle, iconColor: "text-amber-500", badgeBg: "bg-amber-50", badgeText: "text-amber-700" },
    info: { accent: "border-l-[3px] border-l-blue-400", iconBg: "bg-blue-50", icon: Bell, iconColor: "text-blue-500", badgeBg: "bg-blue-50", badgeText: "text-blue-700" },
    pending: { accent: "border-l-[3px] border-l-gray-300", iconBg: "bg-gray-50", icon: Clock, iconColor: "text-gray-400", badgeBg: "bg-gray-100", badgeText: "text-gray-600" },
};

const statusConfig: Record<string, { dot: string; text: string; bg: string }> = {
    "Verified": { dot: "bg-emerald-500", text: "text-emerald-700", bg: "bg-emerald-50" },
    "Rejected": { dot: "bg-red-500", text: "text-red-700", bg: "bg-red-50" },
    "Fraud": { dot: "bg-orange-500", text: "text-orange-700", bg: "bg-orange-50" },
    "Pending": { dot: "bg-amber-400", text: "text-amber-700", bg: "bg-amber-50" },
    "Under Survey": { dot: "bg-blue-500", text: "text-blue-700", bg: "bg-blue-50" },
    "Survey Completed": { dot: "bg-violet-500", text: "text-violet-700", bg: "bg-violet-50" },
};

const riskConfig: Record<string, { text: string; bg: string }> = {
    "HIGH": { text: "text-red-600", bg: "bg-red-50" },
    "MEDIUM": { text: "text-amber-600", bg: "bg-amber-50" },
    "LOW": { text: "text-emerald-600", bg: "bg-emerald-50" },
    "CRITICAL": { text: "text-red-800", bg: "bg-red-100" },
};

function StatusPill({ status }: { status: string }) {
    const cfg = statusConfig[status] || { dot: "bg-gray-400", text: "text-gray-600", bg: "bg-gray-50" };
    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${cfg.bg} ${cfg.text}`}>
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${cfg.dot}`} />
            {status}
        </span>
    );
}

function CustomerAccountPageContent() {
    const router = useRouter();
    const { open, setOpen } = useSidebar();

    const [username, setUsername] = useState("");
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [notifications, setNotifications] = useState<Notification[]>([]);
    const [claims, setClaims] = useState<Claim[]>([]);
    const [unreadCount, setUnreadCount] = useState(0);
    const [activeView, setActiveView] = useState<"notifications" | "claims">("notifications");
    const [downloadingId, setDownloadingId] = useState<number | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [expandedId, setExpandedId] = useState<number | null>(null);

    useEffect(() => {
        const token = localStorage.getItem("access_token");
        const user = localStorage.getItem("username") || "";
        if (!token) { router.push("/login"); return; }
        setUsername(user);
        fetchData();
    }, []);

    const fetchData = useCallback(async () => {
        const token = localStorage.getItem("access_token");
        if (!token) return;
        try {
            const [notifRes, claimsRes] = await Promise.all([
                fetch(`${API_BASE}/customer/notifications/`, { headers: { Authorization: `Bearer ${token}` } }),
                fetch(`${API_BASE}/customer/claims/`, { headers: { Authorization: `Bearer ${token}` } }),
            ]);
            if (notifRes.ok) { const nd = await notifRes.json(); setNotifications(nd.notifications || []); setUnreadCount(nd.unread_count || 0); }
            if (claimsRes.ok) { const cd = await claimsRes.json(); setClaims(cd.claims || []); }
        } catch (e) { console.error("Failed to fetch:", e); }
        finally { setLoading(false); setRefreshing(false); }
    }, [router]);

    const handleRefresh = () => { setRefreshing(true); fetchData(); };
    const handleLogout = () => { localStorage.clear(); router.push("/login"); };
    const handleNewClaim = () => { router.push(`/claim?username=${encodeURIComponent(username)}`); };

    const handleDownloadPdf = async (claimId: number, claimNumber: string) => {
        try {
            setDownloadingId(claimId);
            const token = localStorage.getItem("access_token");
            const res = await fetch(`${API_BASE}/claims/${claimId}/report-pdf/`, { headers: { Authorization: `Bearer ${token}` } });
            if (!res.ok) throw new Error("Download failed");
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url; a.download = `SmartClaim_Report_${claimNumber}.pdf`; a.click();
            window.URL.revokeObjectURL(url);
        } catch { alert("PDF download failed. Please try again."); }
        finally { setDownloadingId(null); }
    };

    const filteredClaims = claims.filter(c =>
        c.claim_number.toLowerCase().includes(searchQuery.toLowerCase()) ||
        c.status.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const stats = {
        total: claims.length,
        approved: claims.filter(c => c.status === "Verified").length,
        pending: claims.filter(c => ["Pending", "Under Survey", "Survey Completed"].includes(c.status)).length,
        rejected: claims.filter(c => c.status === "Rejected").length,
    };

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-white">
                <div className="text-center">
                    <div className="w-10 h-10 border-2 border-gray-900 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                    <p className="text-gray-400 text-sm">Loading your account...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex w-full min-h-screen bg-gray-50">

            {/* ════════ SIDEBAR ════════ */}
            <Sidebar className="border-r border-gray-100 bg-white">
                <SidebarContent className="flex flex-col h-full">
                    <SidebarGroup className="flex-1">

                        <SidebarGroupLabel className="px-5 py-5 border-b border-gray-100">
                            <div className="flex items-center gap-3">
                                <div className="w-8 h-8 bg-gray-900 rounded-lg flex items-center justify-center flex-shrink-0">
                                    <ShieldCheck className="h-4 w-4 text-white" />
                                </div>
                                <div>
                                    <p className="text-gray-900 font-bold text-sm leading-none">SmartClaim</p>
                                    <p className="text-gray-400 text-xs mt-0.5 font-normal">Customer Portal</p>
                                </div>
                            </div>
                        </SidebarGroupLabel>

                        <SidebarGroupContent className="mt-2 px-3">
                            <SidebarMenu className="space-y-0.5">

                                <SidebarMenuItem>
                                    <SidebarMenuButton asChild>
                                        <button
                                            onClick={() => setActiveView("notifications")}
                                            className={`flex items-center gap-3 px-4 py-2.5 w-full text-left rounded-lg transition-all text-sm ${activeView === "notifications"
                                                ? "bg-gray-900 text-white"
                                                : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                                                }`}
                                        >
                                            <Bell className="h-4 w-4 flex-shrink-0" />
                                            <span className="font-medium">Notifications</span>
                                            {unreadCount > 0 && (
                                                <span className={`ml-auto text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 ${activeView === "notifications" ? "bg-white text-gray-900" : "bg-gray-900 text-white"
                                                    }`}>
                                                    {unreadCount > 9 ? "9+" : unreadCount}
                                                </span>
                                            )}
                                        </button>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>

                                <SidebarMenuItem>
                                    <SidebarMenuButton asChild>
                                        <button
                                            onClick={() => setActiveView("claims")}
                                            className={`flex items-center gap-3 px-4 py-2.5 w-full text-left rounded-lg transition-all text-sm ${activeView === "claims"
                                                ? "bg-gray-900 text-white"
                                                : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                                                }`}
                                        >
                                            <FileText className="h-4 w-4 flex-shrink-0" />
                                            <span className="font-medium">My Claims</span>
                                            {claims.length > 0 && (
                                                <span className={`ml-auto text-xs font-medium px-1.5 py-0.5 rounded ${activeView === "claims" ? "bg-white/20 text-white" : "bg-gray-100 text-gray-500"
                                                    }`}>
                                                    {claims.length}
                                                </span>
                                            )}
                                        </button>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>

                                <div className="my-2 mx-1 border-t border-gray-100" />

                                <SidebarMenuItem>
                                    <SidebarMenuButton asChild>
                                        <button
                                            onClick={handleNewClaim}
                                            className="flex items-center gap-3 px-4 py-2.5 w-full text-left rounded-lg transition-all text-sm text-blue-600 hover:bg-blue-50"
                                        >
                                            <TrendingUp className="h-4 w-4 flex-shrink-0" />
                                            <span className="font-semibold">File New Claim</span>
                                            <ArrowRight className="h-3.5 w-3.5 ml-auto" />
                                        </button>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>

                            </SidebarMenu>
                        </SidebarGroupContent>
                    </SidebarGroup>
                </SidebarContent>

                <SidebarFooter className="p-4 border-t border-gray-100">
                    <div className="flex items-center gap-3 px-1 mb-3">
                        <div className="w-8 h-8 rounded-full bg-gray-900 flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                            {username.charAt(0).toUpperCase()}
                        </div>
                        <div className="min-w-0">
                            <p className="font-semibold text-sm text-gray-900 truncate">{username}</p>
                            <p className="text-xs text-gray-400">Policyholder</p>
                        </div>
                    </div>
                    <button
                        onClick={handleLogout}
                        className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg border border-gray-200 text-sm text-gray-600 hover:bg-gray-50 hover:text-gray-900 transition-colors"
                    >
                        <LogOut className="h-4 w-4" />
                        <span className="font-medium">Sign out</span>
                    </button>
                </SidebarFooter>
            </Sidebar>

            {/* ════════ MAIN ════════ */}
            <div className="flex-1 flex flex-col min-h-screen overflow-hidden">

                {/* Header */}
                <header className="bg-white border-b border-gray-100 px-4 lg:px-8 py-4 flex items-center justify-between sticky top-0 z-10">
                    <div className="flex items-center gap-3">
                        <button className="lg:hidden p-1.5 rounded-lg hover:bg-gray-100 transition-colors" onClick={() => setOpen(!open)}>
                            <Menu className="h-5 w-5 text-gray-600" />
                        </button>
                        <div>
                            <h1 className="text-base font-semibold text-gray-900">
                                {activeView === "notifications" ? "Notifications" : "My Claims"}
                            </h1>
                            <p className="text-xs text-gray-400 hidden sm:block">
                                {activeView === "notifications"
                                    ? unreadCount > 0 ? `${unreadCount} unread update${unreadCount !== 1 ? "s" : ""}` : "All caught up"
                                    : `${claims.length} claim${claims.length !== 1 ? "s" : ""} on file`}
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button onClick={handleRefresh} className="p-2 rounded-lg hover:bg-gray-50 transition-colors" title="Refresh">
                            <RefreshCw className={`h-4 w-4 text-gray-400 ${refreshing ? "animate-spin" : ""}`} />
                        </button>
                        <button
                            onClick={handleNewClaim}
                            className="hidden sm:flex items-center gap-2 px-4 py-2 bg-gray-900 text-white text-sm font-medium rounded-lg hover:bg-gray-800 transition-colors"
                        >
                            + New Claim
                        </button>
                    </div>
                </header>

                <main className="flex-1 overflow-y-auto">
                    <div className="max-w-5xl mx-auto px-4 lg:px-8 py-6 space-y-5">

                        {/* Stats */}
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                            {[
                                { label: "Total Claims", value: stats.total, icon: Activity, color: "text-gray-900", iconColor: "text-gray-400", border: "border-gray-100" },
                                { label: "Approved", value: stats.approved, icon: CheckCircle, color: "text-emerald-700", iconColor: "text-emerald-500", border: "border-emerald-100" },
                                { label: "In Progress", value: stats.pending, icon: Clock, color: "text-amber-700", iconColor: "text-amber-500", border: "border-amber-100" },
                                { label: "Rejected", value: stats.rejected, icon: XCircle, color: "text-red-600", iconColor: "text-red-400", border: "border-red-100" },
                            ].map(({ label, value, icon: Icon, color, iconColor, border }) => (
                                <div key={label} className={`bg-white rounded-xl border ${border} p-5 shadow-sm`}>
                                    <div className="flex items-start justify-between">
                                        <div>
                                            <p className="text-xs text-gray-400 font-medium mb-1">{label}</p>
                                            <p className={`text-3xl font-bold ${color}`}>{value}</p>
                                        </div>
                                        <div className="p-2 rounded-lg bg-gray-50">
                                            <Icon className={`h-4 w-4 ${iconColor}`} />
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* ── NOTIFICATIONS ── */}
                        {activeView === "notifications" && (
                            <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                                <div className="px-6 py-4 border-b border-gray-50 flex items-center justify-between">
                                    <h2 className="font-semibold text-gray-900 text-sm">All Updates</h2>
                                    {unreadCount > 0 && <span className="text-xs text-gray-400">{unreadCount} new</span>}
                                </div>

                                {notifications.length === 0 ? (
                                    <div className="py-20 text-center">
                                        <div className="w-12 h-12 bg-gray-50 rounded-xl flex items-center justify-center mx-auto mb-4">
                                            <Bell className="h-6 w-6 text-gray-300" />
                                        </div>
                                        <p className="text-gray-500 font-medium text-sm">No notifications yet</p>
                                        <p className="text-gray-400 text-xs mt-1">Claim decisions will appear here</p>
                                    </div>
                                ) : (
                                    <div className="divide-y divide-gray-50">
                                        {notifications.map((notif) => {
                                            const cfg = typeConfig[notif.type];
                                            const Icon = cfg.icon;
                                            const isExpanded = expandedId === notif.claim_id;
                                            return (
                                                <div key={notif.claim_id} className={`px-6 py-5 hover:bg-gray-50/50 transition-colors ${cfg.accent}`}>
                                                    <div className="flex items-start gap-4">
                                                        <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${cfg.iconBg}`}>
                                                            <Icon className={`h-4 w-4 ${cfg.iconColor}`} />
                                                        </div>
                                                        <div className="flex-1 min-w-0">
                                                            <div className="flex flex-wrap items-center gap-2 mb-1.5">
                                                                <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${cfg.badgeBg} ${cfg.badgeText}`}>
                                                                    {notif.status}
                                                                </span>
                                                                <span className="text-xs text-gray-400 font-mono">{notif.claim_number}</span>
                                                                <span className="text-xs text-gray-300">·</span>
                                                                <span className="text-xs text-gray-400">
                                                                    {new Date(notif.updated_at).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })}
                                                                </span>
                                                            </div>
                                                            <p className="font-semibold text-gray-900 text-sm">{notif.title}</p>
                                                            <p className="text-gray-500 text-sm mt-0.5 leading-relaxed">{notif.message}</p>
                                                            {notif.type === "success" && (
                                                                <p className="text-xs text-emerald-600 font-semibold mt-1.5">{formatINR(notif.claim_amount)}</p>
                                                            )}
                                                            {notif.rejection_reason && (
                                                                <div className="mt-3">
                                                                    <button
                                                                        onClick={() => setExpandedId(isExpanded ? null : notif.claim_id)}
                                                                        className="flex items-center gap-1 text-red-500 text-xs font-semibold hover:text-red-600"
                                                                    >
                                                                        {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                                                                        {isExpanded ? "Hide" : "Show"} rejection reason
                                                                    </button>
                                                                    {isExpanded && (
                                                                        <div className="mt-2 p-3 bg-red-50 border border-red-100 rounded-lg">
                                                                            <p className="text-red-800 text-sm leading-relaxed">{notif.rejection_reason}</p>
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            )}
                                                        </div>
                                                        {notif.action_available && (
                                                            <button
                                                                onClick={() => handleDownloadPdf(notif.claim_id, notif.claim_number)}
                                                                disabled={downloadingId === notif.claim_id}
                                                                className="flex items-center gap-1.5 px-3 py-1.5 border border-gray-200 rounded-lg text-xs font-medium text-gray-600 hover:bg-gray-50 hover:border-gray-300 transition-all whitespace-nowrap flex-shrink-0"
                                                            >
                                                                {downloadingId === notif.claim_id
                                                                    ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                                                    : <FileDown className="h-3.5 w-3.5" />}
                                                                PDF
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>
                        )}

                        {/* ── CLAIMS ── */}
                        {activeView === "claims" && (
                            <div className="space-y-4">
                                <div className="flex items-center gap-3">
                                    <div className="relative flex-1">
                                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-300" />
                                        <input
                                            type="text"
                                            placeholder="Search by claim number or status..."
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            className="w-full pl-9 pr-4 py-2.5 bg-white border border-gray-200 rounded-lg text-sm text-gray-900 placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
                                        />
                                    </div>
                                    <button
                                        onClick={handleNewClaim}
                                        className="flex items-center gap-2 px-4 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-lg hover:bg-gray-800 transition-colors whitespace-nowrap"
                                    >
                                        + New Claim
                                    </button>
                                </div>

                                {filteredClaims.length === 0 ? (
                                    <div className="bg-white rounded-xl border border-gray-100 shadow-sm py-20 text-center">
                                        <div className="w-12 h-12 bg-gray-50 rounded-xl flex items-center justify-center mx-auto mb-4">
                                            <FileText className="h-6 w-6 text-gray-300" />
                                        </div>
                                        <p className="text-gray-500 font-medium text-sm">No claims found</p>
                                        <p className="text-gray-400 text-xs mt-1">
                                            {searchQuery ? "Try a different search term" : "File your first claim to get started"}
                                        </p>
                                    </div>
                                ) : (
                                    <>
                                        {/* Desktop table */}
                                        <div className="hidden md:block bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                                            <table className="w-full">
                                                <thead>
                                                    <tr className="border-b border-gray-50">
                                                        {["Claim", "Amount", "Risk", "Status", "Submitted", "Actions"].map(h => (
                                                            <th key={h} className="px-6 py-3.5 text-left text-xs font-semibold text-gray-400 uppercase tracking-wide">{h}</th>
                                                        ))}
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-gray-50">
                                                    {filteredClaims.map((claim) => {
                                                        const risk = riskConfig[claim.risk_level] || { text: "text-gray-600", bg: "bg-gray-50" };
                                                        return (
                                                            <tr key={claim.id} className="hover:bg-gray-50/50 transition-colors">
                                                                <td className="px-6 py-4">
                                                                    <p className="text-sm font-semibold text-gray-900 font-mono">{claim.claim_number}</p>
                                                                    {claim.fraud_detected && <p className="text-xs text-orange-500 font-medium mt-0.5">⚠ Flagged</p>}
                                                                </td>
                                                                <td className="px-6 py-4">
                                                                    <p className="text-sm font-semibold text-gray-900">{formatINR(claim.claim_amount)}</p>
                                                                </td>
                                                                <td className="px-6 py-4">
                                                                    <span className={`text-xs font-semibold px-2 py-1 rounded-md ${risk.bg} ${risk.text}`}>
                                                                        {claim.risk_level}
                                                                    </span>
                                                                </td>
                                                                <td className="px-6 py-4"><StatusPill status={claim.status} /></td>
                                                                <td className="px-6 py-4">
                                                                    <p className="text-sm text-gray-500">
                                                                        {claim.submitted_at ? new Date(claim.submitted_at).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "2-digit" }) : "—"}
                                                                    </p>
                                                                </td>
                                                                <td className="px-6 py-4">
                                                                    <div className="flex items-center gap-2">
                                                                        {claim.status === "Rejected" && claim.rejection_reason && (
                                                                            <button
                                                                                onClick={() => setExpandedId(expandedId === claim.id ? null : claim.id)}
                                                                                className="text-xs text-red-500 hover:text-red-600 font-medium flex items-center gap-1"
                                                                            >
                                                                                {expandedId === claim.id ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                                                                                Reason
                                                                            </button>
                                                                        )}
                                                                        {claim.has_report && (
                                                                            <button
                                                                                onClick={() => handleDownloadPdf(claim.id, claim.claim_number)}
                                                                                disabled={downloadingId === claim.id}
                                                                                className="flex items-center gap-1 px-2.5 py-1.5 border border-gray-200 rounded-lg text-xs font-medium text-gray-600 hover:bg-gray-50 transition-all"
                                                                            >
                                                                                {downloadingId === claim.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <FileDown className="h-3 w-3" />}
                                                                                PDF
                                                                            </button>
                                                                        )}
                                                                    </div>
                                                                </td>
                                                            </tr>
                                                        );
                                                    })}
                                                </tbody>
                                            </table>
                                            {filteredClaims.map((claim) =>
                                                claim.rejection_reason && expandedId === claim.id ? (
                                                    <div key={`r-${claim.id}`} className="px-6 py-4 bg-red-50 border-t border-red-100">
                                                        <p className="text-xs font-bold text-red-400 uppercase tracking-wide mb-1">Rejection Reason — {claim.claim_number}</p>
                                                        <p className="text-red-800 text-sm leading-relaxed">{claim.rejection_reason}</p>
                                                    </div>
                                                ) : null
                                            )}
                                        </div>

                                        {/* Mobile cards */}
                                        <div className="md:hidden space-y-3">
                                            {filteredClaims.map((claim) => {
                                                const risk = riskConfig[claim.risk_level] || { text: "text-gray-600", bg: "bg-gray-50" };
                                                return (
                                                    <div key={claim.id} className="bg-white rounded-xl border border-gray-100 shadow-sm p-5">
                                                        <div className="flex items-start justify-between mb-3">
                                                            <div>
                                                                <p className="font-semibold text-gray-900 font-mono text-sm">{claim.claim_number}</p>
                                                                <p className="text-xs text-gray-400 mt-0.5">
                                                                    {claim.submitted_at ? new Date(claim.submitted_at).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" }) : "—"}
                                                                </p>
                                                            </div>
                                                            <StatusPill status={claim.status} />
                                                        </div>
                                                        <div className="flex items-center justify-between">
                                                            <div>
                                                                <p className="text-lg font-bold text-gray-900">{formatINR(claim.claim_amount)}</p>
                                                                <span className={`text-xs font-semibold px-2 py-0.5 rounded-md mt-1 inline-block ${risk.bg} ${risk.text}`}>
                                                                    {claim.risk_level} risk
                                                                </span>
                                                            </div>
                                                            <div className="flex gap-2">
                                                                {claim.status === "Rejected" && claim.rejection_reason && (
                                                                    <button onClick={() => setExpandedId(expandedId === claim.id ? null : claim.id)} className="px-3 py-1.5 border border-red-200 rounded-lg text-xs font-medium text-red-600">
                                                                        Reason
                                                                    </button>
                                                                )}
                                                                {claim.has_report && (
                                                                    <button onClick={() => handleDownloadPdf(claim.id, claim.claim_number)} disabled={downloadingId === claim.id} className="flex items-center gap-1 px-3 py-1.5 border border-gray-200 rounded-lg text-xs font-medium text-gray-600">
                                                                        <FileDown className="h-3.5 w-3.5" /> PDF
                                                                    </button>
                                                                )}
                                                            </div>
                                                        </div>
                                                        {claim.fraud_detected && <p className="text-xs text-orange-500 font-medium mt-2">⚠ Fraud indicators detected</p>}
                                                        {expandedId === claim.id && claim.rejection_reason && (
                                                            <div className="mt-3 p-3 bg-red-50 border border-red-100 rounded-lg">
                                                                <p className="text-red-800 text-xs leading-relaxed">{claim.rejection_reason}</p>
                                                            </div>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </>
                                )}
                            </div>
                        )}

                    </div>
                </main>
            </div>
        </div>
    );
}

export default function CustomerAccountPage() {
    return (
        <SidebarProvider>
            <CustomerAccountPageContent />
        </SidebarProvider>
    );
}