"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
    CheckCircle, XCircle, AlertTriangle, Clock, Search,
    Bell, FileDown, LogOut, Loader2,
    RefreshCw, ChevronDown, ChevronUp, ShieldCheck,
    FileText, TrendingUp, Menu, ArrowRight, Activity, ListChecks, User
} from "lucide-react";

// Shadcn Components
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogHeader,
    AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
    Drawer,
    DrawerClose,
    DrawerContent,
    DrawerDescription,
    DrawerFooter,
    DrawerHeader,
    DrawerTitle,
    DrawerTrigger,
} from "@/components/ui/drawer";
import ClaimTimeline from "@/components/ClaimTimeline";

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

interface PolicyholderData {
    email: string;
    username: string;
    sex: string;
    marital_status: string;
    age: number;
    address_area: string;
    policy_type: string;
    base_policy: string;
    number_of_cars: number;
    agent_type: string;
    vehicle_make: string;
    vehicle_category: string;
    vehicle_price_category: string;
    age_of_vehicle: string;
    year_of_vehicle: number | null;
    driver_rating: number | null;
    past_number_of_claims: number;
    vehicle_model?: string;
}

function formatPolicyholderKey(key: string): string {
    return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
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
        style: "currency",
        currency: "INR",
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(amount);

function CustomerAccountPageContent() {
    const router = useRouter();

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
    const [selectedTimelineClaimId, setSelectedTimelineClaimId] = useState<number | null>(null);
    const [policyholderData, setPolicyholderData] = useState<PolicyholderData | null>(null);

    useEffect(() => {
        checkAuth();
    }, []);

    const checkAuth = async () => {
        const token = localStorage.getItem("access_token");
        const user = localStorage.getItem("username");
        if (!token || !user) {
            router.push("/login");
            return;
        }
        setUsername(user);
        await fetchData();
    };

    const fetchData = useCallback(async () => {
        try {
            const token = localStorage.getItem("access_token");
            if (!token) return;

            const [notifsRes, claimsRes] = await Promise.all([
                fetch(`${API_BASE}/customer/notifications/`, {
                    headers: { Authorization: `Bearer ${token}` },
                }),
                fetch(`${API_BASE}/customer/claims/`, {
                    headers: { Authorization: `Bearer ${token}` },
                }),
            ]);

            if (notifsRes.ok) {
                const notifsData = await notifsRes.json();
                setNotifications(notifsData.notifications || []);
                setUnreadCount(notifsData.unread_count || 0);
            }

            if (claimsRes.ok) {
                const claimsData = await claimsRes.json();
                setClaims(claimsData.claims || []);
            }

            // Fetch policyholder data for "View Info" drawer
            const user = localStorage.getItem("username");
            if (user) {
                try {
                    const phRes = await fetch(`${API_BASE}/policyholder/${user}/`, {
                        headers: { Authorization: `Bearer ${token}` },
                    });
                    if (phRes.ok) setPolicyholderData(await phRes.json());
                } catch (_) { /* ignore */ }
            }
        } catch (error) {
            console.error("Error fetching data:", error);
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    }, []);

    const handleRefresh = async () => {
        setRefreshing(true);
        await fetchData();
    };

    const handleNewClaim = () => {
        router.push("/claim-draft");
    };

    const handleLogout = () => {
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("username");
        router.push("/login");
    };

    const handleDownloadPdf = async (claimId: number, claimNumber: string) => {
        try {
            setDownloadingId(claimId);
            const token = localStorage.getItem("access_token");
            const response = await fetch(
                `${API_BASE}/claims/${claimId}/report-pdf/`,
                { headers: { Authorization: `Bearer ${token}` } }
            );
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = `${claimNumber}_report.pdf`;
                link.click();
            }
        } catch (error) {
            console.error("Error downloading PDF:", error);
        } finally {
            setDownloadingId(null);
        }
    };

    const filteredClaims = claims.filter(
        (claim) =>
            claim.claim_number.toLowerCase().includes(searchQuery.toLowerCase()) ||
            claim.status.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const stats = {
        total: claims.length,
        approved: claims.filter((c) => c.status === "Verified").length,
        pending: claims.filter((c) => ["Pending", "Under Survey", "Survey Completed"].includes(c.status)).length,
        rejected: claims.filter((c) => c.status === "Rejected").length,
    };

    if (loading) {
        return (
            <div className="flex w-full min-h-screen bg-background">
                <div className="flex-1 flex items-center justify-center">
                    <div className="space-y-4 w-full max-w-md px-4">
                        <Skeleton className="h-12 w-full" />
                        <Skeleton className="h-12 w-full" />
                        <Skeleton className="h-12 w-full" />
                    </div>
                </div>
            </div>
        );
    }

    function getStatusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
        const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
            "Verified": "default",
            "Rejected": "destructive",
            "Fraud": "destructive",
            "Pending": "secondary",
            "Under Survey": "secondary",
            "Survey Completed": "secondary",
        };
        return variants[status] || "outline";
    }

    return (
        <div className="flex w-full min-h-screen bg-background flex-col">
            {/* Header */}
            <div className="border-b bg-background sticky top-0 z-10">
                <div className="flex items-center justify-between px-4 lg:px-8 py-4">
                    <div>
                        <h1 className="font-semibold text-lg">
                            {activeView === "notifications" ? "Notifications" : "My Claims"}
                        </h1>
                        <p className="text-sm text-muted-foreground">
                            {activeView === "notifications"
                                ? unreadCount > 0
                                    ? `${unreadCount} unread update${unreadCount !== 1 ? "s" : ""}`
                                    : "All caught up"
                                : `${claims.length} claim${claims.length !== 1 ? "s" : ""} on file`}
                        </p>
                    </div>

                    <div className="flex items-center gap-2">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={handleRefresh}
                            disabled={refreshing}
                        >
                            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
                        </Button>
                        <Drawer direction="right">
                            <DrawerTrigger asChild>
                                <Button variant="outline" size="sm" className="flex">
                                    <User className="h-4 w-4 sm:mr-2" />
                                    <span className="hidden sm:inline">View My Info</span>
                                </Button>
                            </DrawerTrigger>
                            <DrawerContent className="inset-y-0 left-auto right-0 top-0 h-full w-[580px] max-w-[95vw] rounded-l-[10px] rounded-r-none mt-0 [&>div:first-child]:hidden">
                                <div className="flex flex-col h-full w-full overflow-hidden min-h-0">
                                    <DrawerHeader className="shrink-0">
                                        <DrawerTitle>Your Policyholder Details</DrawerTitle>
                                        <DrawerDescription>
                                            All information associated with your policy
                                        </DrawerDescription>
                                    </DrawerHeader>
                                    <div className="p-4 pb-0 flex-1 min-h-0 overflow-hidden">
                                        {policyholderData ? (
                                            (() => {
                                                const entries = Object.entries(policyholderData).filter(
                                                    ([key]) => !["id", "created_at", "updated_at"].includes(key)
                                                );
                                                const mid = Math.ceil(entries.length / 2);
                                                const col1 = entries.slice(0, mid);
                                                const col2 = entries.slice(mid);
                                                const renderItem = ([key, value]: [string, unknown]) => {
                                                    const displayValue = value === null || value === "" ? "N/A" : value;
                                                    return (
                                                        <div key={key} className="flex flex-col gap-0.5 py-2 border-b border-gray-200 last:border-0">
                                                            <dt className="text-xs font-medium text-muted-foreground">
                                                                {formatPolicyholderKey(key)}
                                                            </dt>
                                                            <dd className="text-sm font-semibold">{displayValue}</dd>
                                                        </div>
                                                    );
                                                };
                                                return (
                                                    <dl className="grid grid-cols-2 gap-x-6 gap-y-2">
                                                        <div className="space-y-0">{col1.map(renderItem)}</div>
                                                        <div className="space-y-0">{col2.map(renderItem)}</div>
                                                    </dl>
                                                );
                                            })()
                                        ) : (
                                            <p className="text-muted-foreground text-center py-8">
                                                No policyholder data. Complete your profile to see details here.
                                            </p>
                                        )}
                                    </div>
                                    <DrawerFooter>
                                        <DrawerClose asChild>
                                            <Button variant="outline">Close</Button>
                                        </DrawerClose>
                                    </DrawerFooter>
                                </div>
                            </DrawerContent>
                        </Drawer>
                        <Button onClick={handleNewClaim} className="hidden sm:flex">
                            <TrendingUp className="h-4 w-4 mr-2" />
                            File Claim
                        </Button>
                        <Button onClick={handleLogout} variant="ghost" size="icon">
                            <LogOut className="h-4 w-4" />
                        </Button>
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-y-auto">
                <div className="max-w-5xl mx-auto px-4 lg:px-8 py-6 space-y-6">
                    {/* Stats Cards */}
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                        <Card>
                            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                                <CardTitle className="text-sm font-medium">Total Claims</CardTitle>
                                <Activity className="h-4 w-4 text-muted-foreground" />
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold">{stats.total}</div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                                <CardTitle className="text-sm font-medium">Approved</CardTitle>
                                <CheckCircle className="h-4 w-4 text-green-600" />
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-green-600">{stats.approved}</div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                                <CardTitle className="text-sm font-medium">In Progress</CardTitle>
                                <Clock className="h-4 w-4 text-amber-600" />
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-amber-600">{stats.pending}</div>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                                <CardTitle className="text-sm font-medium">Rejected</CardTitle>
                                <XCircle className="h-4 w-4 text-red-600" />
                            </CardHeader>
                            <CardContent>
                                <div className="text-2xl font-bold text-red-600">{stats.rejected}</div>
                            </CardContent>
                        </Card>
                    </div>

                    {/* Tabs */}
                    <Tabs value={activeView} onValueChange={(v) => setActiveView(v as "notifications" | "claims")}>
                        <TabsList className="grid w-full grid-cols-2">
                            <TabsTrigger value="notifications">
                                <Bell className="h-4 w-4 mr-2" />
                                Notifications
                            </TabsTrigger>
                            <TabsTrigger value="claims">
                                <FileText className="h-4 w-4 mr-2" />
                                Claims
                            </TabsTrigger>
                        </TabsList>

                        {/* Notifications Tab */}
                        <TabsContent value="notifications" className="space-y-4">
                            {notifications.length === 0 ? (
                                <Card>
                                    <CardContent className="flex flex-col items-center justify-center py-12">
                                        <Bell className="h-12 w-12 text-muted-foreground mb-4" />
                                        <h3 className="font-semibold mb-1">No notifications</h3>
                                        <p className="text-sm text-muted-foreground">
                                            Claim decisions will appear here
                                        </p>
                                    </CardContent>
                                </Card>
                            ) : (
                                <div className="space-y-4">
                                    {notifications.map((notif) => (
                                        <Card key={notif.claim_id}>
                                            <CardHeader className="pb-3">
                                                <div className="flex items-start justify-between">
                                                    <div className="space-y-1">
                                                        <CardTitle className="text-sm">{notif.title}</CardTitle>
                                                        <p className="text-sm text-muted-foreground">
                                                            {notif.claim_number}
                                                        </p>
                                                    </div>
                                                    <Badge variant={notif.type === "success" ? "default" : notif.type === "error" ? "destructive" : "secondary"}>
                                                        {notif.status}
                                                    </Badge>
                                                </div>
                                            </CardHeader>
                                            <CardContent className="space-y-3">
                                                <p className="text-sm">{notif.message}</p>
                                                <p className="text-xs text-muted-foreground">
                                                    {new Date(notif.updated_at).toLocaleDateString()}
                                                </p>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            )}
                        </TabsContent>

                        {/* Claims Tab */}
                        <TabsContent value="claims" className="space-y-4">
                            <div className="flex flex-col md:flex-row gap-4">
                                <div className="flex-1 relative">
                                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                                    <Input
                                        placeholder="Search by claim number or status..."
                                        value={searchQuery}
                                        onChange={(e) => setSearchQuery(e.target.value)}
                                        className="pl-9"
                                    />
                                </div>
                                <Button onClick={handleNewClaim}>
                                    <TrendingUp className="h-4 w-4 mr-2" />
                                    New Claim
                                </Button>
                            </div>

                            {filteredClaims.length === 0 ? (
                                <Card>
                                    <CardContent className="flex flex-col items-center justify-center py-12">
                                        <FileText className="h-12 w-12 text-muted-foreground mb-4" />
                                        <h3 className="font-semibold mb-1">No claims found</h3>
                                        <p className="text-sm text-muted-foreground">
                                            {searchQuery ? "Try a different search" : "File your first claim to get started"}
                                        </p>
                                    </CardContent>
                                </Card>
                            ) : (
                                <>
                                    {/* Desktop Table */}
                                    <div className="hidden md:block rounded-lg border overflow-hidden">
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>Claim</TableHead>
                                                    <TableHead>Amount</TableHead>
                                                    <TableHead>Risk</TableHead>
                                                    <TableHead>Status</TableHead>
                                                    <TableHead>Submitted</TableHead>
                                                    <TableHead>Actions</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {filteredClaims.map((claim) => (
                                                    <TableRow key={claim.id}>
                                                        <TableCell className="font-mono text-sm">
                                                            {claim.claim_number}
                                                            {claim.fraud_detected && (
                                                                <Badge variant="destructive" className="ml-2">
                                                                    Flagged
                                                                </Badge>
                                                            )}
                                                        </TableCell>
                                                        <TableCell>{formatINR(claim.claim_amount)}</TableCell>
                                                        <TableCell>
                                                            <Badge
                                                                variant={
                                                                    claim.risk_level === "HIGH"
                                                                        ? "destructive"
                                                                        : claim.risk_level === "MEDIUM"
                                                                          ? "secondary"
                                                                          : "outline"
                                                                }
                                                            >
                                                                {claim.risk_level}
                                                            </Badge>
                                                        </TableCell>
                                                        <TableCell>
                                                            <Badge variant={getStatusVariant(claim.status)}>
                                                                {claim.status}
                                                            </Badge>
                                                        </TableCell>
                                                        <TableCell className="text-sm text-muted-foreground">
                                                            {claim.submitted_at
                                                                ? new Date(claim.submitted_at).toLocaleDateString()
                                                                : "—"}
                                                        </TableCell>
                                                        <TableCell>
                                                            <div className="flex gap-2">
                                                                <Button
                                                                    variant="outline"
                                                                    size="sm"
                                                                    onClick={() => setSelectedTimelineClaimId(claim.id)}
                                                                >
                                                                    <ListChecks className="h-3 w-3 mr-1" />
                                                                    Timeline
                                                                </Button>
                                                                {claim.has_report && (
                                                                    <Button
                                                                        variant="outline"
                                                                        size="sm"
                                                                        onClick={() => handleDownloadPdf(claim.id, claim.claim_number)}
                                                                        disabled={downloadingId === claim.id}
                                                                    >
                                                                        <FileDown className="h-3 w-3 mr-1" />
                                                                        PDF
                                                                    </Button>
                                                                )}
                                                            </div>
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </div>

                                    {/* Mobile Cards */}
                                    <div className="md:hidden space-y-4">
                                        {filteredClaims.map((claim) => (
                                            <Card key={claim.id}>
                                                <CardHeader className="pb-3">
                                                    <div className="flex items-start justify-between">
                                                        <div>
                                                            <CardTitle className="text-sm font-mono">
                                                                {claim.claim_number}
                                                            </CardTitle>
                                                            <p className="text-xs text-muted-foreground mt-1">
                                                                {claim.submitted_at
                                                                    ? new Date(claim.submitted_at).toLocaleDateString()
                                                                    : "—"}
                                                            </p>
                                                        </div>
                                                        <Badge variant={getStatusVariant(claim.status)}>
                                                            {claim.status}
                                                        </Badge>
                                                    </div>
                                                </CardHeader>
                                                <CardContent className="space-y-3">
                                                    <div className="flex items-start justify-between">
                                                        <div>
                                                            <p className="text-sm font-semibold">
                                                                {formatINR(claim.claim_amount)}
                                                            </p>
                                                            <Badge className="mt-2" variant="outline">
                                                                {claim.risk_level} risk
                                                            </Badge>
                                                        </div>
                                                        <div className="space-y-2">
                                                            <Button
                                                                variant="outline"
                                                                size="sm"
                                                                className="w-full"
                                                                onClick={() => setSelectedTimelineClaimId(claim.id)}
                                                            >
                                                                <ListChecks className="h-3 w-3 mr-1" />
                                                                Timeline
                                                            </Button>
                                                            {claim.has_report && (
                                                                <Button
                                                                    variant="outline"
                                                                    size="sm"
                                                                    className="w-full"
                                                                    onClick={() => handleDownloadPdf(claim.id, claim.claim_number)}
                                                                    disabled={downloadingId === claim.id}
                                                                >
                                                                    <FileDown className="h-3 w-3 mr-1" />
                                                                    PDF
                                                                </Button>
                                                            )}
                                                        </div>
                                                    </div>
                                                    {claim.rejection_reason && (
                                                        <>
                                                            <Separator />
                                                            <Alert variant="destructive">
                                                                <AlertTriangle className="h-4 w-4" />
                                                                <AlertDescription>
                                                                    {claim.rejection_reason}
                                                                </AlertDescription>
                                                            </Alert>
                                                        </>
                                                    )}
                                                </CardContent>
                                            </Card>
                                        ))}
                                    </div>
                                </>
                            )}
                        </TabsContent>
                    </Tabs>
                </div>

                {/* Timeline Drawer */}
                <Sheet open={selectedTimelineClaimId !== null} onOpenChange={(open) => !open && setSelectedTimelineClaimId(null)}>
                    <SheetContent side="right" className="w-full sm:w-[600px] overflow-y-auto">
                        <SheetHeader className="mb-6">
                            <SheetTitle>Claim Status Timeline</SheetTitle>
                        </SheetHeader>
                        {selectedTimelineClaimId && (
                            <ClaimTimeline
                                claimId={selectedTimelineClaimId}
                                onClose={() => setSelectedTimelineClaimId(null)}
                            />
                        )}
                    </SheetContent>
                </Sheet>
            </div>
        </div>
    );
}

export default function CustomerAccountPage() {
    return <CustomerAccountPageContent />;
}
