"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
    CheckCircle, XCircle, AlertTriangle, Clock, Search,
    Bell, FileDown, LogOut, Loader2,
    RefreshCw, ChevronDown, ChevronUp, ShieldCheck,
    FileText, TrendingUp, Menu, ArrowRight, Activity, ListChecks, User, MessageCircle, X
} from "lucide-react";

import {
    ResizableHandle,
    ResizablePanel,
    ResizablePanelGroup,
} from "@/components/ui/resizable";

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
import CustomerChatAppointment from "@/components/CustomerChatAppointment";

const API_BASE = `/api/detection`;

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
    assigned_surveyor?: string;
    assigned_surveyor_name?: string;
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
    const [error, setError] = useState<string | null>(null);
    const [notifications, setNotifications] = useState<Notification[]>([]);
    const [claims, setClaims] = useState<Claim[]>([]);
    const [unreadCount, setUnreadCount] = useState(0);
    const [activeView, setActiveView] = useState<"notifications" | "claims">("notifications");
    const [downloadingId, setDownloadingId] = useState<number | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [expandedId, setExpandedId] = useState<number | null>(null);
    const [selectedTimelineClaimId, setSelectedTimelineClaimId] = useState<number | null>(null);
    const [selectedCommunicationClaimId, setSelectedCommunicationClaimId] = useState<number | null>(null);
    const [policyholderData, setPolicyholderData] = useState<PolicyholderData | null>(null);

    useEffect(() => {
        checkAuth();
    }, []);

    const checkAuth = async () => {
        console.log("=== CUSTOMER AUTH CHECK START ===");
        
        const token = localStorage.getItem("access_token");
        const user = localStorage.getItem("username");
        const userRole = localStorage.getItem("user_role");
        
        console.log("Customer auth check:", { 
            hasToken: !!token, 
            tokenLength: token?.length,
            user, 
            userRole,
            allLocalStorage: { 
                access_token: token?.substring(0, 20) + "...", 
                username: user, 
                user_role: userRole 
            }
        });
        
        if (!token || !user) {
            console.log("Missing token or user, redirecting to login");
            console.log("Token exists:", !!token);
            console.log("User exists:", !!user);
            console.log("Full localStorage:", {
                access_token: localStorage.getItem("access_token"),
                username: localStorage.getItem("username"),
                user_role: localStorage.getItem("user_role")
            });
            console.log("=== CUSTOMER AUTH CHECK END - REDIRECT ===");
            router.push("/login");
            return;
        }
        
        // Check if user is a customer or if role is not set (default to customer)
        if (userRole && userRole.toLowerCase() !== 'customer') {
            console.log(`User role is '${userRole}', redirecting to appropriate page`);
            // Redirect to appropriate page based on role
            switch (userRole.toLowerCase()) {
                case 'admin':
                    console.log("Redirecting to admin");
                    router.push("/admin");
                    break;
                case 'surveyor':
                    console.log("Redirecting to surveyor");
                    router.push("/surveyor");
                    break;
                default:
                    // If role is unrecognized, allow access to customer as fallback
                    console.log("Unrecognized role '${userRole}', allowing customer access as fallback");
                    break;
            }
            console.log("=== CUSTOMER AUTH CHECK END - ROLE REDIRECT ===");
            return;
        }
        
        console.log("User authenticated as customer, proceeding to fetch data");
        setUsername(user);
        await fetchData();
        console.log("=== CUSTOMER AUTH CHECK END - SUCCESS ===");
    };

    const fetchData = useCallback(async () => {
        try {
            const token = localStorage.getItem("access_token");
            if (!token) {
                console.log("No token found in fetchData");
                return;
            }

            console.log("Fetching customer data...");
            
            const [notifsRes, claimsRes] = await Promise.all([
                fetch(`${API_BASE}/customer/notifications/`, {
                    headers: { Authorization: `Bearer ${token}` },
                }),
                fetch(`${API_BASE}/customer/claims/`, {
                    headers: { Authorization: `Bearer ${token}` },
                }),
            ]);

            console.log("API responses:", { 
                notificationsStatus: notifsRes.status, 
                claimsStatus: claimsRes.status 
            });

            if (notifsRes.ok) {
                const notifsData = await notifsRes.json();
                setNotifications(notifsData.notifications || []);
                setUnreadCount(notifsData.unread_count || 0);
            } else {
                console.error("Notifications API error:", notifsRes.status, await notifsRes.text());
            }

            if (claimsRes.ok) {
                const claimsData = await claimsRes.json();
                setClaims(claimsData.claims || []);
            } else {
                console.error("Claims API error:", claimsRes.status, await claimsRes.text());
                if (claimsRes.status === 401) {
                    console.log("Unauthorized, clearing token and redirecting to login");
                    localStorage.removeItem("access_token");
                    localStorage.removeItem("username");
                    localStorage.removeItem("user_role");
                    router.push("/login");
                    return;
                }
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
            console.error("Error in fetchData:", error);
            if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
                setError("Network error: Unable to connect to the backend server. Please ensure the Django server is running on http://127.0.0.1:8000");
            } else {
                setError("An error occurred while fetching data. Please try again.");
            }
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
        router.push("/claim");
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

    const handleResubmit = async (claimId: number) => {
        try {
            const token = localStorage.getItem("access_token");
            const response = await fetch(`${API_BASE}/claims/${claimId}/resubmit/`, {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json"
                }
            });
            
            if (response.ok) {
                alert("Claim successfully resubmitted for verification.");
                handleRefresh();
            } else {
                const data = await response.json();
                alert(`Error: ${data.error || "Failed to resubmit claim."}`);
            }
        } catch (error) {
            console.error("Error resubmitting claim:", error);
            alert("An error occurred while resubmitting.");
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

    if (error) {
        return (
            <div className="flex w-full min-h-screen bg-background">
                <div className="flex-1 flex items-center justify-center">
                    <div className="space-y-4 w-full max-w-md px-4 text-center">
                        <AlertTriangle className="h-16 w-16 text-red-500 mx-auto" />
                        <h2 className="text-2xl font-bold text-gray-900">Connection Error</h2>
                        <p className="text-gray-600">{error}</p>
                        <div className="space-y-2">
                            <p className="text-sm text-gray-500">To fix this issue:</p>
                            <ul className="text-sm text-gray-600 text-left space-y-1">
                                <li>• Make sure the Django backend server is running</li>
                                <li>• Check that the server is running on http://127.0.0.1:8000</li>
                                <li>• Verify your internet connection</li>
                                <li>• Try refreshing the page</li>
                            </ul>
                        </div>
                        <Button onClick={() => window.location.reload()} className="mt-4">
                            Refresh Page
                        </Button>
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
                            My Claims
                        </h1>
                        <p className="text-sm text-muted-foreground">
                            {`${claims.length} claim${claims.length !== 1 ? "s" : ""} on file`}
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
                                                            <dd className="text-sm font-semibold">{String(displayValue)}</dd>
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
            <ResizablePanelGroup direction="horizontal" className="flex-1 overflow-hidden">
                <ResizablePanel defaultSize={selectedCommunicationClaimId ? 65 : 100} minSize={30} className="h-full overflow-y-auto">
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

                    <div className="space-y-4 pt-2">
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
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                    {filteredClaims.map((claim) => (
                                        <Card key={claim.id} className="flex flex-col shadow-sm hover:shadow-md transition-shadow">
                                            <CardHeader className="pb-3 border-b border-border/50">
                                                <div className="flex items-start justify-between">
                                                    <div>
                                                        <CardTitle className="text-sm font-mono flex items-center gap-2">
                                                            {claim.claim_number}
                                                            {claim.fraud_detected && (
                                                                <Badge variant="destructive" className="h-5 px-1.5 text-[10px]">Flagged</Badge>
                                                            )}
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
                                            <CardContent className="pt-4 flex-1 flex flex-col justify-between space-y-4">
                                                <div className="flex items-center justify-between">
                                                    <div>
                                                        <p className="text-xs text-muted-foreground mb-1">Assessed Amount</p>
                                                        <p className="text-base font-semibold">
                                                            {formatINR(claim.claim_amount)}
                                                        </p>
                                                    </div>
                                                    <div className="text-right">
                                                        <p className="text-xs text-muted-foreground mb-1">Risk Level</p>
                                                        <Badge variant={
                                                            claim.risk_level === "HIGH" ? "destructive" : 
                                                            claim.risk_level === "MEDIUM" ? "secondary" : "outline"
                                                        }>
                                                            {claim.risk_level}
                                                        </Badge>
                                                    </div>
                                                </div>
                                                
                                                {claim.rejection_reason && (
                                                    <Alert variant="destructive" className="py-2.5 px-3">
                                                        <AlertTriangle className="h-3.5 w-3.5" />
                                                        <AlertDescription className="text-xs ml-2">
                                                            {claim.rejection_reason}
                                                        </AlertDescription>
                                                    </Alert>
                                                )}

                                                <div className="grid grid-cols-2 gap-2 mt-auto pt-3">
                                                    <Button
                                                        variant="outline"
                                                        size="sm"
                                                        className="w-full text-xs"
                                                        onClick={() => setSelectedTimelineClaimId(claim.id)}
                                                    >
                                                        <ListChecks className="h-3.5 w-3.5 mr-1" />
                                                        Timeline
                                                    </Button>
                                                    <Button
                                                        variant="outline"
                                                        size="sm"
                                                        className="w-full text-xs"
                                                        onClick={() => setSelectedCommunicationClaimId(claim.id)}
                                                        disabled={!claim.assigned_surveyor}
                                                    >
                                                        <MessageCircle className="h-3.5 w-3.5 mr-1" />
                                                        Chat
                                                    </Button>
                                                    {claim.has_report && (
                                                        <Button
                                                            variant="outline"
                                                            size="sm"
                                                            className="w-full text-xs col-span-1"
                                                            onClick={() => handleDownloadPdf(claim.id, claim.claim_number)}
                                                            disabled={downloadingId === claim.id}
                                                        >
                                                            <FileDown className="h-3.5 w-3.5 mr-1" />
                                                            PDF
                                                        </Button>
                                                    )}
                                                    {claim.status === "Rejected" && (
                                                        <Button
                                                            variant="outline"
                                                            size="sm"
                                                            className="w-full text-xs col-span-1"
                                                            onClick={() => handleResubmit(claim.id)}
                                                        >
                                                            <RefreshCw className="h-3.5 w-3.5 mr-1" />
                                                            Appeal
                                                        </Button>
                                                    )}
                                                </div>
                                            </CardContent>
                                        </Card>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </ResizablePanel>

                {selectedCommunicationClaimId && (
                    <>
                        <ResizableHandle withHandle />
                        <ResizablePanel defaultSize={35} minSize={25} className="h-full flex flex-col bg-muted/10 border-l relative overflow-hidden">
                            {/* Header with Close Icon */}
                            <div className="flex items-center justify-between p-4 border-b bg-background shrink-0">
                                <h3 className="font-semibold text-sm flex items-center gap-2">
                                    <MessageCircle className="h-4 w-4 text-blue-600" />
                                    Chat & Appointments
                                </h3>
                                <Button 
                                    variant="ghost" 
                                    size="icon" 
                                    className="h-8 w-8 rounded-full" 
                                    onClick={() => setSelectedCommunicationClaimId(null)}
                                >
                                    <X className="h-4 w-4" />
                                </Button>
                            </div>
                            <div className="flex-1 overflow-y-auto min-h-0 w-full relative">
                                <CustomerChatAppointment 
                                    claim={{
                                        id: selectedCommunicationClaimId,
                                        claim_number: claims.find(c => c.id === selectedCommunicationClaimId)?.claim_number || '',
                                        status: claims.find(c => c.id === selectedCommunicationClaimId)?.status || '',
                                        assigned_surveyor_name: claims.find(c => c.id === selectedCommunicationClaimId)?.assigned_surveyor_name || ''
                                    }} 
                                />
                            </div>
                        </ResizablePanel>
                    </>
                )}
            </ResizablePanelGroup>

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
    );
}

export default function CustomerAccountPage() {
    return <CustomerAccountPageContent />;
}
