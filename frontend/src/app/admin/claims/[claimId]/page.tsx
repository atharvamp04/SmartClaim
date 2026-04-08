"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ArrowLeft, CheckCircle, XCircle, Loader2,
  AlertTriangle, AlertCircle, Calendar, User, DollarSign,
  FileText, Shield, Image as ImageIcon, Database,
  Activity, TrendingUp, Home, Clock, LogOut, Menu,
  UserCheck, MapPin, Camera, ClipboardCheck, Download,
  FileDown, Brain, MessageCircle
} from "lucide-react";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
} from "@/components/ui/sidebar";
import { useSidebar } from "@/components/ui/sidebar";

const API_BASE_URL = "http://localhost:8000/api/detection";
const MEDIA_BASE_URL = "http://localhost:8000";

export default function ClaimDetailPage() {
  const { open, setOpen } = useSidebar();
  const [claim, setClaim] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [claimId, setClaimId] = useState<string | null>(null);
  const [adminName, setAdminName] = useState("Admin");

  // Surveyor assignment
  const [surveyors, setSurveyors] = useState<any[]>([]);
  const [selectedSurveyor, setSelectedSurveyor] = useState("");
  const [assigningsurveyor, setAssigningSurveyor] = useState(false);
  const [assignSuccess, setAssignSuccess] = useState<string | null>(null);

  // Decision modal state
  const [showDecisionModal, setShowDecisionModal] = useState(false);
  const [pendingDecision, setPendingDecision] = useState<"Verified" | "Rejected" | null>(null);
  const [rejectionReason, setRejectionReason] = useState("");
  const [adminNotes, setAdminNotes] = useState("");
  const [sendEmail, setSendEmail] = useState(true);
  const [decisionResult, setDecisionResult] = useState<any>(null);

  // PDF download
  const [downloadingPdf, setDownloadingPdf] = useState(false);

  useEffect(() => {
    const username = localStorage.getItem("admin_name") || "Admin";
    setAdminName(username);
    const pathSegments = window.location.pathname.split("/");
    const id = pathSegments[pathSegments.length - 1];
    setClaimId(id);
  }, []);

  useEffect(() => {
    if (claimId) {
      fetchClaim();
      fetchSurveyors();
    }
  }, [claimId]);

  const fetchClaim = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem("access_token");
      if (!token) { window.location.href = "/login"; return; }

      const res = await fetch(`${API_BASE_URL}/claims/${claimId}/?t=${Date.now()}`, {
        headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" }
      });

      if (!res.ok) {
        if (res.status === 401) { window.location.href = "/login"; return; }
        throw new Error(`Failed to load claim: ${res.status}`);
      }
      const data = await res.json();
      console.log('=== FRESH API CALL DEBUG ===');
      console.log('Raw API data received:', data);
      console.log('fraud_explanation in raw data:', data.fraud_explanation);
      console.log('Type of fraud_explanation in raw data:', typeof data.fraud_explanation);
      console.log('All keys in API response:', Object.keys(data));
      console.log('=== END DEBUG ===');

      setClaim(data);
      setError(null);

      // Debug after state update
      setTimeout(() => {
        console.log('Claim state after update:', claim);
        console.log('fraud_explanation in claim state:', claim?.fraud_explanation);
      }, 100);
    } catch (err: any) {
      setError(err.message || "Failed to load claim");
    } finally {
      setLoading(false);
    }
  };

  const fetchSurveyors = async () => {
    try {
      const token = localStorage.getItem("access_token");
      if (!token) {
        console.error("No token found for fetchSurveyors");
        return;
      }

      const res = await fetch(`${API_BASE_URL}/admin/surveyors/`, {
        headers: { "Authorization": `Bearer ${token}` }
      });

      if (res.ok) {
        const data = await res.json();
        setSurveyors(data.surveyors || []);
      } else {
        console.error(`Failed to fetch surveyors: ${res.status} ${res.statusText}`);
      }
    } catch (err) {
      console.error("Failed to fetch surveyors:", err);
    }
  };

  const handleAssignSurveyor = async () => {
    if (!selectedSurveyor) return alert("Please select a surveyor.");
    if (!window.confirm(`Assign ${selectedSurveyor} to this claim?`)) return;
    try {
      setAssigningSurveyor(true);
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${API_BASE_URL}/claims/${claimId}/assign-surveyor/`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ surveyor_username: selectedSurveyor })
      });
      if (!res.ok) throw new Error("Failed to assign surveyor");
      const data = await res.json();
      setAssignSuccess(data.message);
      await fetchClaim();
      await fetchSurveyors(); // Refresh surveyors list to update active claims count
    } catch (err: any) {
      alert(err.message || "Failed to assign surveyor");
    } finally {
      setAssigningSurveyor(false);
    }
  };

  // Open the decision modal (instead of directly calling API)
  const openDecisionModal = (decision: "Verified" | "Rejected") => {
    setPendingDecision(decision);
    setRejectionReason("");
    setAdminNotes("");
    setSendEmail(true);
    setDecisionResult(null);
    setShowDecisionModal(true);
  };

  const handleConfirmDecision = async () => {
    if (!pendingDecision) return;
    if (pendingDecision === "Rejected" && !rejectionReason.trim()) {
      alert("Please provide a rejection reason.");
      return;
    }

    try {
      setProcessing(true);
      const token = localStorage.getItem("access_token");

      const res = await fetch(`${API_BASE_URL}/claims/${claimId}/decision/`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({
          status: pendingDecision,
          rejection_reason: rejectionReason.trim(),
          admin_notes: adminNotes.trim(),
          send_email: sendEmail,
        })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || "Failed to update claim");
      }

      const data = await res.json();
      setDecisionResult(data);
      setClaim(data.claim);
      await fetchClaim();
    } catch (err: any) {
      alert(err.message || "Failed to process decision");
    } finally {
      setProcessing(false);
    }
  };

  const handleDownloadPdf = async () => {
    try {
      setDownloadingPdf(true);
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${API_BASE_URL}/claims/${claimId}/report-pdf/`, {
        headers: { "Authorization": `Bearer ${token}` }
      });
      if (!res.ok) throw new Error("Failed to download PDF");
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `SmartClaim_Report_${claim?.claim_number || claimId}.pdf`;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(err.message || "PDF download failed");
    } finally {
      setDownloadingPdf(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("admin_name");
    window.location.href = "/login";
  };

  const handleNavigation = (view: string) => {
    window.location.href = view === "dashboard" ? "/admin" : `/admin?view=${view}`;
  };

  const menuItems = [
    { title: "Dashboard", icon: Home, view: "dashboard" },
    { title: "All Claims", icon: TrendingUp, view: "all" },
    { title: "Pending Review", icon: Clock, view: "pending" },
    { title: "Verified", icon: CheckCircle, view: "verified" },
    { title: "Fraud", icon: XCircle, view: "fraud" },
    { title: "High Risk", icon: AlertTriangle, view: "high-risk" },
    { title: "Under Survey", icon: UserCheck, view: "under-survey" },
    { title: "Survey Completed", icon: ClipboardCheck, view: "survey-completed" },
  ];

  const getRiskBadgeColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "HIGH": return "bg-red-600 text-white";
      case "MEDIUM": return "bg-yellow-600 text-white";
      case "LOW": return "bg-green-600 text-white";
      case "CRITICAL": return "bg-red-900 text-white";
      default: return "bg-gray-600 text-white";
    }
  };

  const getStatusBadgeColor = (status: string) => {
    switch (status) {
      case "Verified": return "bg-green-600 text-white";
      case "Fraud": return "bg-red-600 text-white";
      case "Rejected": return "bg-red-800 text-white";
      case "Pending": return "bg-yellow-600 text-white";
      case "Under Survey": return "bg-blue-600 text-white";
      case "Survey Completed": return "bg-purple-600 text-white";
      default: return "bg-gray-600 text-white";
    }
  };

  const formatCurrency = (amount: number | string) =>
    new Intl.NumberFormat("en-IN", {
      style: "currency", currency: "INR",
      minimumFractionDigits: 0, maximumFractionDigits: 0,
    }).format(Number(amount));

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="text-center">
          <Loader2 className="animate-spin h-12 w-12 text-gray-900 mx-auto mb-4" />
          <p className="text-gray-500">Loading claim details...</p>
        </div>
      </div>
    );
  }

  if (error || !claim) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="text-center">
          <AlertTriangle className="h-12 w-12 text-red-600 mx-auto mb-4" />
          <p className="text-red-800 mb-4">{error || "Claim not found"}</p>
          <Button onClick={() => window.location.href = "/admin"} variant="outline">
            <ArrowLeft className="h-4 w-4 mr-2" /> Back to Dashboard
          </Button>
        </div>
      </div>
    );
  }

  return (
    <>
      {/* ============ DECISION MODAL ============ */}
      {showDecisionModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden">
            {/* Modal header */}
            <div className={`px-6 py-5 ${pendingDecision === "Verified" ? "bg-green-600" : "bg-red-700"}`}>
              <div className="flex items-center gap-3">
                {pendingDecision === "Verified"
                  ? <CheckCircle className="h-7 w-7 text-white" />
                  : <XCircle className="h-7 w-7 text-white" />}
                <div>
                  <h2 className="text-xl font-bold text-white">
                    {pendingDecision === "Verified" ? "Approve Claim" : "Reject Claim"}
                  </h2>
                  <p className="text-sm text-white/80">{claim.claim_number}</p>
                </div>
              </div>
            </div>

            {/* Modal body */}
            <div className="px-6 py-5 space-y-4">
              {decisionResult ? (
                /* Success result */
                <div className="space-y-3">
                  <div className={`p-4 rounded-xl border-2 text-center ${pendingDecision === "Verified"
                      ? "bg-green-50 border-green-300"
                      : "bg-red-50 border-red-300"
                    }`}>
                    <p className="text-lg font-bold">
                      {pendingDecision === "Verified" ? "✅ Claim Approved!" : "❌ Claim Rejected!"}
                    </p>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className={`p-3 rounded-lg ${decisionResult.pdf_generated ? "bg-green-50" : "bg-gray-50"} border`}>
                      <p className="font-semibold">{decisionResult.pdf_generated ? "✅" : "⚠️"} PDF Report</p>
                      <p className="text-gray-600 text-xs mt-1">
                        {decisionResult.pdf_generated ? "Generated successfully" : decisionResult.pdf_error || "Failed"}
                      </p>
                    </div>
                    <div className={`p-3 rounded-lg ${decisionResult.email_sent ? "bg-green-50" : "bg-gray-50"} border`}>
                      <p className="font-semibold">{decisionResult.email_sent ? "✅" : "⚠️"} Email</p>
                      <p className="text-gray-600 text-xs mt-1">
                        {decisionResult.email_sent ? "Sent to customer" : (decisionResult.email_error || "Not sent (email not configured)")}
                      </p>
                    </div>
                  </div>
                  <Button
                    onClick={() => { setShowDecisionModal(false); setDecisionResult(null); }}
                    className="w-full"
                  >
                    Close
                  </Button>
                </div>
              ) : (
                /* Decision form */
                <div className="space-y-4">
                  {/* Claim summary */}
                  <div className="bg-gray-50 rounded-lg p-4 border text-sm space-y-1">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Policyholder:</span>
                      <span className="font-semibold">{claim.policyholder?.username}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Amount:</span>
                      <span className="font-semibold">{formatCurrency(claim.final_claim_amount || claim.claim_amount)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Risk Level:</span>
                      <Badge className={`${getRiskBadgeColor(claim.risk_level)} text-xs`}>{claim.risk_level}</Badge>
                    </div>
                  </div>

                  {/* Rejection reason (only for reject) */}
                  {pendingDecision === "Rejected" && (
                    <div>
                      <label className="block text-sm font-semibold text-red-800 mb-1">
                        Reason for Rejection <span className="text-red-500">*</span>
                      </label>
                      <textarea
                        value={rejectionReason}
                        onChange={(e) => setRejectionReason(e.target.value)}
                        placeholder="Explain clearly why the claim is being rejected. This message will be visible to the customer in their account and in the emailed PDF report."
                        rows={4}
                        className="w-full border-2 border-red-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-red-400 resize-none"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        {rejectionReason.length} characters — be clear and professional
                      </p>
                    </div>
                  )}

                  {/* Admin notes */}
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-1">
                      Internal Admin Notes <span className="text-gray-400">(optional)</span>
                    </label>
                    <textarea
                      value={adminNotes}
                      onChange={(e) => setAdminNotes(e.target.value)}
                      placeholder="Internal notes for audit trail (not visible to customer)"
                      rows={2}
                      className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 resize-none"
                    />
                  </div>

                  {/* Send email toggle */}
                  <label className="flex items-center gap-3 cursor-pointer">
                    <div
                      onClick={() => setSendEmail(!sendEmail)}
                      className={`relative w-11 h-6 rounded-full transition-colors ${sendEmail ? "bg-blue-600" : "bg-gray-300"}`}
                    >
                      <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${sendEmail ? "translate-x-5" : ""}`} />
                    </div>
                    <span className="text-sm font-medium text-gray-700">
                      Send email notification + PDF to customer
                    </span>
                  </label>

                  {/* Action buttons */}
                  <div className="flex gap-3 pt-2">
                    <Button
                      variant="outline"
                      onClick={() => setShowDecisionModal(false)}
                      className="flex-1"
                      disabled={processing}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={handleConfirmDecision}
                      disabled={processing || (pendingDecision === "Rejected" && !rejectionReason.trim())}
                      className={`flex-1 text-white ${pendingDecision === "Verified"
                        ? "bg-green-600 hover:bg-green-700"
                        : "bg-red-700 hover:bg-red-800"
                        }`}
                    >
                      {processing ? (
                        <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Processing...</>
                      ) : (
                        <>
                          {pendingDecision === "Verified"
                            ? <><CheckCircle className="h-4 w-4 mr-2" />Approve Claim</>
                            : <><XCircle className="h-4 w-4 mr-2" />Reject Claim</>}
                        </>
                      )}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <Sidebar className="border-r bg-white w-64">
        <SidebarContent className="flex flex-col h-full">
          <SidebarGroup className="flex-1">
            <SidebarGroupLabel className="text-xl font-bold px-6 py-5 text-black border-b">
              ClaimAI Admin
            </SidebarGroupLabel>
            <SidebarGroupContent className="mt-4">
              <SidebarMenu>
                {menuItems.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <button
                        onClick={() => handleNavigation(item.view)}
                        className="flex items-center gap-3 px-6 py-3 w-full text-left transition-colors rounded-lg mx-2 text-gray-700 hover:bg-gray-100"
                      >
                        <item.icon className="h-5 w-5" />
                        <span className="font-medium">{item.title}</span>
                      </button>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>

        <SidebarFooter className="p-4 border-t">
          <div className="mb-3 px-2">
            <p className="text-xs text-gray-500">Logged in as</p>
            <p className="font-semibold text-sm text-gray-900">{adminName}</p>
          </div>
          <Button variant="outline" onClick={handleLogout} className="w-full flex items-center justify-center gap-2">
            <LogOut className="h-4 w-4" /> Logout
          </Button>
        </SidebarFooter>
      </Sidebar>

      <div className="flex-1 flex flex-col min-h-screen overflow-hidden">
        <header className="bg-white border-b px-4 py-3 lg:px-6 flex items-center justify-between sticky top-0 z-10">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setOpen(!open)}>
              <Menu className="h-5 w-5" />
            </Button>
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="sm" onClick={() => window.location.href = "/admin"}>
                <ArrowLeft className="h-4 w-4 mr-2" /> Back
              </Button>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">Claim {claim.claim_number}</h1>
                <p className="text-xs text-gray-500">Submitted {new Date(claim.submitted_at).toLocaleDateString()}</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Badge className={getRiskBadgeColor(claim.risk_level)}>{claim.risk_level}</Badge>
            <Badge className={getStatusBadgeColor(claim.status)}>{claim.status}</Badge>
            {/* PDF Download Button */}
            {(claim.status === "Verified" || claim.status === "Rejected") && (
              <Button
                size="sm"
                variant="outline"
                onClick={handleDownloadPdf}
                disabled={downloadingPdf}
                className="flex items-center gap-2 border-blue-300 text-blue-700 hover:bg-blue-50"
              >
                {downloadingPdf
                  ? <Loader2 className="h-4 w-4 animate-spin" />
                  : <FileDown className="h-4 w-4" />}
                <span className="hidden sm:inline">{downloadingPdf ? "Generating..." : "Download Report"}</span>
              </Button>
            )}
          </div>
        </header>

        <main className="flex-1 overflow-y-auto bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 py-6">

            {/* ============ ASSIGN SURVEYOR (Pending/Fraud) ============ */}
            {(claim.status === "Pending" || claim.status === "Fraud") && (
              <Card className="mb-6 border-blue-200">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-blue-800">
                    <UserCheck className="h-5 w-5" /> Assign Field Surveyor
                  </CardTitle>
                  <CardDescription>Send this claim for physical field verification by a surveyor</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {claim.assigned_surveyor_name ? (
                    <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg flex items-center gap-3">
                      <UserCheck className="h-5 w-5 text-blue-600" />
                      <div>
                        <p className="font-semibold text-blue-900">
                          Currently assigned to: <span className="font-bold">{claim.assigned_surveyor_name}</span>
                        </p>
                        {claim.assigned_at && (
                          <p className="text-sm text-blue-700">Assigned on {new Date(claim.assigned_at).toLocaleDateString()}</p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No surveyor assigned yet.</p>
                  )}

                  {assignSuccess && (
                    <div className="p-3 bg-green-50 border border-green-200 rounded-lg text-green-800 text-sm font-medium">
                      ✓ {assignSuccess}
                    </div>
                  )}

                  <div className="flex flex-col sm:flex-row gap-3">
                    <select
                      value={selectedSurveyor}
                      onChange={(e) => setSelectedSurveyor(e.target.value)}
                      className="flex-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">— Select a surveyor —</option>
                      {surveyors.map((s) => {
                        const activeClaims = s.active_claims || s.activeClaims || 0;
                        return (
                          <option key={s.username} value={s.username}>
                            {s.username}{s.assigned_region ? ` (${s.assigned_region})` : ""}
                            {` · ${activeClaims} active claim${activeClaims !== 1 ? "s" : ""}`}
                          </option>
                        );
                      })}
                    </select>
                    <Button
                      onClick={handleAssignSurveyor}
                      disabled={assigningsurveyor || !selectedSurveyor}
                      className="bg-blue-600 hover:bg-blue-700 text-white flex items-center gap-2 whitespace-nowrap"
                    >
                      <UserCheck className="h-4 w-4" />
                      {assigningsurveyor ? "Assigning..." : claim.assigned_surveyor_name ? "Reassign" : "Assign Surveyor"}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* ============ SURVEY STATUS CARD (Under Survey / Survey Completed) ============ */}
            {(claim.status === "Under Survey" || claim.status === "Survey Completed") && (
              <Card className="mb-6 border-purple-200">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-purple-800">
                    <ClipboardCheck className="h-5 w-5" /> Field Survey Status
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="p-4 bg-gray-50 rounded-lg border">
                      <p className="text-sm text-gray-600">Assigned Surveyor</p>
                      <p className="font-semibold text-lg">{claim.assigned_surveyor_name || "N/A"}</p>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-lg border">
                      <p className="text-sm text-gray-600">Assigned On</p>
                      <p className="font-semibold">
                        {claim.assigned_at ? new Date(claim.assigned_at).toLocaleDateString() : "N/A"}
                      </p>
                    </div>
                  </div>

                  {claim.status === "Survey Completed" && (
                    <>
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                          <p className="text-sm text-gray-600">Recommendation</p>
                          <p className="font-bold text-lg text-purple-800">{claim.surveyor_recommendation || "N/A"}</p>
                        </div>
                        <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                          <p className="text-sm text-gray-600">Damage Verified</p>
                          <p className="font-bold text-lg">
                            {claim.damage_verified === true ? "✓ Yes" : claim.damage_verified === false ? "✗ No" : "N/A"}
                          </p>
                        </div>
                        <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                          <p className="text-sm text-gray-600">Surveyor Assessed Amount</p>
                          <p className="font-bold text-lg">
                            {claim.surveyor_assessed_amount ? formatCurrency(claim.surveyor_assessed_amount) : "Not provided"}
                          </p>
                        </div>
                      </div>

                      {claim.surveyor_notes && (
                        <div className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600 mb-2 font-medium">Field Survey Notes</p>
                          <p className="text-gray-800 leading-relaxed">{claim.surveyor_notes}</p>
                        </div>
                      )}

                      {/* Field Photos */}
                      {claim.field_photos && claim.field_photos.length > 0 && (
                        <div>
                          <p className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
                            <Camera className="h-4 w-4" /> Field Photos ({claim.field_photos.length})
                          </p>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            {claim.field_photos.map((photo: any, idx: number) => (
                              <div key={idx} className="border rounded-lg overflow-hidden">
                                <img src={photo.photo_url} alt={`Field photo ${idx + 1}`} className="w-full h-40 object-cover" />
                                {photo.caption && <p className="text-xs text-gray-600 p-2">{photo.caption}</p>}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* FINAL DECISION BUTTONS (after survey) */}
                      <div className="pt-4 border-t">
                        <p className="text-sm font-semibold text-gray-700 mb-3">
                          Final Decision — Admin Action Required
                        </p>
                        <div className="flex flex-wrap gap-3">
                          <Button
                            onClick={() => openDecisionModal("Verified")}
                            disabled={processing}
                            className="bg-green-600 hover:bg-green-700 text-white flex items-center gap-2"
                          >
                            <CheckCircle className="h-4 w-4" /> Approve Claim
                          </Button>
                          <Button
                            onClick={() => openDecisionModal("Rejected")}
                            disabled={processing}
                            className="bg-red-600 hover:bg-red-700 text-white flex items-center gap-2"
                          >
                            <XCircle className="h-4 w-4" /> Reject Claim
                          </Button>
                        </div>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>
            )}

            {/* ============ APPROVED BANNER ============ */}
            {claim.status === "Verified" && (
              <div className="bg-green-50 border border-green-300 rounded-xl p-5 mb-6">
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-6 w-6 text-green-600 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-bold text-green-900 text-lg">Claim Approved ✅</p>
                    <p className="text-sm text-green-700 mt-1">
                      This claim has been verified and approved.
                      {claim.reviewed_by && ` Approved by ${claim.reviewed_by}`}
                      {claim.reviewed_at && ` on ${new Date(claim.reviewed_at).toLocaleDateString()}`}.
                    </p>
                    {claim.admin_notes && (
                      <p className="text-sm text-green-800 mt-2 italic">"{claim.admin_notes}"</p>
                    )}
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={handleDownloadPdf}
                    disabled={downloadingPdf}
                    className="border-green-400 text-green-700 hover:bg-green-100 flex items-center gap-2 whitespace-nowrap"
                  >
                    <FileDown className="h-4 w-4" />
                    {downloadingPdf ? "..." : "PDF Report"}
                  </Button>
                </div>
              </div>
            )}

            {/* ============ REJECTED BANNER ============ */}
            {claim.status === "Rejected" && (
              <div className="bg-red-50 border border-red-300 rounded-xl p-5 mb-6">
                <div className="flex items-start gap-3">
                  <XCircle className="h-6 w-6 text-red-600 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-bold text-red-900 text-lg">Claim Rejected ❌</p>
                    <p className="text-sm text-red-700 mt-1">
                      {claim.reviewed_by && `Rejected by ${claim.reviewed_by}`}
                      {claim.reviewed_at && ` on ${new Date(claim.reviewed_at).toLocaleDateString()}`}.
                    </p>
                    {claim.rejection_reason && (
                      <div className="mt-3 p-3 bg-red-100 border border-red-200 rounded-lg">
                        <p className="text-xs font-semibold text-red-800 uppercase mb-1">Rejection Reason (visible to customer):</p>
                        <p className="text-sm text-red-900">{claim.rejection_reason}</p>
                      </div>
                    )}
                    {claim.admin_notes && (
                      <p className="text-xs text-red-700 mt-2 italic">Internal note: "{claim.admin_notes}"</p>
                    )}
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={handleDownloadPdf}
                    disabled={downloadingPdf}
                    className="border-red-400 text-red-700 hover:bg-red-100 flex items-center gap-2 whitespace-nowrap"
                  >
                    <FileDown className="h-4 w-4" />
                    {downloadingPdf ? "..." : "PDF Report"}
                  </Button>
                </div>
              </div>
            )}

            {/* ============ FRAUD BANNER ============ */}
            {claim.status === "Fraud" && (
              <div className="bg-orange-50 border border-orange-300 rounded-xl p-5 mb-6">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="h-6 w-6 text-orange-600" />
                  <div>
                    <p className="font-bold text-orange-900">Fraud Detected ⚠️</p>
                    <p className="text-sm text-orange-700">AI system detected potential fraud. Assign surveyor for field verification.</p>
                  </div>
                </div>
              </div>
            )}

            {/* ============ TABS ============ */}
            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="w-full grid grid-cols-3 lg:grid-cols-8 mb-6">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="analysis">Analysis</TabsTrigger>
                <TabsTrigger value="reasoning">AI Reasoning</TabsTrigger>
                <TabsTrigger value="documents">Documents</TabsTrigger>
                <TabsTrigger value="images">Images</TabsTrigger>
                <TabsTrigger value="database">Database</TabsTrigger>
                <TabsTrigger value="history">History</TabsTrigger>
                <TabsTrigger value="surveyor">Surveyor</TabsTrigger>
              </TabsList>

              {/* OVERVIEW TAB */}
              <TabsContent value="overview">
                <Card>
                  <CardHeader>
                    <CardTitle>Claim Overview</CardTitle>
                    <CardDescription>Key information and details about this claim</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 border rounded-lg">
                        <p className="text-sm text-gray-600 flex items-center gap-2 mb-2"><DollarSign className="h-4 w-4" /> Claim Amount</p>
                        <p className="text-2xl font-bold">{formatCurrency(claim.claim_amount)}</p>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <p className="text-sm text-gray-600 flex items-center gap-2 mb-2"><Activity className="h-4 w-4" /> Confidence Score</p>
                        <p className="text-2xl font-bold">{Number(claim.confidence_score).toFixed(1)}%</p>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <p className="text-sm text-gray-600 flex items-center gap-2 mb-2"><ImageIcon className="h-4 w-4" /> Images</p>
                        <p className="text-2xl font-bold">{claim.total_images_submitted || 0}</p>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <p className="text-sm text-gray-600 flex items-center gap-2 mb-2"><AlertTriangle className="h-4 w-4" /> Damage Areas</p>
                        <p className="text-2xl font-bold">{claim.total_damage_areas || 0}</p>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2"><User className="h-5 w-5" /> Policyholder</h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600">Username</p>
                          <p className="font-semibold text-lg">{claim.policyholder?.username}</p>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600">Email</p>
                          <p className="font-semibold text-lg">{claim.policyholder?.email}</p>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2"><FileText className="h-5 w-5" /> Claim Details</h3>
                      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {[
                          ["Claim Number", claim.claim_number],
                          ["Accident Date", claim.accident_date ? new Date(claim.accident_date).toLocaleDateString() : "N/A"],
                          ["Submitted Date", new Date(claim.submitted_at).toLocaleDateString()],
                          ["DL Number", claim.dl_number || "N/A"],
                          ["Vehicle Reg No", claim.vehicle_reg_no || "N/A"],
                          ["FIR Number", claim.fir_number || "N/A"],
                        ].map(([label, value]) => (
                          <div key={label} className="p-4 bg-gray-50 rounded-lg border">
                            <p className="text-sm text-gray-600">{label}</p>
                            <p className="font-semibold">{value}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold mb-3">Description</h3>
                      <div className="p-4 bg-gray-50 rounded-lg border">
                        <p className="text-gray-700 leading-relaxed">{claim.claim_description}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* ANALYSIS TAB */}
              <TabsContent value="analysis">
                <Card>
                  <CardHeader>
                    <CardTitle>Fraud Analysis</CardTitle>
                    <CardDescription>AI-powered fraud detection analysis results</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid md:grid-cols-3 gap-4">
                      <div className={`p-6 rounded-lg border ${claim.fraud_detected ? "bg-red-50 border-red-300" : "bg-green-50 border-green-300"}`}>
                        <p className="text-sm font-medium text-gray-600 mb-2">Final Detection</p>
                        <p className={`text-3xl font-bold ${claim.fraud_detected ? "text-red-700" : "text-green-700"}`}>
                          {claim.fraud_detected ? "🚨 FRAUD" : "✓ LEGIT"}
                        </p>
                        <p className="text-sm mt-2 text-gray-600">Confidence: <strong>{Number(claim.confidence_score).toFixed(1)}%</strong></p>
                      </div>
                      <div className="p-6 border rounded-lg">
                        <p className="text-sm font-medium text-gray-600 mb-2">Tabular Analysis</p>
                        <p className="text-3xl font-bold">{Number(claim.tabular_fraud_probability || 0).toFixed(1)}%</p>
                        <p className="text-sm mt-2 text-gray-600">Fraud Probability</p>
                      </div>
                      <div className="p-6 border rounded-lg">
                        <p className="text-sm font-medium text-gray-600 mb-2">Image Analysis</p>
                        <p className="text-3xl font-bold">{Number(claim.image_fraud_probability || 0).toFixed(1)}%</p>
                        <p className="text-sm mt-2 text-gray-600">Fraud Probability</p>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold mb-4">Damage Assessment</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {[
                          ["Overall Severity", claim.overall_damage_severity],
                          ["Damage Areas", String(claim.total_damage_areas || 0)],
                          ["Avg Damage %", `${Number(claim.average_damage_percentage || 0).toFixed(1)}%`],
                          ["Max Damage %", `${Number(claim.max_damage_percentage || 0).toFixed(1)}%`],
                        ].map(([label, value]) => (
                          <div key={label} className="p-4 border rounded-lg">
                            <p className="text-sm text-gray-600">{label}</p>
                            <p className="text-xl font-bold">{value}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* AI REASONING TAB */}
              <TabsContent value="reasoning">
                <Card>
                  <CardHeader>
                    <CardTitle>AI Reasoning - Fraud Explainability</CardTitle>
                    <CardDescription>SHAP-based explanation of fraud detection factors</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Executive Summary - Human Readable Paragraphs */}
                    {claim.fraud_explanation && (
                      <div className="p-6 border rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50">
                        <h4 className="font-semibold text-blue-900 mb-4 flex items-center gap-2">
                          <Brain className="h-5 w-5" /> Executive Summary
                        </h4>
                        
                        {/* Overall Assessment Paragraph */}
                        <div className="mb-4 p-4 bg-white rounded-lg border border-blue-100">
                          <p className="text-gray-800 leading-relaxed">
                            <strong>Overall Assessment:</strong> Our AI analysis has determined this claim carries a 
                            <span className={`font-bold ${claim.fraud_explanation.base_fraud_rate > 0.5 ? 'text-red-600' : 'text-yellow-600'}`}>
                              {claim.fraud_explanation.base_fraud_rate > 0.7 ? ' high' : 
                               claim.fraud_explanation.base_fraud_rate > 0.4 ? ' moderate' : ' elevated'}
                            </span> risk profile. The base fraud rate for similar claims in the market is 
                            <span className="font-semibold text-blue-700">
                              {claim.fraud_explanation.base_fraud_rate <= 1 && claim.fraud_explanation.base_fraud_rate >= 0
                                ? ` ${(claim.fraud_explanation.base_fraud_rate * 100).toFixed(1)}%`
                                : ` based on complex risk modeling`}
                            </span>, while this specific claim shows additional risk factors that require careful consideration.
                          </p>
                        </div>

                        {/* Key Fraud Indicators Paragraph */}
                        {claim.fraud_explanation.fraud_factors && claim.fraud_explanation.fraud_factors.length > 0 && (
                          <div className="mb-4 p-4 bg-red-50 rounded-lg border border-red-100">
                            <p className="text-gray-800 leading-relaxed">
                              <strong className="text-red-700">Primary Fraud Indicators:</strong> The most significant concern is 
                              <span className="font-bold text-red-600"> {claim.fraud_explanation.top_factor || 'multiple risk factors'}</span>.
                              {claim.fraud_explanation.fraud_factors.slice(0, 2).map((factor: any, idx: number) => (
                                <span key={idx}>
                                  {idx === 0 ? ' Specifically, ' : ' Additionally, '}
                                  the <span className="font-semibold"> {factor.label.toLowerCase()}</span> 
                                  <span className="text-red-600"> increases fraud risk by {Math.abs(factor.shap_value * 100).toFixed(1)}%</span>
                                  {factor.impact_level === 'HIGH' ? ' and represents a critical warning sign' : 
                                   factor.impact_level === 'MEDIUM' ? ' and warrants further investigation' : 
                                   ' and should be monitored closely'}.
                                </span>
                              ))}
                              {claim.fraud_explanation.fraud_factors.length > 2 && 
                                <span className="text-red-600"> Additional risk factors have also been identified.</span>
                              }
                            </p>
                          </div>
                        )}

                        {/* Legitimacy Factors Paragraph */}
                        {claim.fraud_explanation.legitimate_factors && claim.fraud_explanation.legitimate_factors.length > 0 && (
                          <div className="mb-4 p-4 bg-green-50 rounded-lg border border-green-100">
                            <p className="text-gray-800 leading-relaxed">
                              <strong className="text-green-700">Mitigating Factors:</strong> On the positive side, 
                              {claim.fraud_explanation.legitimate_factors.slice(0, 2).map((factor: any, idx: number) => (
                                <span key={idx}>
                                  {idx === 0 ? ' ' : ' '}
                                  the <span className="font-semibold text-green-600"> {factor.label.toLowerCase()}</span>
                                  <span className="text-green-600"> reduces fraud risk by {Math.abs(factor.shap_value * 100).toFixed(1)}%</span>
                                  {idx < claim.fraud_explanation.legitimate_factors.length - 1 && idx < 1 ? ', ' : '.'}
                                </span>
                              ))}
                              {claim.fraud_explanation.legitimate_factors.length > 2 && 
                                <span className="text-green-600"> These legitimate indicators help balance the overall risk assessment.</span>
                              }
                            </p>
                          </div>
                        )}

                        {/* Recommendation Paragraph */}
                        <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-100">
                          <p className="text-gray-800 leading-relaxed">
                            <strong className="text-yellow-700">Recommendation:</strong> 
                            {claim.fraud_explanation.fraud_factors && claim.fraud_explanation.fraud_factors.some((f: any) => f.impact_level === 'HIGH') 
                              ? ' Given the presence of high-risk indicators, we strongly recommend field verification by a surveyor before proceeding with this claim. The combination of risk factors warrants thorough investigation.'
                              : claim.fraud_explanation.fraud_factors && claim.fraud_explanation.fraud_factors.length > 2
                              ? ' While some risk factors are present, they may be within acceptable ranges. Consider additional documentation or partial verification before final decision.'
                              : ' This claim shows relatively normal patterns with minor risk considerations. Standard verification procedures should suffice.'}
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Main Analysis Summary */}
                    {claim.fraud_explanation ? (
                      <>
                        {/* Summary Section */}
                        <div className="p-4 border rounded-lg bg-blue-50">
                          <h4 className="font-semibold text-blue-900 mb-2">Summary</h4>
                          <div className="grid md:grid-cols-2 gap-4">
                            <div>
                              <p className="text-sm text-blue-800">
                                <strong>Top Fraud Factor:</strong> {claim.fraud_explanation.top_factor || 'N/A'}
                              </p>
                              <p className="text-sm text-blue-800 mt-1">
                                <strong>Explanation Method:</strong> {claim.fraud_explanation.method === 'rule_based' ? 'Rule-based Heuristics' : 'SHAP (SHapley Additive exPlanations)'}
                              </p>
                            </div>
                            <div className="p-3 bg-white/50 rounded-lg border border-blue-100">
                              <p className="text-xs text-blue-400">Base Fraud Rate (Population Average)</p>
                              <p className="text-sm font-semibold text-blue-700">
                                {claim.fraud_explanation.base_fraud_rate <= 1 && claim.fraud_explanation.base_fraud_rate >= 0
                                  ? `${(claim.fraud_explanation.base_fraud_rate * 100).toFixed(1)}%`
                                  : `Log-odds: ${Number(claim.fraud_explanation.base_fraud_rate).toFixed(3)}`}
                              </p>
                              <p className="text-[10px] text-blue-500 mt-0.5">
                                Market-wide average risk for similar claims
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* Fraud Factors */}
                        {claim.fraud_explanation.fraud_factors && claim.fraud_explanation.fraud_factors.length > 0 && (
                          <div>
                            <h4 className="font-semibold text-red-700 mb-3 flex items-center gap-2">
                              <span>🔴</span> Fraud Indicators (Increased Risk)
                            </h4>
                            <div className="space-y-3">
                              {claim.fraud_explanation.fraud_factors.map((factor: any, idx: number) => (
                                <div key={idx} className="border rounded-lg p-4 bg-red-50 border-red-200">
                                  <div className="flex justify-between items-start mb-2">
                                    <div className="flex-1">
                                      <p className="font-medium text-red-900">{factor.label}</p>
                                      {/* Raw feature hidden - only for audit trace if needed */}
                                      {/* <p className="text-[10px] text-red-300">Internal Key: {factor.feature}</p> */}
                                    </div>
                                    <div className="text-right">
                                      <Badge className={
                                        factor.impact_level === 'HIGH' ? 'bg-red-600' :
                                          factor.impact_level === 'MEDIUM' ? 'bg-orange-500' :
                                            'bg-yellow-500'
                                      }>
                                        {factor.impact_level}
                                      </Badge>
                                    </div>
                                  </div>

                                  {/* SHAP Impact Bar */}
                                  <div className="mt-3">
                                    <div className="flex justify-between items-center text-xs text-gray-600 mb-1">
                                      <span>SHAP Impact</span>
                                      <span>{(factor.shap_value * 100).toFixed(2)}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                      <div
                                        className="bg-red-500 h-2 rounded-full transition-all duration-300"
                                        style={{ width: `${Math.min(Math.abs(factor.shap_value) * 100, 100)}%` }}
                                      ></div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Legitimate Factors */}
                        {claim.fraud_explanation.legitimate_factors && claim.fraud_explanation.legitimate_factors.length > 0 && (
                          <div>
                            <h4 className="font-semibold text-green-700 mb-3 flex items-center gap-2">
                              <span>🟢</span> Legitimacy Indicators (Decreased Risk)
                            </h4>
                            <div className="space-y-3">
                              {claim.fraud_explanation.legitimate_factors.map((factor: any, idx: number) => (
                                <div key={idx} className="border rounded-lg p-4 bg-green-50 border-green-200">
                                  <div className="flex justify-between items-start mb-2">
                                    <div className="flex-1">
                                      <p className="font-medium text-green-900">{factor.label}</p>
                                      {/* Raw feature hidden */}
                                      {/* <p className="text-[10px] text-green-300">Internal Key: {factor.feature}</p> */}
                                    </div>
                                    <div className="text-right">
                                      <Badge className={
                                        factor.impact_level === 'HIGH' ? 'bg-green-600' :
                                          factor.impact_level === 'MEDIUM' ? 'bg-lime-500' :
                                            'bg-green-400'
                                      }>
                                        {factor.impact_level}
                                      </Badge>
                                    </div>
                                  </div>

                                  {/* SHAP Impact Bar (Negative) */}
                                  <div className="mt-3">
                                    <div className="flex justify-between items-center text-xs text-gray-600 mb-1">
                                      <span>SHAP Impact (Reduces Risk)</span>
                                      <span>{(Math.abs(factor.shap_value) * 100).toFixed(2)}%</span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                      <div
                                        className="bg-green-500 h-2 rounded-full transition-all duration-300"
                                        style={{ width: `${Math.min(Math.abs(factor.shap_value) * 100, 100)}%` }}
                                      ></div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Technical Details */}
                        <div className="p-4 border rounded-lg bg-gray-50">
                          <h4 className="font-semibold text-gray-900 mb-2">Technical Details</h4>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                              <p className="text-gray-600">Fraud Factors Count</p>
                              <p className="font-bold">{claim.fraud_explanation.fraud_factors?.length || 0}</p>
                            </div>
                            <div>
                              <p className="text-gray-600">Legitimate Factors Count</p>
                              <p className="font-bold">{claim.fraud_explanation.legitimate_factors?.length || 0}</p>
                            </div>
                            <div>
                              <p className="text-gray-600">Total Factors Analyzed</p>
                              <p className="font-bold">
                                {(claim.fraud_explanation.fraud_factors?.length || 0) + (claim.fraud_explanation.legitimate_factors?.length || 0)}
                              </p>
                            </div>
                            <div>
                              <p className="text-gray-600">Explanation Available</p>
                              <p className="font-bold text-green-600">✅ Yes</p>
                            </div>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-8">
                        <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">No AI Reasoning Available</h3>
                        <p className="text-gray-600 max-w-md mx-auto">
                          Fraud explanation data is not available for this claim. This could be because:
                        </p>
                        <ul className="text-sm text-gray-600 mt-2 max-w-md mx-auto text-left">
                          <li>• The claim was processed before SHAP explanations were implemented</li>
                          <li>• SHAP explanation generation failed during processing</li>
                          <li>• The explanation data was not saved to the database</li>
                        </ul>
                        <p className="text-xs text-gray-500 mt-4">
                          Claim ID: {claim.id} | Check backend logs for more details
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* DOCUMENTS TAB */}
              <TabsContent value="documents">
                <Card>
                  <CardHeader>
                    <CardTitle>Submitted Documents</CardTitle>
                    <CardDescription>Required documents and verification status</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {[
                      { label: "Driving License", value: claim.dl_number },
                      { label: "Vehicle Registration", value: claim.vehicle_reg_no },
                      { label: "FIR / Police Report", value: claim.fir_number },
                    ].map(({ label, value }) => (
                      <div key={label} className={`p-6 rounded-lg border flex flex-col sm:flex-row justify-between sm:items-center gap-3 ${value ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"}`}>
                        <div>
                          <p className="font-semibold text-lg">{label}</p>
                          <p className="text-sm text-gray-600 mt-1">{value || "Not provided"}</p>
                        </div>
                        <Badge variant="outline" className={value ? "bg-green-100" : "bg-red-100"}>
                          {value ? "✓ Provided" : "✗ Missing"}
                        </Badge>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* IMAGES TAB */}
              <TabsContent value="images">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <ImageIcon className="h-5 w-5" /> Damage Images ({claim.images?.length || 0})
                    </CardTitle>
                    <CardDescription>Uploaded damage images with AI analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {claim.images && claim.images.length > 0 ? (
                      <div className="grid md:grid-cols-2 gap-6">
                        {claim.images.map((img: any, idx: number) => {
                          const imageUrl = img.image_url ? `${MEDIA_BASE_URL}${img.image_url}` : null;
                          const annotatedUrl = img.annotated_image_url ? `${MEDIA_BASE_URL}${img.annotated_image_url}` : null;
                          return (
                            <div key={idx} className="border rounded-lg overflow-hidden">
                              {annotatedUrl ? (
                                <div className="relative">
                                  <img src={annotatedUrl} alt={`Damage ${idx + 1}`} className="w-full h-64 object-cover" onError={(e: any) => { if (imageUrl) e.target.src = imageUrl; }} />
                                  <Badge className="absolute top-2 right-2 bg-black">Annotated</Badge>
                                </div>
                              ) : imageUrl ? (
                                <img src={imageUrl} alt={`Damage ${idx + 1}`} className="w-full h-64 object-cover" />
                              ) : (
                                <div className="w-full h-64 bg-gray-200 flex items-center justify-center">
                                  <p className="text-gray-500">No image available</p>
                                </div>
                              )}
                              <div className="p-4 bg-gray-50">
                                <p className="font-semibold mb-3">Image {idx + 1} Analysis</p>
                                <div className="grid grid-cols-2 gap-3 text-sm">
                                  {[
                                    ["Damage %", `${Number(img.damage_percentage || 0).toFixed(1)}%`],
                                    ["Confidence", `${Number(img.confidence || 0).toFixed(1)}%`],
                                    ["Areas Detected", String(img.damage_areas_count || 0)],
                                  ].map(([label, value]) => (
                                    <div key={label} className="p-2 bg-white rounded border">
                                      <p className="text-gray-600">{label}</p>
                                      <p className="font-bold text-lg">{value}</p>
                                    </div>
                                  ))}
                                  <div className="p-2 bg-white rounded border">
                                    <p className="text-gray-600">Severity</p>
                                    <Badge className={getRiskBadgeColor(img.severity_level)}>{img.severity_level}</Badge>
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="text-center py-12 border rounded-lg">
                        <ImageIcon className="h-12 w-12 text-gray-400 mx-auto mb-3" />
                        <p className="text-gray-500">No images available</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* DATABASE TAB */}
              <TabsContent value="database">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Database className="h-5 w-5" /> Complete Database Record
                    </CardTitle>
                    <CardDescription>Full claim data from database</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div>
                      <h4 className="font-semibold mb-3">Claim Fields</h4>
                      <div className="grid md:grid-cols-2 gap-3">
                        {Object.entries(claim).filter(([key]) =>
                          !["policyholder", "images", "history", "detailed_analysis"].includes(key)
                        ).map(([key, value]) => (
                          <div key={key} className="p-3 bg-gray-50 rounded border">
                            <p className="text-xs text-gray-600 uppercase font-medium">{key.replace(/_/g, " ")}</p>
                            <p className="text-sm font-mono mt-1 break-all">
                              {value === null || value === undefined ? "null" : typeof value === "object" ? JSON.stringify(value) : String(value)}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-semibold">Complete JSON Export</h4>
                        <Button size="sm" variant="outline" onClick={() => {
                          const blob = new Blob([JSON.stringify(claim, null, 2)], { type: "application/json" });
                          const url = URL.createObjectURL(blob);
                          const link = document.createElement("a");
                          link.href = url;
                          link.download = `claim_${claim.claim_number}_data.json`;
                          link.click();
                        }}>Download JSON</Button>
                      </div>
                      <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto max-h-96">
                        <pre className="text-xs">{JSON.stringify(claim, null, 2)}</pre>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* HISTORY TAB */}
              <TabsContent value="history">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><Calendar className="h-5 w-5" /> Claim History & Audit Trail</CardTitle>
                    <CardDescription>Complete history of all actions and changes</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {claim.history && claim.history.length > 0 ? (
                      <div className="space-y-3">
                        {claim.history.map((entry: any, idx: number) => (
                          <div key={idx} className="p-5 bg-gray-50 rounded-lg border-l-4 border-black">
                            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-2 mb-2">
                              <div>
                                <p className="font-bold text-lg capitalize">{entry.action}</p>
                                {entry.old_status && entry.new_status && (
                                  <p className="text-sm text-gray-600 mt-1 flex flex-wrap items-center gap-2">
                                    Status: <Badge className={getStatusBadgeColor(entry.old_status)}>{entry.old_status}</Badge>
                                    <span>→</span>
                                    <Badge className={getStatusBadgeColor(entry.new_status)}>{entry.new_status}</Badge>
                                  </p>
                                )}
                              </div>
                              <p className="text-sm text-gray-500 whitespace-nowrap">{new Date(entry.timestamp).toLocaleString()}</p>
                            </div>
                            {entry.performed_by && <p className="text-sm text-gray-600 mt-2"><strong>Performed by:</strong> {entry.performed_by}</p>}
                            {entry.notes && (
                              <div className="mt-3 p-3 bg-white rounded border">
                                <p className="text-sm text-gray-700 italic">{entry.notes}</p>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-12 border rounded-lg">
                        <Calendar className="h-12 w-12 text-gray-400 mx-auto mb-3" />
                        <p className="text-gray-500">No history available</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* SURVEYOR TAB */}
              <TabsContent value="surveyor">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2"><UserCheck className="h-5 w-5" /> Surveyor Information</CardTitle>
                    <CardDescription>Field survey assignment and report details</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid md:grid-cols-2 gap-4">
                      {[
                        ["Assigned Surveyor", claim.assigned_surveyor_name || "Not assigned"],
                        ["Assigned At", claim.assigned_at ? new Date(claim.assigned_at).toLocaleString() : "N/A"],
                        ["Survey Completed At", claim.survey_completed_at ? new Date(claim.survey_completed_at).toLocaleString() : "Pending"],
                        ["Damage Verified on Site", claim.damage_verified === true ? "✓ Yes" : claim.damage_verified === false ? "✗ No" : "Pending"],
                        ["Recommendation", claim.surveyor_recommendation || "Pending"],
                        ["Surveyor Assessed Amount", claim.surveyor_assessed_amount ? formatCurrency(claim.surveyor_assessed_amount) : "Not provided"],
                      ].map(([label, value]) => (
                        <div key={label} className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600">{label}</p>
                          <p className="font-semibold">{value}</p>
                        </div>
                      ))}
                    </div>
                    {claim.surveyor_notes && (
                      <div className="p-4 bg-gray-50 rounded-lg border">
                        <p className="text-sm text-gray-600 mb-2">Field Survey Notes</p>
                        <p className="text-gray-800 leading-relaxed">{claim.surveyor_notes}</p>
                      </div>
                    )}
                    {claim.assigned_surveyor_name && (
                      <div className="p-4 border rounded-lg bg-green-50">
                        <h4 className="font-semibold text-green-900 mb-3 flex items-center gap-2">
                          <MessageCircle className="h-4 w-4" />
                          Communication Status
                        </h4>
                        <div className="grid md:grid-cols-2 gap-4 text-sm">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span className="text-gray-700">Chat Available</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Calendar className="h-3 w-3 text-blue-600" />
                            <span className="text-gray-700">Appointment Scheduling Enabled</span>
                          </div>
                        </div>
                        <p className="text-xs text-green-700 mt-2">
                          Customer and surveyor can communicate directly through the chat and appointment system.
                        </p>
                      </div>
                    )}
                    {claim.is_assigned_to_surveyor && (
                      <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800">
                        Final Claim Amount (AI or Surveyor Override): <strong>{formatCurrency(claim.final_claim_amount)}</strong>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>

            {/* QUICK STATS */}
            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><TrendingUp className="h-5 w-5" /> Quick Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  {[
                    { label: "Status", content: <Badge className={getStatusBadgeColor(claim.status)}>{claim.status}</Badge> },
                    { label: "Risk Level", content: <Badge className={getRiskBadgeColor(claim.risk_level)}>{claim.risk_level}</Badge> },
                    { label: "Fraud Detected", content: <p className="font-bold text-lg">{claim.fraud_detected ? "🚨 YES" : "✓ NO"}</p> },
                    { label: "Confidence", content: <p className="font-bold text-lg">{Number(claim.confidence_score || 0).toFixed(0)}%</p> },
                    { label: "Images", content: <p className="font-bold text-lg">{claim.total_images_submitted || 0}</p> },
                    { label: "Damage Areas", content: <p className="font-bold text-lg">{claim.total_damage_areas || 0}</p> },
                  ].map(({ label, content }) => (
                    <div key={label} className="text-center p-3 bg-gray-50 rounded border">
                      <p className="text-xs text-gray-600 mb-2">{label}</p>
                      {content}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </>
  );
}