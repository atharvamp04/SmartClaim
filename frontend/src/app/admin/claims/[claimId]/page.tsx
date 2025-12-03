"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  ArrowLeft, CheckCircle, XCircle, Loader2, 
  AlertTriangle, Calendar, User, DollarSign,
  FileText, Shield, Image as ImageIcon, Database,
  Activity, TrendingUp, Home, Clock, LogOut, Menu
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
  const [claim, setClaim] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [claimId, setClaimId] = useState(null);
  const [adminName, setAdminName] = useState("Admin");

  useEffect(() => {
    const username = localStorage.getItem("admin_name") || "Admin";
    setAdminName(username);
    
    // Get claim ID from URL
    const pathSegments = window.location.pathname.split('/');
    const id = pathSegments[pathSegments.length - 1];
    setClaimId(id);
  }, []);

  useEffect(() => {
    if (claimId) {
      fetchClaim();
    }
  }, [claimId]);

  const fetchClaim = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem("access_token");
      
      if (!token) {
        window.location.href = "/login";
        return;
      }

      const res = await fetch(`${API_BASE_URL}/claims/${claimId}/`, {
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      if (!res.ok) {
        if (res.status === 401) {
          window.location.href = "/login";
          return;
        }
        throw new Error(`Failed to load claim: ${res.status}`);
      }

      const data = await res.json();
      setClaim(data);
      setError(null);
    } catch (err) {
      console.error("Error fetching claim:", err);
      setError(err.message || "Failed to load claim");
    } finally {
      setLoading(false);
    }
  };

  const handleStatusUpdate = async (newStatus) => {
    const confirmMessage = newStatus === "Verified" 
      ? "Are you sure you want to APPROVE this claim?" 
      : "Are you sure you want to REJECT this claim?";
    
    if (!window.confirm(confirmMessage)) return;

    try {
      setProcessing(true);
      const token = localStorage.getItem("access_token");

      const res = await fetch(`${API_BASE_URL}/claims/${claimId}/status/`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          status: newStatus,
          notes: `Status updated to ${newStatus} by ${adminName}`
        })
      });

      if (!res.ok) throw new Error("Failed to update status");

      const data = await res.json();
      setClaim(data.claim);
      
      alert(`Claim ${newStatus.toLowerCase()} successfully!`);
      
      await fetchClaim();
    } catch (err) {
      console.error("Error updating status:", err);
      alert(err.message || "Failed to update claim status");
    } finally {
      setProcessing(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("admin_name");
    window.location.href = "/login";
  };

  const handleNavigation = (view) => {
    if (view === "dashboard") {
      window.location.href = "/admin";
    } else {
      window.location.href = `/admin?view=${view}`;
    }
  };

  const menuItems = [
    { title: "Dashboard", icon: Home, view: "dashboard" },
    { title: "All Claims", icon: TrendingUp, view: "all" },
    { title: "Pending Review", icon: Clock, view: "pending" },
    { title: "Verified", icon: CheckCircle, view: "verified" },
    { title: "Fraud", icon: XCircle, view: "fraud" },
    { title: "High Risk", icon: AlertTriangle, view: "high-risk" },
  ];

  const getRiskBadgeColor = (riskLevel) => {
    switch(riskLevel) {
      case "HIGH": return "bg-red-600 text-white";
      case "MEDIUM": return "bg-yellow-600 text-white";
      case "LOW": return "bg-green-600 text-white";
      case "CRITICAL": return "bg-red-900 text-white";
      default: return "bg-gray-600 text-white";
    }
  };

  const getStatusBadgeColor = (status) => {
    switch(status) {
      case "Verified": return "bg-green-600 text-white";
      case "Fraud": return "bg-red-600 text-white";
      case "Rejected": return "bg-red-800 text-white";
      case "Pending": return "bg-yellow-600 text-white";
      default: return "bg-gray-600 text-white";
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

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
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Dashboard
          </Button>
        </div>
      </div>
    );
  }

  return (
    <>
      <Sidebar className="border-r bg-white">
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
          <Button 
            variant="outline" 
            onClick={handleLogout} 
            className="w-full flex items-center justify-center gap-2 hover:bg-gray-100 transition-colors"
          >
            <LogOut className="h-4 w-4" /> 
            <span className="font-medium">Logout</span>
          </Button>
        </SidebarFooter>
      </Sidebar>

      <div className="flex-1 flex flex-col min-h-screen overflow-hidden">
        <header className="bg-white border-b px-4 py-3 lg:px-6 flex items-center justify-between sticky top-0 z-10">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              onClick={() => setOpen(!open)}
            >
              <Menu className="h-5 w-5" />
            </Button>
            <div className="flex items-center gap-4">
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => window.location.href = "/admin"}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">
                  Claim {claim.claim_number}
                </h1>
                <p className="text-xs text-gray-500">
                  Submitted {new Date(claim.submitted_at).toLocaleDateString()}
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <Badge className={getRiskBadgeColor(claim.risk_level)}>
              {claim.risk_level}
            </Badge>
            <Badge className={getStatusBadgeColor(claim.status)}>
              {claim.status}
            </Badge>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 py-6">
            {(claim.status === "Pending" || claim.status === "Fraud") && (
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="h-5 w-5" />
                    Review & Decision
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex flex-wrap gap-4">
                    <Button
                      onClick={() => handleStatusUpdate("Verified")}
                      disabled={processing}
                      className="bg-green-600 hover:bg-green-700 text-white flex items-center gap-2"
                      size="lg"
                    >
                      <CheckCircle className="h-5 w-5" />
                      {processing ? "Processing..." : "Approve Claim"}
                    </Button>
                    <Button
                      onClick={() => handleStatusUpdate("Rejected")}
                      disabled={processing}
                      className="bg-red-600 hover:bg-red-700 text-white flex items-center gap-2"
                      size="lg"
                    >
                      <XCircle className="h-5 w-5" />
                      {processing ? "Processing..." : "Reject Claim"}
                    </Button>
                  </div>
                  {claim.fraud_detected && claim.status === "Fraud" && (
                    <div className="p-4 bg-red-50 border border-red-300 rounded-lg">
                      <div className="flex items-start gap-3">
                        <AlertTriangle className="h-6 w-6 text-red-600 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="font-bold text-red-900">⚠️ FRAUD INDICATORS DETECTED</p>
                          <p className="text-sm text-red-800 mt-2">
                            This claim has been flagged with a <strong>{(claim.confidence_score).toFixed(1)}%</strong> fraud confidence score.
                            Please review all information carefully before making a decision.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {claim.status === "Verified" && (
              <div className="bg-green-50 border border-green-300 rounded-lg p-4 mb-6">
                <div className="flex items-center gap-3">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                  <div>
                    <p className="font-bold text-green-900">Claim Approved</p>
                    <p className="text-sm text-green-700">This claim has been verified and approved.</p>
                  </div>
                </div>
              </div>
            )}

            {claim.status === "Rejected" && (
              <div className="bg-red-50 border border-red-300 rounded-lg p-4 mb-6">
                <div className="flex items-center gap-3">
                  <XCircle className="h-6 w-6 text-red-600" />
                  <div>
                    <p className="font-bold text-red-900">Claim Rejected</p>
                    <p className="text-sm text-red-700">This claim has been rejected.</p>
                  </div>
                </div>
              </div>
            )}

            <Tabs defaultValue="overview" className="w-full">
              <TabsList className="w-full grid grid-cols-3 lg:grid-cols-6 mb-6">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="analysis">Analysis</TabsTrigger>
                <TabsTrigger value="documents">Documents</TabsTrigger>
                <TabsTrigger value="images">Images</TabsTrigger>
                <TabsTrigger value="database">Database</TabsTrigger>
                <TabsTrigger value="history">History</TabsTrigger>
              </TabsList>

              <TabsContent value="overview">
                <Card>
                  <CardHeader>
                    <CardTitle>Claim Overview</CardTitle>
                    <CardDescription>
                      Key information and details about this claim
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 border rounded-lg">
                        <p className="text-sm text-gray-600 flex items-center gap-2 mb-2">
                          <DollarSign className="h-4 w-4" />
                          Claim Amount
                        </p>
                        <p className="text-2xl font-bold">{formatCurrency(claim.claim_amount)}</p>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <p className="text-sm text-gray-600 flex items-center gap-2 mb-2">
                          <Activity className="h-4 w-4" />
                          Confidence Score
                        </p>
                        <p className="text-2xl font-bold">{(claim.confidence_score).toFixed(1)}%</p>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <p className="text-sm text-gray-600 flex items-center gap-2 mb-2">
                          <ImageIcon className="h-4 w-4" />
                          Images
                        </p>
                        <p className="text-2xl font-bold">{claim.total_images_submitted || 0}</p>
                      </div>
                      <div className="p-4 border rounded-lg">
                        <p className="text-sm text-gray-600 flex items-center gap-2 mb-2">
                          <AlertTriangle className="h-4 w-4" />
                          Damage Areas
                        </p>
                        <p className="text-2xl font-bold">{claim.total_damage_areas || 0}</p>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <User className="h-5 w-5" />
                        Policyholder Information
                      </h3>
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
                      <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <FileText className="h-5 w-5" />
                        Claim Details
                      </h3>
                      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                        <div className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600">Claim Number</p>
                          <p className="font-semibold">{claim.claim_number}</p>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600">Accident Date</p>
                          <p className="font-semibold">{new Date(claim.accident_date).toLocaleDateString()}</p>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600">Submitted Date</p>
                          <p className="font-semibold">{new Date(claim.submitted_at).toLocaleDateString()}</p>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600">DL Number</p>
                          <p className="font-semibold">{claim.dl_number || 'N/A'}</p>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600">Vehicle Reg No</p>
                          <p className="font-semibold">{claim.vehicle_reg_no || 'N/A'}</p>
                        </div>
                        <div className="p-4 bg-gray-50 rounded-lg border">
                          <p className="text-sm text-gray-600">FIR Number</p>
                          <p className="font-semibold">{claim.fir_number || 'N/A'}</p>
                        </div>
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

              <TabsContent value="analysis">
                <Card>
                  <CardHeader>
                    <CardTitle>Fraud Analysis</CardTitle>
                    <CardDescription>
                      AI-powered fraud detection analysis results
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="grid md:grid-cols-3 gap-4">
                      <div className={`p-6 rounded-lg border ${
                        claim.fraud_detected ? 'bg-red-50 border-red-300' : 'bg-green-50 border-green-300'
                      }`}>
                        <p className="text-sm font-medium text-gray-600 mb-2">Final Detection</p>
                        <p className={`text-3xl font-bold ${claim.fraud_detected ? 'text-red-700' : 'text-green-700'}`}>
                          {claim.fraud_detected ? '🚨 FRAUD' : '✓ LEGITIMATE'}
                        </p>
                        <p className="text-sm mt-2 text-gray-600">
                          Confidence: <strong>{(claim.confidence_score).toFixed(1)}%</strong>
                        </p>
                      </div>

                      <div className="p-6 border rounded-lg">
                        <p className="text-sm font-medium text-gray-600 mb-2">Tabular Analysis</p>
                        <p className="text-3xl font-bold">
                          {(claim.tabular_fraud_probability || 0).toFixed(1)}%
                        </p>
                        <p className="text-sm mt-2 text-gray-600">Fraud Probability</p>
                      </div>

                      <div className="p-6 border rounded-lg">
                        <p className="text-sm font-medium text-gray-600 mb-2">Image Analysis</p>
                        <p className="text-3xl font-bold">
                          {(claim.image_fraud_probability || 0).toFixed(1)}%
                        </p>
                        <p className="text-sm mt-2 text-gray-600">Fraud Probability</p>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold mb-4">Damage Assessment</h3>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 border rounded-lg">
                          <p className="text-sm text-gray-600">Overall Severity</p>
                          <p className="text-xl font-bold">{claim.overall_damage_severity}</p>
                        </div>
                        <div className="p-4 border rounded-lg">
                          <p className="text-sm text-gray-600">Damage Areas</p>
                          <p className="text-xl font-bold">{claim.total_damage_areas || 0}</p>
                        </div>
                        <div className="p-4 border rounded-lg">
                          <p className="text-sm text-gray-600">Avg Damage %</p>
                          <p className="text-xl font-bold">
                            {(claim.average_damage_percentage || 0).toFixed(1)}%
                          </p>
                        </div>
                        <div className="p-4 border rounded-lg">
                          <p className="text-sm text-gray-600">Max Damage %</p>
                          <p className="text-xl font-bold">
                            {(claim.max_damage_percentage || 0).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

<TabsContent value="documents">
  <Card>
    <CardHeader>
      <CardTitle>Submitted Documents</CardTitle>
      <CardDescription>
        Required documents and verification status
      </CardDescription>
    </CardHeader>
    <CardContent className="space-y-4">
      <div className={`p-6 rounded-lg border flex flex-col sm:flex-row justify-between sm:items-center gap-3 ${
        claim.dl_number ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
      }`}>
        <div>
          <p className="font-semibold text-lg">Driving License</p>
          <p className="text-sm text-gray-600 mt-1">{claim.dl_number || 'Not provided'}</p>
          
        </div>
        <Badge variant="outline" className={claim.dl_number ? 'bg-green-100' : 'bg-red-100'}>
          {claim.dl_number ? '✓ Provided' : '✗ Missing'}
        </Badge>
      </div>
      
      <div className={`p-6 rounded-lg border flex flex-col sm:flex-row justify-between sm:items-center gap-3 ${
        claim.vehicle_reg_no ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
      }`}>
        <div>
          <p className="font-semibold text-lg">Vehicle Registration</p>
          <p className="text-sm text-gray-600 mt-1">{claim.vehicle_reg_no || 'Not provided'}</p>
          
        </div>
        <Badge variant="outline" className={claim.vehicle_reg_no ? 'bg-green-100' : 'bg-red-100'}>
          {claim.vehicle_reg_no ? '✓ Provided' : '✗ Missing'}
        </Badge>
      </div>
      
      <div className={`p-6 rounded-lg border flex flex-col sm:flex-row justify-between sm:items-center gap-3 ${
        claim.fir_number ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
      }`}>
        <div>
          <p className="font-semibold text-lg">FIR / Police Report</p>
          <p className="text-sm text-gray-600 mt-1">{claim.fir_number || 'Not provided'}</p>
        
        </div>
        <Badge variant="outline" className={claim.fir_number ? 'bg-green-100' : 'bg-red-100'}>
          {claim.fir_number ? '✓ Provided' : '✗ Missing'}
        </Badge>
      </div>
    </CardContent>
  </Card>
</TabsContent>

              <TabsContent value="images">
  <Card>
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <ImageIcon className="h-5 w-5" />
        Damage Images ({claim.images?.length || 0})
      </CardTitle>
      <CardDescription>
        Uploaded damage images with AI analysis
      </CardDescription>
    </CardHeader>
    <CardContent>
      {claim.images && claim.images.length > 0 ? (
        <div className="grid md:grid-cols-2 gap-6">
          {claim.images.map((img, idx) => {
            const imageUrl = img.image_url ? `${MEDIA_BASE_URL}${img.image_url}` : null;
            const annotatedUrl = img.annotated_image_url ? `${MEDIA_BASE_URL}${img.annotated_image_url}` : null;
            
            return (
            <div key={idx} className="border rounded-lg overflow-hidden">
              {annotatedUrl ? (
                <div className="relative">
                  <img
                    src={annotatedUrl}
                    alt={`Damage ${idx + 1} - Annotated`}
                    className="w-full h-64 object-cover"
                    onError={(e) => {
                      console.error(`Failed to load annotated image: ${annotatedUrl}`);
                      if (imageUrl) {
                        e.target.src = imageUrl;
                      }
                    }}
                  />
                  <Badge className="absolute top-2 right-2 bg-black">Annotated</Badge>
                </div>
              ) : imageUrl ? (
                <img
                  src={imageUrl}
                  alt={`Damage ${idx + 1}`}
                  className="w-full h-64 object-cover"
                  onError={(e) => {
                    console.error(`Failed to load image: ${imageUrl}`);
                    e.target.style.display = 'none';
                  }}
                />
              ) : (
                <div className="w-full h-64 bg-gray-200 flex items-center justify-center">
                  <p className="text-gray-500">No image available</p>
                </div>
              )}
              <div className="p-4 bg-gray-50">
                <p className="font-semibold mb-3">Image {idx + 1} Analysis</p>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="p-2 bg-white rounded border">
                    <p className="text-gray-600">Damage %</p>
                    <p className="font-bold text-lg">{img.damage_percentage?.toFixed(1)}%</p>
                  </div>
                  <div className="p-2 bg-white rounded border">
                    <p className="text-gray-600">Severity</p>
                    <Badge className={getRiskBadgeColor(img.severity_level)}>
                      {img.severity_level}
                    </Badge>
                  </div>
                  <div className="p-2 bg-white rounded border">
                    <p className="text-gray-600">Confidence</p>
                    <p className="font-bold text-lg">{(img.confidence || 0).toFixed(1)}%</p>
                  </div>
                  <div className="p-2 bg-white rounded border">
                    <p className="text-gray-600">Areas Detected</p>
                    <p className="font-bold text-lg">{img.damage_areas_count || 0}</p>
                  </div>
                </div>
                {img.damage_areas && img.damage_areas.length > 0 && (
                  <div className="mt-3 p-2 bg-gray-100 rounded border">
                    <p className="text-xs text-gray-600 mb-1">Detected Areas:</p>
                    <div className="flex flex-wrap gap-1">
                      {img.damage_areas.map((area, areaIdx) => (
                        <Badge key={areaIdx} variant="outline" className="text-xs">
                          {area}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
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

              <TabsContent value="database">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Database className="h-5 w-5" />
                      Complete Database Record
                    </CardTitle>
                    <CardDescription>
                      Full claim data from database
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div>
                      <h4 className="font-semibold mb-3">Claim Fields</h4>
                      <div className="grid md:grid-cols-2 gap-3">
                        {Object.entries(claim).filter(([key]) => 
                          !['policyholder', 'images', 'history', 'detailed_analysis'].includes(key)
                        ).map(([key, value]) => (
                          <div key={key} className="p-3 bg-gray-50 rounded border">
                            <p className="text-xs text-gray-600 uppercase font-medium">{key.replace(/_/g, ' ')}</p>
                            <p className="text-sm font-mono mt-1 break-all">
                              {value === null || value === undefined ? 'null' : 
                               typeof value === 'object' ? JSON.stringify(value) : 
                               String(value)}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>

                    {claim.policyholder && (
                      <div>
                        <h4 className="font-semibold mb-3">Policyholder Data</h4>
                        <div className="grid md:grid-cols-2 gap-3">
                          {Object.entries(claim.policyholder).map(([key, value]) => (
                            <div key={key} className="p-3 bg-gray-50 rounded border">
                              <p className="text-xs text-gray-600 uppercase font-medium">{key.replace(/_/g, ' ')}</p>
                              <p className="text-sm font-mono mt-1 break-all">
                                {value === null || value === undefined ? 'null' : String(value)}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-semibold">Complete JSON Export</h4>
                        <Button 
                          size="sm" 
                          variant="outline"
                          onClick={() => {
                            const dataStr = JSON.stringify(claim, null, 2);
                            const dataBlob = new Blob([dataStr], { type: 'application/json' });
                            const url = URL.createObjectURL(dataBlob);
                            const link = document.createElement('a');
                            link.href = url;
                            link.download = `claim_${claim.claim_number}_data.json`;
                            link.click();
                          }}
                        >
                          Download JSON
                        </Button>
                      </div>
                      <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto max-h-96">
                        <pre className="text-xs">{JSON.stringify(claim, null, 2)}</pre>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="history">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Calendar className="h-5 w-5" />
                      Claim History & Audit Trail
                    </CardTitle>
                    <CardDescription>
                      Complete history of all actions and changes
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {claim.history && claim.history.length > 0 ? (
                      <div className="space-y-3">
                        {claim.history.map((entry, idx) => (
                          <div key={idx} className="p-5 bg-gray-50 rounded-lg border-l-4 border-black">
                            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-2 mb-2">
                              <div>
                                <p className="font-bold text-lg capitalize">{entry.action}</p>
                                {entry.old_status && entry.new_status && (
                                  <p className="text-sm text-gray-600 mt-1 flex flex-wrap items-center gap-2">
                                    Status changed: 
                                    <Badge className={`${getStatusBadgeColor(entry.old_status)}`}>
                                      {entry.old_status}
                                    </Badge>
                                    <span>→</span>
                                    <Badge className={getStatusBadgeColor(entry.new_status)}>
                                      {entry.new_status}
                                    </Badge>
                                  </p>
                                )}
                              </div>
                              <p className="text-sm text-gray-500 whitespace-nowrap">
                                {new Date(entry.timestamp).toLocaleString()}
                              </p>
                            </div>
                            {entry.performed_by && (
                              <p className="text-sm text-gray-600 mt-2">
                                <strong>Performed by:</strong> {entry.performed_by}
                              </p>
                            )}
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
            </Tabs>

            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Quick Statistics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  <div className="text-center p-3 bg-gray-50 rounded border">
                    <p className="text-xs text-gray-600 mb-2">Status</p>
                    <Badge className={getStatusBadgeColor(claim.status)}>
                      {claim.status}
                    </Badge>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded border">
                    <p className="text-xs text-gray-600 mb-2">Risk Level</p>
                    <Badge className={getRiskBadgeColor(claim.risk_level)}>
                      {claim.risk_level}
                    </Badge>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded border">
                    <p className="text-xs text-gray-600 mb-2">Fraud Detected</p>
                    <p className="font-bold text-lg">{claim.fraud_detected ? '🚨 YES' : '✓ NO'}</p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded border">
                    <p className="text-xs text-gray-600 mb-2">Confidence</p>
                    <p className="font-bold text-lg">{claim.confidence_score?.toFixed(0)}%</p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded border">
                    <p className="text-xs text-gray-600 mb-2">Images</p>
                    <p className="font-bold text-lg">{claim.total_images_submitted || 0}</p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 rounded border">
                    <p className="text-xs text-gray-600 mb-2">Damage Areas</p>
                    <p className="font-bold text-lg">{claim.total_damage_areas || 0}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </>
  );
}