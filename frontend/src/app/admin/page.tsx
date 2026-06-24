"use client";

import React, { useEffect, useState } from "react";
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
import { 
  Home, Clock, CheckCircle, XCircle, LogOut, AlertTriangle, 
  TrendingUp, Search, Loader2, Eye, Menu, DollarSign, Users,
  FileText, BarChart3, PieChart, Activity
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSidebar } from "@/components/ui/sidebar";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, PieChart as RechartsPieChart, Pie, Cell, LineChart, Line, AreaChart, Area, ResponsiveContainer, Legend } from "recharts";

const API_BASE_URL = `/api/detection`;

export default function AdminDashboard() {
  const router = useRouter();
  const { open, setOpen } = useSidebar();
  const [adminName, setAdminName] = useState<string>("Admin");
  const [activeView, setActiveView] = useState<string>("dashboard");
  const [claims, setClaims] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [riskFilter, setRiskFilter] = useState<string>("all");
  const [statistics, setStatistics] = useState<any>(null);

  useEffect(() => {
    const username = localStorage.getItem("admin_name") || "Admin";
    setAdminName(username);
    
    const token = localStorage.getItem("access_token");
    if (!token) {
      router.push("/login");
      return;
    }
    
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    await Promise.all([
      fetchClaims(),
      fetchStatistics()
    ]);
  };

  const fetchClaims = async (statusFilter?: string) => {
    try {
      setLoading(true);
      const token = localStorage.getItem("access_token");
      
      let endpoint = `${API_BASE_URL}/claims/`;
      
      // Apply specific endpoint based on view
      if (statusFilter === "pending") {
        // Fetch both pending and fraud claims
        const [pendingRes, fraudRes] = await Promise.all([
          fetch(`${API_BASE_URL}/claims/pending/`, {
            headers: {
              "Authorization": `Bearer ${token}`,
              "Content-Type": "application/json"
            }
          }),
          fetch(`${API_BASE_URL}/claims/fraud/`, {
            headers: {
              "Authorization": `Bearer ${token}`,
              "Content-Type": "application/json"
            }
          })
        ]);

        if (!pendingRes.ok || !fraudRes.ok) {
          throw new Error('Failed to load claims');
        }

        const pendingData = await pendingRes.json();
        const fraudData = await fraudRes.json();
        
        const combinedClaims = [
          ...(pendingData.claims || []),
          ...(fraudData.claims || [])
        ];
        
        setClaims(combinedClaims);
        setError(null);
        setLoading(false);
        return;
      } else if (statusFilter === "verified") {
        endpoint = `${API_BASE_URL}/claims/verified/`;
      } else if (statusFilter === "fraud") {
        endpoint = `${API_BASE_URL}/claims/fraud/`;
      } else if (statusFilter === "high-risk") {
        endpoint = `${API_BASE_URL}/claims/high-risk/`;
      }

      const res = await fetch(endpoint, {
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      if (!res.ok) {
        throw new Error(`Failed to load claims: ${res.status}`);
      }

      const data = await res.json();
      
      if (data.claims) {
        setClaims(data.claims);
      } else if (Array.isArray(data)) {
        setClaims(data);
      } else {
        setClaims([]);
      }
      
      setError(null);
    } catch (err: any) {
      console.error("Error fetching claims:", err);
      setError(err.message || "Failed to load claims");
      setClaims([]);
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const token = localStorage.getItem("access_token");
      
      const res = await fetch(`${API_BASE_URL}/claims/statistics/`, {
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      if (res.ok) {
        const data = await res.json();
        setStatistics(data.statistics || data);
      }
    } catch (err) {
      console.error("Error fetching statistics:", err);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      fetchClaims();
      return;
    }

    try {
      setLoading(true);
      const token = localStorage.getItem("access_token");
      
      const params = new URLSearchParams({
        q: searchQuery
      });

      if (riskFilter !== "all") {
        params.append("risk_level", riskFilter.toUpperCase());
      }

      const res = await fetch(`${API_BASE_URL}/claims/search/?${params.toString()}`, {
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json"
        }
      });

      if (!res.ok) throw new Error("Search failed");

      const data = await res.json();
      setClaims(data.claims || []);
      setError(null);
    } catch (err: any) {
      console.error("Search error:", err);
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("admin_name");
    router.push("/login");
  };

  const handleViewClaim = (claimId: number) => {
    router.push(`/admin/claims/${claimId}`);
  };

  const menuItems = [
    { title: "Dashboard", icon: Home, view: "dashboard" },
    { title: "All Claims", icon: TrendingUp, view: "all" },
    { title: "Pending Review", icon: Clock, view: "pending" },
    { title: "Verified", icon: CheckCircle, view: "verified" },
    { title: "Fraud", icon: XCircle, view: "fraud" },
    { title: "High Risk", icon: AlertTriangle, view: "high-risk" },
  ];

  useEffect(() => {
    if (activeView !== "dashboard") {
      if (activeView === "all") {
        fetchClaims();
      } else {
        fetchClaims(activeView);
      }
    }
  }, [activeView]);

  const getFilteredClaims = () => {
    let filtered = claims;

    if (riskFilter !== "all") {
      filtered = filtered.filter(c => c.risk_level === riskFilter.toUpperCase());
    }

    return filtered;
  };

  const getStats = () => {
    if (statistics) {
      return {
        total: statistics.total_claims || 0,
        verified: statistics.verified_claims || 0,
        fraud: statistics.fraud_claims || 0,
        pending: statistics.pending_claims || 0,
        rejected: statistics.rejected_claims || 0,
        highRisk: statistics.high_risk_count || 0,
        avgConfidence: statistics.avg_confidence_score || 0,
        avgAmount: statistics.avg_claim_amount || 0,
      };
    }

    const total = claims.length;
    const verified = claims.filter(c => c.status === "Verified").length;
    const fraud = claims.filter(c => c.status === "Fraud").length;
    const pending = claims.filter(c => c.status === "Pending" || c.status === "Fraud" || !c.status).length;
    const rejected = claims.filter(c => c.status === "Rejected").length;
    const highRisk = claims.filter(c => c.risk_level === "HIGH").length;
    
    return { total, verified, fraud, pending, rejected, highRisk, avgConfidence: 0, avgAmount: 0 };
  };

  const stats = getStats();

  // Chart data preparation functions
  const getStatusDistributionData = () => [
    { name: 'Verified', value: stats.verified, color: '#10b981' },
    { name: 'Fraud', value: stats.fraud, color: '#ef4444' },
    { name: 'Pending', value: stats.pending, color: '#f59e0b' },
    { name: 'Rejected', value: stats.rejected, color: '#6b7280' },
  ].filter(item => item.value > 0);

  const getRiskDistributionData = () => {
    const riskCounts = claims.reduce((acc: any, claim) => {
      const risk = claim.risk_level || 'LOW';
      acc[risk] = (acc[risk] || 0) + 1;
      return acc;
    }, {});

    return [
      { name: 'Low', value: riskCounts.LOW || 0, color: '#10b981' },
      { name: 'Medium', value: riskCounts.MEDIUM || 0, color: '#f59e0b' },
      { name: 'High', value: riskCounts.HIGH || 0, color: '#ef4444' },
      { name: 'Critical', value: riskCounts.CRITICAL || 0, color: '#7c2d12' },
    ].filter(item => item.value > 0);
  };

  const getMonthlyTrendData = () => {
    const monthlyData = claims.reduce((acc: any, claim) => {
      const date = new Date(claim.submitted_at);
      const monthKey = date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
      
      if (!acc[monthKey]) {
        acc[monthKey] = { month: monthKey, total: 0, fraud: 0, verified: 0 };
      }
      
      acc[monthKey].total++;
      if (claim.status === 'Fraud') acc[monthKey].fraud++;
      if (claim.status === 'Verified') acc[monthKey].verified++;
      
      return acc;
    }, {});

    return Object.values(monthlyData).slice(-6); // Last 6 months
  };

  const getConfidenceDistributionData = () => {
    const ranges = [
      { range: '0-25%', min: 0, max: 25, color: '#10b981' },
      { range: '26-50%', min: 25, max: 50, color: '#84cc16' },
      { range: '51-75%', min: 50, max: 75, color: '#f59e0b' },
      { range: '76-100%', min: 75, max: 100, color: '#ef4444' },
    ];

    return ranges.map(range => ({
      name: range.range,
      value: claims.filter(c => {
        const confidence = parseFloat(c.confidence_score || 0);
        return confidence >= range.min && confidence <= range.max;
      }).length,
      color: range.color,
    })).filter(item => item.value > 0);
  };

  const formatCurrency = (amount: number) => 
    new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);

  const getRiskBadgeColor = (riskLevel: string) => {
    switch(riskLevel) {
      case "HIGH": return "bg-red-600 text-white";
      case "MEDIUM": return "bg-yellow-600 text-white";
      case "LOW": return "bg-green-600 text-white";
      case "CRITICAL": return "bg-red-900 text-white";
      default: return "bg-gray-600 text-white";
    }
  };

  const getStatusBadgeColor = (status: string) => {
    switch(status) {
      case "Verified": return "bg-green-600 text-white";
      case "Fraud": return "bg-red-600 text-white";
      case "Rejected": return "bg-red-800 text-white";
      case "Pending": return "bg-yellow-600 text-white";
      default: return "bg-gray-600 text-white";
    }
  };

  return (
    <>
      <Sidebar className="border-r bg-white w-64">
        <SidebarContent className="flex flex-col h-full">
          <SidebarGroup className="flex-1">
            <SidebarGroupLabel className="text-xl font-bold px-6 py-5 text-black border-b">
              SmartClaim Admin
            </SidebarGroupLabel>
            <SidebarGroupContent className="mt-4">
              <SidebarMenu>
                {menuItems.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <button 
                        onClick={() => setActiveView(item.view)}
                        className={`flex items-center gap-3 px-6 py-3 w-full text-left transition-colors rounded-lg mx-2 ${
                          activeView === item.view 
                            ? 'bg-gray-900 text-white' 
                            : 'text-gray-700 hover:bg-gray-100'
                        }`}
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
            <h1 className="text-xl font-semibold text-gray-900">
              {activeView === "dashboard" ? "Dashboard" : 
               activeView === "pending" ? "Pending Review (Pending + Fraud)" :
               `${activeView.charAt(0).toUpperCase() + activeView.slice(1)} Claims`}
            </h1>
          </div>
          
          {activeView !== "dashboard" && (
            <div className="hidden md:flex items-center gap-2 flex-1 max-w-xl mx-4">
              <Input
                type="text"
                placeholder="Search claims..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                className="flex-1"
              />
              <Select value={riskFilter} onValueChange={setRiskFilter}>
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="Risk" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                  <SelectItem value="critical">Critical</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={handleSearch} size="sm">
                <Search className="h-4 w-4" />
              </Button>
            </div>
          )}
        </header>

        {activeView !== "dashboard" && (
          <div className="md:hidden bg-white border-b px-4 py-3">
            <div className="flex gap-2">
              <Input
                type="text"
                placeholder="Search claims..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                className="flex-1"
              />
              <Select value={riskFilter} onValueChange={setRiskFilter}>
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Med</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={handleSearch} size="sm">
                <Search className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}

        <main className="flex-1 overflow-y-auto bg-gray-50">
          {error && (
            <div className="mx-4 mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
              <AlertTriangle className="h-5 w-5 text-red-600 flex-shrink-0" />
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Loader2 className="animate-spin h-12 w-12 text-gray-900 mx-auto mb-4" />
                <p className="text-gray-500">Loading claims data...</p>
              </div>
            </div>
          ) : (
            <div className="h-full">
              {activeView === "dashboard" && (
                <div className="p-4 lg:p-6 space-y-6">
                  {/* Enhanced Statistics Cards */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-white p-6 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-gray-600">Total Claims</p>
                        <TrendingUp className="h-5 w-5 text-blue-500" />
                      </div>
                      <p className="text-3xl font-bold text-gray-900">{stats.total}</p>
                      <p className="text-xs text-gray-500 mt-1">All time</p>
                    </div>

                    <div className="bg-white p-6 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-gray-600">Verified</p>
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      </div>
                      <p className="text-3xl font-bold text-green-600">{stats.verified}</p>
                      <p className="text-xs text-gray-500 mt-1">{stats.total > 0 ? ((stats.verified / stats.total) * 100).toFixed(1) : 0}% approved</p>
                    </div>

                    <div className="bg-white p-6 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-gray-600">Fraud Detected</p>
                        <XCircle className="h-5 w-5 text-red-600" />
                      </div>
                      <p className="text-3xl font-bold text-red-600">{stats.fraud}</p>
                      <p className="text-xs text-gray-500 mt-1">{stats.total > 0 ? ((stats.fraud / stats.total) * 100).toFixed(1) : 0}% fraud rate</p>
                    </div>

                    <div className="bg-white p-6 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-gray-600">Pending Review</p>
                        <Clock className="h-5 w-5 text-yellow-600" />
                      </div>
                      <p className="text-3xl font-bold text-yellow-600">{stats.pending}</p>
                      <p className="text-xs text-gray-500 mt-1">Need attention</p>
                    </div>
                  </div>

                  {/* Charts Section */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Status Distribution Pie Chart */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <PieChart className="h-5 w-5" />
                          Claims Status Distribution
                        </CardTitle>
                        <CardDescription>Overview of claim statuses in the system</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ChartContainer config={{}} className="h-80">
                          <ResponsiveContainer width="100%" height="100%">
                            <RechartsPieChart>
                              <Pie
                                data={getStatusDistributionData()}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                              >
                                {getStatusDistributionData().map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Pie>
                              <ChartTooltip content={<ChartTooltipContent />} />
                            </RechartsPieChart>
                          </ResponsiveContainer>
                        </ChartContainer>
                      </CardContent>
                    </Card>

                    {/* Risk Level Distribution Bar Chart */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5" />
                          Risk Level Distribution
                        </CardTitle>
                        <CardDescription>Claims categorized by risk level</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ChartContainer config={{}} className="h-80">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={getRiskDistributionData()}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis />
                              <ChartTooltip content={<ChartTooltipContent />} />
                              <Bar dataKey="value" fill="#8884d8">
                                {getRiskDistributionData().map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </ChartContainer>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Monthly Trend and Confidence Distribution */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Monthly Claims Trend */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Activity className="h-5 w-5" />
                          Monthly Claims Trend
                        </CardTitle>
                        <CardDescription>Claims submitted over the last 6 months</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ChartContainer config={{}} className="h-80">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={getMonthlyTrendData()}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="month" />
                              <YAxis />
                              <ChartTooltip content={<ChartTooltipContent />} />
                              <Legend />
                              <Line type="monotone" dataKey="total" stroke="#3b82f6" strokeWidth={2} name="Total Claims" />
                              <Line type="monotone" dataKey="fraud" stroke="#ef4444" strokeWidth={2} name="Fraud" />
                              <Line type="monotone" dataKey="verified" stroke="#10b981" strokeWidth={2} name="Verified" />
                            </LineChart>
                          </ResponsiveContainer>
                        </ChartContainer>
                      </CardContent>
                    </Card>

                    {/* Confidence Score Distribution */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5" />
                          Confidence Score Distribution
                        </CardTitle>
                        <CardDescription>Fraud detection confidence ranges</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ChartContainer config={{}} className="h-80">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={getConfidenceDistributionData()}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis />
                              <ChartTooltip content={<ChartTooltipContent />} />
                              <Bar dataKey="value" fill="#8884d8">
                                {getConfidenceDistributionData().map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </ChartContainer>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Additional Statistics Row */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-white p-6 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-gray-600">High Risk</p>
                        <AlertTriangle className="h-5 w-5 text-orange-600" />
                      </div>
                      <p className="text-3xl font-bold text-orange-600">{stats.highRisk}</p>
                      <p className="text-xs text-gray-500 mt-1">Requires review</p>
                    </div>

                    <div className="bg-white p-6 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-gray-600">Rejected</p>
                        <XCircle className="h-5 w-5 text-gray-600" />
                      </div>
                      <p className="text-3xl font-bold text-gray-900">{stats.rejected}</p>
                      <p className="text-xs text-gray-500 mt-1">Not approved</p>
                    </div>

                    <div className="bg-white p-6 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-gray-600">Avg Confidence</p>
                        <Activity className="h-5 w-5 text-purple-600" />
                      </div>
                      <p className="text-3xl font-bold text-purple-600">{stats.avgConfidence.toFixed(1)}%</p>
                      <p className="text-xs text-gray-500 mt-1">Detection accuracy</p>
                    </div>

                    <div className="bg-white p-6 rounded-lg border shadow-sm hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-sm text-gray-600">Avg Amount</p>
                        <DollarSign className="h-5 w-5 text-green-600" />
                      </div>
                      <p className="text-3xl font-bold text-green-600">{formatCurrency(stats.avgAmount)}</p>
                      <p className="text-xs text-gray-500 mt-1">Per claim</p>
                    </div>
                  </div>
                </div>
              )}

              {activeView !== "dashboard" && (
                <div className="p-4 lg:p-6">
                  {getFilteredClaims().length > 0 ? (
                    <div className="bg-white rounded-lg border shadow-sm overflow-hidden">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead className="bg-gray-50 border-b">
                            <tr>
                              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-900 uppercase">Claim #</th>
                              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-900 uppercase">Policyholder</th>
                              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-900 uppercase">Amount</th>
                              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-900 uppercase hidden md:table-cell">Confidence</th>
                              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-900 uppercase">Risk</th>
                              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-900 uppercase">Status</th>
                              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-900 uppercase hidden lg:table-cell">Date</th>
                              <th className="px-4 py-3 text-left text-xs font-semibold text-gray-900 uppercase">Actions</th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {getFilteredClaims().map((claim) => (
                              <tr key={claim.id} className="hover:bg-gray-50 transition-colors">
                                <td className="px-4 py-3 text-sm font-medium text-gray-900">
                                  {claim.claim_number || `#${claim.id}`}
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-700">
                                  <div>
                                    <p className="font-medium">{claim.policyholder?.name || claim.policyholder?.username}</p>
                                    <p className="text-xs text-gray-500 truncate max-w-[150px]">{claim.policyholder?.email}</p>
                                  </div>
                                </td>
                                <td className="px-4 py-3 text-sm font-semibold text-gray-900 whitespace-nowrap">
                                  ₹{claim.claim_amount?.toLocaleString()}
                                </td>
                                <td className="px-4 py-3 text-sm hidden md:table-cell">
                                  <div className="flex items-center gap-2">
                                    <div className="w-12 bg-gray-200 rounded-full h-2">
                                      <div 
                                        className={`h-2 rounded-full ${
                                          claim.confidence_score >= 70 ? 'bg-red-600' :
                                          claim.confidence_score >= 40 ? 'bg-yellow-600' :
                                          'bg-green-600'
                                        }`}
                                        style={{ width: `${claim.confidence_score}%` }}
                                      />
                                    </div>
                                    <span className="font-medium text-xs">{claim.confidence_score?.toFixed(0)}%</span>
                                  </div>
                                </td>
                                <td className="px-4 py-3 text-sm">
                                  <Badge className={`${getRiskBadgeColor(claim.risk_level)} text-xs`}>
                                    {claim.risk_level}
                                  </Badge>
                                </td>
                                <td className="px-4 py-3 text-sm">
                                  <Badge className={`${getStatusBadgeColor(claim.status)} text-xs`}>
                                    {claim.status || 'Pending'}
                                  </Badge>
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-700 hidden lg:table-cell whitespace-nowrap">
                                  {new Date(claim.submitted_at).toLocaleDateString()}
                                </td>
                                <td className="px-4 py-3 text-sm">
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => handleViewClaim(claim.id)}
                                    className="flex items-center gap-1 text-xs"
                                  >
                                    <Eye className="h-3 w-3" />
                                    <span className="hidden sm:inline">View</span>
                                  </Button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <div className="bg-white rounded-lg border shadow-sm p-12 text-center">
                      <AlertTriangle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-600">
                        No {activeView === "all" ? "" : activeView} claims found
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </main>
      </div>
    </>
  );
}