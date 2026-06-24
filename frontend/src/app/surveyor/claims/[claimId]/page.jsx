"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { ArrowLeft, AlertTriangle, CheckCircle, Clock, MessageCircle, Calendar } from "lucide-react";
import SurveyorChatAppointment from "@/components/SurveyorChatAppointment";

const API_BASE = `/api/detection`;

export default function ClaimDetail() {
  const router = useRouter();
  const params = useParams();
  const claimId = params.claimId;
  const [claim, setClaim] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (!token) {
      router.push("/login");
      return;
    }
    fetchClaimDetail();
  }, [claimId]);

  const fetchClaimDetail = async () => {
    try {
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${API_BASE}/claims/${claimId}/`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setClaim(data);
      }
    } catch (err) {
      console.error("Error fetching claim:", err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <p className="text-gray-600">Loading claim details...</p>
      </div>
    );
  }

  if (!claim) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <Button variant="outline" onClick={() => router.back()}>
          <ArrowLeft className="h-4 w-4 mr-2" /> Back
        </Button>
        <div className="mt-4 text-center text-red-600">Claim not found</div>
      </div>
    );
  }

  const getRiskColor = (risk) => {
    switch (risk) {
      case "HIGH": return "destructive";
      case "MEDIUM": return "secondary";
      case "LOW": return "default";
      case "CRITICAL": return "destructive";
      default: return "secondary";
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "Under Survey": return <Clock className="h-4 w-4 text-yellow-600" />;
      case "Survey Completed": return <CheckCircle className="h-4 w-4 text-green-600" />;
      case "Fraud": return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b px-6 py-4 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" onClick={() => router.back()}>
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Claim Details</h1>
              <p className="text-sm text-gray-500">{claim.claim_number}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {getStatusIcon(claim.status)}
            <Badge variant={getRiskColor(claim.risk_level)}>
              {claim.status}
            </Badge>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto p-6 space-y-6">
        {/* Alert Section */}
        {claim.fraud_detected && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              ⚠️ This claim has been flagged as high-risk fraud. Confidence: {claim.confidence_score}%
            </AlertDescription>
          </Alert>
        )}

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600">Claim Amount</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-blue-600">
                ₹{claim.claim_amount?.toLocaleString()}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600">Risk Level</CardTitle>
            </CardHeader>
            <CardContent>
              <Badge variant={getRiskColor(claim.risk_level)} className="text-lg px-3 py-1">
                {claim.risk_level}
              </Badge>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600">Fraud Detected</CardTitle>
            </CardHeader>
            <CardContent>
              <p className={`text-lg font-bold ${claim.fraud_detected ? 'text-red-600' : 'text-green-600'}`}>
                {claim.fraud_detected ? '🚨 Yes' : '✅ No'}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Tabs */}
        <Tabs defaultValue="claim" className="bg-white rounded-lg border">
          <TabsList className="w-full justify-start border-b px-6 py-0">
            <TabsTrigger value="claim">Claim Information</TabsTrigger>
            <TabsTrigger value="policyholder">Policyholder</TabsTrigger>
            <TabsTrigger value="communication">Communication</TabsTrigger>
            <TabsTrigger value="survey">Survey Report</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="claim" className="p-6 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Claim Details</CardTitle>
              </CardHeader>
              <CardContent className="grid md:grid-cols-2 gap-6">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Claim Number</p>
                  <p className="font-semibold">{claim.claim_number}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">Status</p>
                  <Badge>{claim.status}</Badge>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">Accident Date</p>
                  <p className="font-semibold">{new Date(claim.accident_date).toLocaleDateString()}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">Submitted Date</p>
                  <p className="font-semibold">{new Date(claim.submitted_at).toLocaleDateString()}</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Claim Description</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700">{claim.claim_description}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Vehicle Information</CardTitle>
              </CardHeader>
              <CardContent className="grid md:grid-cols-2 gap-6">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Vehicle Make</p>
                  <p className="font-semibold">{claim.policyholder?.vehicle_make}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">Vehicle Model</p>
                  <p className="font-semibold">{claim.policyholder?.vehicle_model}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">DL Number</p>
                  <p className="font-semibold">{claim.dl_number}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">Registration</p>
                  <p className="font-semibold">{claim.vehicle_reg_no}</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="policyholder" className="p-6">
            <Card>
              <CardHeader>
                <CardTitle>Policyholder Details</CardTitle>
              </CardHeader>
              <CardContent className="grid md:grid-cols-2 gap-6">
                <div>
                  <p className="text-sm text-gray-600 mb-1">Name</p>
                  <p className="font-semibold">{claim.policyholder?.username}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">Email</p>
                  <p className="font-semibold">{claim.policyholder?.email}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">Vehicle Make</p>
                  <p className="font-semibold">{claim.policyholder?.vehicle_make}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 mb-1">Vehicle Age</p>
                  <p className="font-semibold">{claim.policyholder?.age_of_vehicle || 'N/A'}</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="communication" className="p-6">
            <SurveyorChatAppointment claim={claim} />
          </TabsContent>

          <TabsContent value="survey" className="p-6">
            <Card>
              <CardHeader>
                <CardTitle>Survey Report</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {claim.survey_completed_at ? (
                  <>
                    <div>
                      <p className="text-sm text-green-600 font-medium mb-2">✅ Survey Completed</p>
                      <p className="text-xs text-gray-500">{new Date(claim.survey_completed_at).toLocaleString()}</p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Surveyor Recommendation</p>
                        <Badge>{claim.surveyor_recommendation}</Badge>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Damage Verified</p>
                        <Badge variant={claim.damage_verified ? "default" : "secondary"}>
                          {claim.damage_verified ? "Yes" : "No"}
                        </Badge>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600 mb-1">AI Estimated Amount</p>
                        <p className="font-semibold">₹{claim.surveyor_assessed_amount?.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600 mb-1">Initial Claim Amount</p>
                        <p className="font-semibold">₹{claim.claim_amount?.toLocaleString()}</p>
                      </div>
                    </div>

                    <div>
                      <p className="text-sm text-gray-600 mb-2">Notes</p>
                      <p className="text-gray-700 bg-gray-50 p-3 rounded">{claim.surveyor_notes}</p>
                    </div>
                  </>
                ) : (
                  <p className="text-gray-500">Survey report not yet submitted</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analysis" className="p-6">
            <Card>
              <CardHeader>
                <CardTitle>AI Analysis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-sm text-gray-600 mb-1">Fraud Probability (Tabular)</p>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-red-600 h-2 rounded-full"
                        style={{ width: `${claim.tabular_fraud_probability || 0}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{claim.tabular_fraud_probability}%</p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-1">Fraud Probability (Images)</p>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-orange-600 h-2 rounded-full"
                        style={{ width: `${claim.image_fraud_probability || 0}%` }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{claim.image_fraud_probability}%</p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-1">Fusion Score</p>
                    <p className="text-2xl font-bold text-blue-600">{claim.fusion_score}%</p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-1">Damage Severity</p>
                    <Badge variant="secondary">{claim.overall_damage_severity}</Badge>
                  </div>
                </div>

                <div>
                  <p className="text-sm text-gray-600 mb-2">Average Damage Percentage</p>
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className="bg-green-600 h-3 rounded-full"
                      style={{ width: `${claim.average_damage_percentage || 0}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">{claim.average_damage_percentage}%</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <Button variant="outline" onClick={() => router.back()} className="w-full">
          <ArrowLeft className="h-4 w-4 mr-2" /> Back to Dashboard
        </Button>
      </main>
    </div>
  );
}
