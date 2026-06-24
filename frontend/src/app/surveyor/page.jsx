"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue
} from "@/components/ui/select";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow
} from "@/components/ui/table";
import {
  ClipboardList, CheckCircle, Clock, LogOut,
  AlertTriangle, Eye, FileText, Camera, Loader2
} from "lucide-react";

const API_BASE = `/api/detection`;

export default function SurveyorDashboard() {
  const router = useRouter();
  const [claims, setClaims] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [showReportModal, setShowReportModal] = useState(false);
  const [surveyorName, setSurveyorName] = useState("Surveyor");
  const [stats, setStats] = useState({ assigned: 0, completed: 0, pending: 0 });

  useEffect(() => {
    const role = localStorage.getItem("user_role");
    const username = localStorage.getItem("username");
    const token = localStorage.getItem("access_token");

    // Check if token exists and role is surveyor (case-insensitive)
    if (!token || !role || role.toLowerCase() !== "surveyor") {
      router.push("/login");
      return;
    }

    setSurveyorName(username || "Surveyor");
    fetchAssignedClaims();
  }, []);

  const fetchAssignedClaims = async () => {
    try {
      const token = localStorage.getItem("access_token");
      const res = await fetch(`${API_BASE}/surveyor/claims/`, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (res.ok) {
        const data = await res.json();
        setClaims(data.claims || []);
        setStats({
          assigned: data.claims?.length || 0,
          pending: data.claims?.filter(c => c.status === "Pending" || c.status === "Under Survey").length || 0,
          completed: data.claims?.filter(c => c.status === "Survey Completed").length || 0,
        });
      }
    } catch (err) {
      console.error("Failed to fetch claims:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    router.push("/login");
  };

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
      case "Under Survey": return <Clock className="h-4 w-4 text-yellow-500" />;
      case "Survey Completed": return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "Fraud": return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default: return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-14 items-center justify-between">
            <div className="flex items-center gap-2">
              <ClipboardList className="h-6 w-6 text-blue-600" />
              <span className="font-semibold text-lg">Surveyor Dashboard</span>
              <span className="text-sm text-gray-500 hidden sm:inline">— {surveyorName}</span>
            </div>
            <Button variant="outline" size="sm" onClick={handleLogout} className="gap-2">
              <LogOut className="h-4 w-4" />
              Logout
            </Button>
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="border-l-4 border-l-blue-500">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <ClipboardList className="h-4 w-4 text-blue-600" />
                Assigned Claims
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-blue-600">{stats.assigned}</p>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-yellow-500">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <Clock className="h-4 w-4 text-yellow-600" />
                Pending Survey
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-yellow-600">{stats.pending}</p>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-green-500">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-600" />
                Surveys Completed
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-3xl font-bold text-green-600">{stats.completed}</p>
            </CardContent>
          </Card>
        </div>

        {/* Claims Table */}
        <Card>
          <CardHeader>
            <CardTitle>My Assigned Claims</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-gray-400 mr-2" />
                <p className="text-gray-500">Loading claims...</p>
              </div>
            ) : claims.length === 0 ? (
              <div className="text-center py-12">
                <ClipboardList className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                <p className="text-gray-500">No claims assigned to you yet.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Claim #</TableHead>
                      <TableHead>Policyholder</TableHead>
                      <TableHead>Amount</TableHead>
                      <TableHead>Risk</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Date</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {claims.map((claim) => (
                      <TableRow key={claim.id} className="hover:bg-gray-50">
                        <TableCell className="font-medium">{claim.claim_number}</TableCell>
                        <TableCell>
                          <div>
                            <p className="font-medium text-gray-900">{claim.policyholder?.username}</p>
                            <p className="text-xs text-gray-500">{claim.policyholder?.email}</p>
                          </div>
                        </TableCell>
                        <TableCell className="font-semibold text-gray-900">
                          ₹{claim.claim_amount?.toLocaleString()}
                        </TableCell>
                        <TableCell>
                          <Badge variant={getRiskColor(claim.risk_level)}>
                            {claim.risk_level}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            {getStatusIcon(claim.status)}
                            <span className="text-sm">{claim.status}</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-sm text-gray-500">
                          {new Date(claim.submitted_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex gap-2 justify-end">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => router.push(`/surveyor/claims/${claim.id}`)}
                            >
                              <Eye className="h-3 w-3 mr-1" /> View
                            </Button>
                            {claim.status !== "Survey Completed" && (
                              <Button
                                size="sm"
                                className="bg-blue-600 hover:bg-blue-700 text-white"
                                onClick={() => {
                                  setSelectedClaim(claim);
                                  setShowReportModal(true);
                                }}
                              >
                                <FileText className="h-3 w-3 mr-1" /> Report
                              </Button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>
      </main>

      {/* Survey Report Modal */}
      {showReportModal && selectedClaim && (
        <SurveyReportModal
          claim={selectedClaim}
          onClose={() => {
            setShowReportModal(false);
            setSelectedClaim(null);
          }}
          onSubmit={fetchAssignedClaims}
        />
      )}
    </div>
  );
}


function SurveyReportModal({ claim, onClose, onSubmit }) {
  const [notes, setNotes] = useState("");
  const [actualAmount, setActualAmount] = useState("");
  const [recommendation, setRecommendation] = useState("APPROVE");
  const [damageVerified, setDamageVerified] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [fieldPhotos, setFieldPhotos] = useState([]);
  const [photoPreviews, setPhotoPreviews] = useState([]);
  const [uploadError, setUploadError] = useState("");

  const handlePhotoChange = (e) => {
    const files = Array.from(e.target.files);
    setUploadError("");

    // Validate files
    for (const file of files) {
      if (file.size > 10 * 1024 * 1024) {
        setUploadError(`"${file.name}" exceeds the 10 MB limit.`);
        return;
      }
      if (!file.type.startsWith("image/")) {
        setUploadError(`"${file.name}" is not a valid image file.`);
        return;
      }
    }

    if (fieldPhotos.length + files.length > 10) {
      setUploadError("Maximum 10 field photos allowed.");
      return;
    }

    setFieldPhotos((prev) => [...prev, ...files]);

    // Generate previews
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (ev) =>
        setPhotoPreviews((prev) => [...prev, ev.target.result]);
      reader.readAsDataURL(file);
    });
  };

  const removePhoto = (index) => {
    setFieldPhotos((prev) => prev.filter((_, i) => i !== index));
    setPhotoPreviews((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      const token = localStorage.getItem("access_token");

      // Use FormData so we can send files + text together
      const formData = new FormData();
      formData.append("surveyor_notes", notes);
      formData.append("surveyor_recommendation", recommendation);
      formData.append("damage_verified", damageVerified);
      if (actualAmount) {
        formData.append("surveyor_assessed_amount", parseFloat(actualAmount));
      }
      // Attach each photo under the key "field_photos"
      fieldPhotos.forEach((photo) => formData.append("field_photos", photo));

      const res = await fetch(
        `/api/detection/surveyor/claims/${claim.id}/report/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            // ⚠️ Do NOT set Content-Type here — browser will set multipart boundary automatically
          },
          body: formData,
        }
      );

      if (res.ok) {
        const data = await res.json();
        const photoMsg =
          data.field_photos_saved > 0
            ? ` ${data.field_photos_saved} photo(s) saved.`
            : "";
        alert(`✅ Survey report submitted successfully!${photoMsg}`);
        onSubmit();
        onClose();
      } else {
        const data = await res.json();
        alert(`❌ ${data.error || "Failed to submit report"}`);
      }
    } catch (err) {
      console.error(err);
      alert("❌ An error occurred while submitting the report");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Submit Survey Report</DialogTitle>
          <DialogDescription>
            Claim {claim.claim_number} — {claim.policyholder?.username}
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-6 py-4">
          {/* Damage Verified */}
          <div>
            <label className="block text-sm font-semibold mb-3">
              Damage Verified on Site?
            </label>
            <div className="flex gap-4">
              <label
                className="flex items-center gap-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50"
                style={{ borderColor: damageVerified ? "#2563eb" : "#d1d5db" }}
              >
                <input
                  type="radio"
                  checked={damageVerified}
                  onChange={() => setDamageVerified(true)}
                  className="w-4 h-4"
                />
                <span className="text-green-700 font-medium">
                  ✅ Yes — Damage Confirmed
                </span>
              </label>
              <label
                className="flex items-center gap-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50"
                style={{ borderColor: !damageVerified ? "#2563eb" : "#d1d5db" }}
              >
                <input
                  type="radio"
                  checked={!damageVerified}
                  onChange={() => setDamageVerified(false)}
                  className="w-4 h-4"
                />
                <span className="text-red-700 font-medium">
                  ❌ No — Damage Not Found
                </span>
              </label>
            </div>
          </div>

          {/* Recommendation */}
          <div>
            <label className="block text-sm font-semibold mb-2">
              Surveyor Recommendation *
            </label>
            <Select value={recommendation} onValueChange={setRecommendation}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="APPROVE">✅ Approve Claim</SelectItem>
                <SelectItem value="REJECT">❌ Reject Claim</SelectItem>
                <SelectItem value="INVESTIGATE">
                  🔍 Further Investigation Needed
                </SelectItem>
                <SelectItem value="PARTIAL">⚡ Partial Approval</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Assessed Amount */}
          <div>
            <label className="block text-sm font-semibold mb-2">
              Surveyor Assessed Amount (₹)
            </label>
            <input
              type="number"
              value={actualAmount}
              onChange={(e) => setActualAmount(e.target.value)}
              placeholder={`AI estimated: ₹${claim.claim_amount?.toLocaleString()}`}
              className="w-full border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 outline-none"
            />
            <p className="text-xs text-gray-500 mt-1">
              Leave blank to keep original amount
            </p>
          </div>

          {/* Notes */}
          <div>
            <label className="block text-sm font-semibold mb-2">
              Field Survey Notes *
            </label>
            <Textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={4}
              required
              placeholder="Describe findings from field visit: actual damage observed, discrepancies with claim, vehicle condition, etc."
              className="w-full rounded-lg border p-3 focus:ring-2 focus:ring-blue-500 outline-none"
            />
          </div>

          {/* Field Photos Upload */}
          <div>
            <label className="block text-sm font-semibold mb-2 flex items-center gap-2">
              <Camera className="h-4 w-4 text-blue-600" />
              Field Inspection Photos
              <span className="text-xs font-normal text-gray-500">
                (optional, max 10 images, 10 MB each)
              </span>
            </label>

            {/* Upload Button */}
            <label className="flex flex-col items-center justify-center w-full h-28 border-2 border-dashed border-gray-300 rounded-lg cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors">
              <div className="flex flex-col items-center gap-1">
                <Camera className="h-6 w-6 text-gray-400" />
                <span className="text-sm text-gray-500">
                  Click to upload field photos
                </span>
                <span className="text-xs text-gray-400">
                  JPG, PNG, WEBP supported
                </span>
              </div>
              <input
                id="field-photos-input"
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={handlePhotoChange}
              />
            </label>

            {uploadError && (
              <p className="text-sm text-red-600 mt-2">{uploadError}</p>
            )}

            {/* Photo Previews */}
            {photoPreviews.length > 0 && (
              <div className="mt-3 grid grid-cols-3 sm:grid-cols-4 gap-3">
                {photoPreviews.map((preview, idx) => (
                  <div key={idx} className="relative group rounded-lg overflow-hidden border bg-gray-50">
                    <img
                      src={preview}
                      alt={`Field photo ${idx + 1}`}
                      className="w-full h-20 object-cover"
                    />
                    <button
                      type="button"
                      onClick={() => removePhoto(idx)}
                      className="absolute top-1 right-1 bg-red-600 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      ×
                    </button>
                    <div className="absolute bottom-0 left-0 right-0 bg-black/40 text-white text-[10px] text-center py-0.5">
                      Photo {idx + 1}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {fieldPhotos.length > 0 && (
              <p className="text-xs text-blue-600 mt-2">
                {fieldPhotos.length} photo{fieldPhotos.length !== 1 ? "s" : ""} ready to upload
              </p>
            )}
          </div>

          {/* Buttons */}
          <div className="flex gap-3 pt-2">
            <Button
              type="submit"
              disabled={submitting}
              className="flex-1 bg-blue-600 hover:bg-blue-700"
            >
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Submitting...
                </>
              ) : (
                <>
                  <FileText className="h-4 w-4 mr-2" />
                  Submit Report
                </>
              )}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              className="flex-1"
            >
              Cancel
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
