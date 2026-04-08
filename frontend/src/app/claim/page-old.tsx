"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";

const ToastNotification = ({ message, type, claimNumber, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const icons = {
    success: (
      <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    error: (
      <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    )
  };

  const colors = {
    success: 'bg-white border-l-4 border-green-500 shadow-lg',
    error: 'bg-white border-l-4 border-red-500 shadow-lg'
  };

  return (
    <div className={`fixed top-4 right-4 z-50 max-w-sm w-full animate-slide-in ${colors[type]} rounded-lg p-4`}>
      <div className="flex items-start">
        <div className="flex-shrink-0">{icons[type]}</div>
        <div className="ml-3 flex-1">
          <p className="text-sm font-semibold text-gray-900">
            {type === 'success' ? '✅ Claim Saved' : '❌ Save Failed'}
          </p>
          <p className="mt-1 text-sm text-gray-600">{message}</p>
          {claimNumber && (
            <p className="mt-2 text-xs font-mono text-gray-500 bg-gray-50 px-2 py-1 rounded">
              {claimNumber}
            </p>
          )}
        </div>
        <button onClick={onClose} className="ml-4 flex-shrink-0 text-gray-400 hover:text-gray-600">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
          </svg>
        </button>
      </div>
      <style jsx>{`
        @keyframes slide-in {
          from { transform: translateX(100%); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        .animate-slide-in { animation: slide-in 0.3s ease-out; }
      `}</style>
    </div>
  );
};


export default function ClaimPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const username = searchParams.get("username") || "";

  // State initialization
  const [formData, setFormData] = useState({
    username: username,
    claim_description: "",
    accident_date: "",
    vehicle_make: "",
    vehicle_model: "",
    dl_number: "",
    vehicle_reg_no: "",
    fir_number: "",
  });
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<string>("results");
  const [selectedImageIndex, setSelectedImageIndex] = useState<number>(0);
  const [toast, setToast] = useState(null);

  // Handle form field changes
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  // Handle select dropdown changes (for vehicle_make)
  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  // Handle image uploads
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files);

      // Validate total number of images
      if (files.length > 10) {
        setError("Maximum 10 images allowed");
        return;
      }

      // Validate each file
      for (const file of files) {
        if (file.size > 10 * 1024 * 1024) {
          setError(`Image ${file.name} exceeds 10MB size limit`);
          return;
        }

        if (!file.type.startsWith('image/')) {
          setError(`${file.name} is not a valid image file`);
          return;
        }
      }

      setImageFiles(files);
      setError("");

      // Generate previews for all images
      const previewPromises = files.map(file => {
        return new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => {
            resolve(e.target?.result as string);
          };
          reader.readAsDataURL(file);
        });
      });

      Promise.all(previewPromises).then(previews => {
        setImagePreviews(previews);
      });
    }
  };

  // Pre-fill vehicle info from policyholder profile
  useEffect(() => {
    const fetchPolicyholderData = async () => {
      if (!username) return;

      try {
        const token = localStorage.getItem("access_token");
        const res = await fetch(`/api/policyholders/${username}/`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });

        if (res.ok) {
          const data = await res.json();

          // Pre-fill vehicle info if available
          setFormData(prev => ({
            ...prev,
            vehicle_make: data.vehicle_make || "",
            vehicle_model: data.vehicle_model || ""
          }));

          console.log("✅ Pre-filled vehicle info:", data.vehicle_make, data.vehicle_model);
        }
      } catch (error) {
        console.error("❌ Failed to fetch policyholder data:", error);
      }
    };

    fetchPolicyholderData();
  }, [username]);

  // Remove image from selection
  const removeImage = (indexToRemove: number) => {
    setImageFiles(prev => prev.filter((_, index) => index !== indexToRemove));
    setImagePreviews(prev => prev.filter((_, index) => index !== indexToRemove));

    // Adjust selected index if needed
    if (selectedImageIndex >= indexToRemove && selectedImageIndex > 0) {
      setSelectedImageIndex(selectedImageIndex - 1);
    }
  };

  // Form validation
  const validateForm = () => {
    if (!formData.username.trim()) {
      setError("Username is required");
      return false;
    }

    if (!formData.claim_description.trim()) {
      setError("Claim description is required");
      return false;
    }

    if (!formData.accident_date) {
      setError("Accident date is required");
      return false;
    }

    const accidentDate = new Date(formData.accident_date);
    const today = new Date();
    today.setHours(23, 59, 59, 999);

    if (accidentDate > today) {
      setError("Accident date cannot be in the future");
      return false;
    }

    const twoYearsAgo = new Date();
    twoYearsAgo.setFullYear(today.getFullYear() - 2);

    if (accidentDate < twoYearsAgo) {
      setError("Accident date cannot be more than 2 years ago");
      return false;
    }

    if (!formData.vehicle_make) {
      setError("Vehicle make is required");
      return false;
    }

    if (!formData.vehicle_model.trim()) {
      setError("Vehicle model is required");
      return false;
    }

    if (!formData.dl_number.trim()) {
      setError("Driving license number is required");
      return false;
    }

    if (!formData.vehicle_reg_no.trim()) {
      setError("Vehicle registration number is required");
      return false;
    }

    if (imageFiles.length === 0) {
      setError("Please upload at least one image of the damaged vehicle");
      return false;
    }

    return true;
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      const token = localStorage.getItem("access_token");
      if (!token) {
        setError("You must be logged in to submit a claim");
        setLoading(false);
        return;
      }

      const data = new FormData();
      data.append("username", formData.username.trim());
      data.append("claim_description", formData.claim_description.trim());
      data.append("accident_date", formData.accident_date);
      data.append("vehicle_make", formData.vehicle_make.trim());
      data.append("vehicle_model", formData.vehicle_model.trim());
      imageFiles.forEach((file) => { data.append("car_images", file); });
      data.append("dl_number", formData.dl_number.trim());
      data.append("vehicle_reg_no", formData.vehicle_reg_no.trim());
      data.append("fir_number", formData.fir_number.trim());

      console.log("🚀 Making request to:", `/api/detection/predict-claim/`);
      console.log("🔑 Token:", token ? "Present" : "Missing");
      console.log("🚗 Vehicle Info:", formData.vehicle_make, formData.vehicle_model);

      const res = await fetch(`/api/detection/predict-claim/`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: data,
      });

      // Debug: Log response details
      console.log("📊 Response Status:", res.status);
      console.log("📊 Response Status Text:", res.statusText);
      console.log("📊 Response Headers:", Object.fromEntries(res.headers.entries()));

      // Get response as text first
      const responseText = await res.text();
      console.log("📝 Raw Response (first 500 chars):", responseText.substring(0, 500));

      // Check if it's HTML (error page)
      if (responseText.trim().startsWith('<!DOCTYPE') || responseText.trim().startsWith('<html')) {
        console.error("❌ ERROR: Received HTML instead of JSON!");
        console.error("Full HTML response:", responseText);
        setError(`Server returned HTML error page. Status: ${res.status}. Check console for details.`);
        setLoading(false);
        return;
      }

      // Try to parse as JSON
      let resData;
      try {
        resData = JSON.parse(responseText);
      } catch (parseError) {
        console.error("❌ Failed to parse response as JSON:", parseError);
        console.error("Response was:", responseText);
        setError(`Invalid response format. Expected JSON but got: ${responseText.substring(0, 100)}...`);
        setLoading(false);
        return;
      }

      console.log("✅ Parsed Response Data:", resData);

      if (!res.ok) {
        setError(`Failed to process claim: ${resData.error || resData.detail || JSON.stringify(resData)}`);
        setLoading(false);
        return;
      }

      setResult(resData);
      setLoading(false);

      if (resData.claim_saved) {
        setToast({
          type: 'success',
          message: `Saved to database. Status: ${resData.claim_status}`,
          claimNumber: resData.claim_number
        });
      } else if (resData.save_error) {
        setToast({
          type: 'error',
          message: 'Analysis complete but database save failed',
          claimNumber: null
        });
      }

    } catch (err) {
      console.error("💥 Fetch Error:", err);
      console.error("Error name:", err.name);
      console.error("Error message:", err.message);
      setError(`Network error: ${err instanceof Error ? err.message : 'Something went wrong'}`);
      setLoading(false);
    }
  };

  // Reset form for new claim
  const resetForm = () => {
    setResult(null);
    setFormData({
      username: username,
      claim_description: "",
      accident_date: "",
      vehicle_make: "",
      vehicle_model: "",
      dl_number: "",
      vehicle_reg_no: "",
      fir_number: "",
    });
    setImageFiles([]);
    setImagePreviews([]);
    setActiveTab("results");
    setSelectedImageIndex(0);
  };

  const formatCurrency = (amount: string | number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(Number(amount));
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'HIGH': return 'text-red-600 bg-red-50 border-red-200';
      case 'MEDIUM': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'LOW': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const CalculationCard = ({ title, children, className = "" }: { title: string, children: React.ReactNode, className?: string }) => (
    <div className={`bg-white border rounded-lg shadow-sm ${className}`}>
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
        <h4 className="font-semibold text-gray-900">{title}</h4>
      </div>
      <div className="p-4">
        {children}
      </div>
    </div>
  );

  const TabButton = ({ tabKey, label, isActive, onClick }: { tabKey: string, label: string, isActive: boolean, onClick: (tab: string) => void }) => (
    <button
      onClick={() => onClick(tabKey)}
      className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${isActive
        ? 'bg-blue-100 text-blue-700 border border-blue-200'
        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
        }`}
    >
      {label}
    </button>
  );

  return (
    <div className="max-w-7xl mx-auto p-8 mt-10">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Insurance Claim Submission</h1>
              <p className="text-blue-100 mt-2">Submit your claim with AI-powered fraud detection analysis</p>
            </div>
            <div className="flex items-center gap-3">
              <a
                href="/customer"
                className="flex items-center gap-2 bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
                My Account
              </a>
            </div>
          </div>
        </div>

        <div className="p-6">

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-800">{error}</p>
                </div>
              </div>
            </div>
          )}

          {result && (
            <div className="mb-6">
              {/* ========================================== */}
              {/* NEW: YOLO CALCULATED CLAIM AMOUNT */}
              {/* ========================================== */}
              {result.calculated_claim_amount && (
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-300 rounded-lg shadow-lg mb-6 overflow-hidden">
                  <div className="bg-gradient-to-r from-green-600 to-emerald-600 text-white px-6 py-4">
                    <h3 className="text-xl font-bold flex items-center">
                      <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      AI-Calculated Claim Amount
                    </h3>
                    <p className="text-green-100 text-sm mt-1">Based on YOLO parts detection + CNN damage analysis</p>
                  </div>
                  <div className="p-6">
                    <div className="text-center mb-6">
                      <p className="text-5xl font-bold text-green-700">
                        {formatCurrency(result.calculated_claim_amount)}
                      </p>
                      <p className="text-sm text-gray-600 mt-2">Auto-calculated based on detected damages</p>
                    </div>

                    {result.claim_calculation_details && (
                      <div className="grid md:grid-cols-3 gap-4">
                        <div className="bg-white border border-blue-200 rounded-lg p-4 text-center">
                          <h5 className="text-sm font-medium text-blue-900">YOLO Base Amount</h5>
                          <p className="text-2xl font-bold text-blue-700 mt-1">
                            {formatCurrency(result.claim_calculation_details.yolo_base_amount)}
                          </p>
                          <p className="text-xs text-gray-600 mt-1">From damaged parts pricing</p>
                        </div>

                        <div className="bg-white border border-purple-200 rounded-lg p-4 text-center">
                          <h5 className="text-sm font-medium text-purple-900">CNN Damage Factor</h5>
                          <p className="text-2xl font-bold text-purple-700 mt-1">
                            {result.claim_calculation_details.cnn_damage_percentage?.toFixed(1) || 0}%
                          </p>
                          <p className="text-xs text-gray-600 mt-1">Multiplier: {result.claim_calculation_details.cnn_multiplier?.toFixed(2) || 1}</p>
                        </div>

                        <div className="bg-white border border-green-200 rounded-lg p-4 text-center">
                          <h5 className="text-sm font-medium text-green-900">Parts Damaged</h5>
                          <p className="text-2xl font-bold text-green-700 mt-1">
                            {result.claim_calculation_details.total_damaged_parts || 0}
                          </p>
                          <p className="text-xs text-gray-600 mt-1">Detected by AI</p>
                        </div>
                      </div>
                    )}

                    {result.claim_calculation_details?.calculation_formula && (
                      <div className="mt-4 p-3 bg-gray-50 rounded border text-sm font-mono text-center">
                        {result.claim_calculation_details.calculation_formula}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Results Summary */}
              <div className="bg-white border rounded-lg shadow-sm mb-6">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h3 className="text-lg font-semibold text-gray-900">Fraud Detection Results</h3>
                  <p className="text-sm text-gray-600 mt-1">
                    Analysis based on {result.total_images_submitted || 1} image(s)
                  </p>
                </div>

                <div className="p-6">
                  <div className="grid md:grid-cols-3 gap-6 mb-6">
                    {/* Fraud Detection Result */}
                    <div className={`p-4 rounded-lg border ${result.fraud_detected ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
                      <div className="flex items-center">
                        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${result.fraud_detected ? 'bg-red-100' : 'bg-green-100'}`}>
                          {result.fraud_detected ? (
                            <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 15.5c-.77.833.192 2.5 1.732 2.5z" />
                            </svg>
                          ) : (
                            <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                        <div className="ml-3">
                          <p className={`font-semibold ${result.fraud_detected ? 'text-red-800' : 'text-green-800'}`}>
                            {result.fraud_detected ? 'Fraud Detected' : 'No Fraud Detected'}
                          </p>
                          <p className={`text-sm ${result.fraud_detected ? 'text-red-600' : 'text-green-600'}`}>
                            Confidence: {(result.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Risk Level */}
                    <div className={`p-4 rounded-lg border ${getRiskColor(result.risk_level)}`}>
                      <div className="text-center">
                        <p className="font-semibold">Risk Level</p>
                        <p className="text-2xl font-bold mt-1">{result.risk_level}</p>
                      </div>
                    </div>

                    {/* Recommended Action */}
                    <div className="p-4 rounded-lg border border-gray-200 bg-gray-50">
                      <div className="text-center">
                        <p className="font-semibold text-gray-900">Action</p>
                        <p className="text-2xl font-bold mt-1 text-blue-600">
                          {result.recommended_action?.action || 'REVIEW'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Detailed Analysis Tabs */}
              <div className="bg-white border rounded-lg shadow-sm">
                <div className="px-6 py-4 border-b border-gray-200">
                  <div className="flex space-x-2 overflow-x-auto">
                    <TabButton tabKey="results" label="Summary" isActive={activeTab === "results"} onClick={setActiveTab} />
                    <TabButton tabKey="yolo" label="🚗 YOLO Detection" isActive={activeTab === "yolo"} onClick={setActiveTab} />
                    <TabButton tabKey="claim-breakdown" label="💰 Claim Breakdown" isActive={activeTab === "claim-breakdown"} onClick={setActiveTab} />
                    <TabButton tabKey="tabular" label="Tabular Analysis" isActive={activeTab === "tabular"} onClick={setActiveTab} />
                    <TabButton tabKey="fusion" label="Fusion Process" isActive={activeTab === "fusion"} onClick={setActiveTab} />
                    <TabButton tabKey="damage" label="Damage Detection" isActive={activeTab === "damage"} onClick={setActiveTab} />
                  </div>
                </div>

                <div className="p-6">
                  {/* ========================================== */}
                  {/* RESULTS/SUMMARY TAB - Default Tab */}
                  {/* ========================================== */}
                  {activeTab === "results" && (
                    <div className="space-y-6 animate-fade-in">
                      {/* Executive Summary Card */}
                      <div className="bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 rounded-2xl p-8 border-2 border-indigo-200 shadow-xl">


                        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                          {/* Claim Status */}
                          <div className="bg-white rounded-xl p-5 shadow-md border border-indigo-100">
                            <div className="flex items-center justify-between mb-2">
                              <p className="text-xs font-semibold text-gray-600 uppercase">Claim Status</p>
                              <div className={`w-3 h-3 rounded-full animate-pulse ${result.fraud_detected ? 'bg-red-500' : 'bg-green-500'
                                }`}></div>
                            </div>
                            <p className={`text-2xl font-black ${result.fraud_detected ? 'text-red-600' : 'text-green-600'
                              }`}>
                              {result.fraud_detected ? 'FLAGGED' : 'APPROVED'}
                            </p>
                          </div>

                          {/* Risk Level */}
                          <div className="bg-white rounded-xl p-5 shadow-md border border-indigo-100">
                            <p className="text-xs font-semibold text-gray-600 uppercase mb-2">Risk Level</p>
                            <p className={`text-2xl font-black ${result.risk_level === 'HIGH' ? 'text-red-600' :
                              result.risk_level === 'MEDIUM' ? 'text-orange-600' :
                                'text-green-600'
                              }`}>
                              {result.risk_level}
                            </p>
                          </div>

                          {/* Confidence Score */}
                          <div className="bg-white rounded-xl p-5 shadow-md border border-indigo-100">
                            <p className="text-xs font-semibold text-gray-600 uppercase mb-2">Confidence</p>
                            <p className="text-2xl font-black text-indigo-600">
                              {(result.confidence * 100).toFixed(1)}%
                            </p>
                          </div>

                          {/* Images Analyzed */}
                          <div className="bg-white rounded-xl p-5 shadow-md border border-indigo-100">
                            <p className="text-xs font-semibold text-gray-600 uppercase mb-2">Images</p>
                            <p className="text-2xl font-black text-purple-600">
                              {result.total_images_submitted}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Key Findings Grid */}
                      <div className="grid md:grid-cols-2 gap-6">
                        {/* Fraud Analysis */}
                        <CalculationCard title="🎯 Fraud Analysis" className="border-2 border-purple-200">
                          <div className="space-y-4">
                            <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-4">
                              <div className="flex justify-between items-center mb-3">
                                <span className="text-sm font-bold text-gray-700">Fraud Probability</span>
                                <span className="text-lg font-black text-purple-600">
                                  {(result.probabilities?.fraud * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                                <div
                                  className="bg-gradient-to-r from-purple-500 to-pink-500 h-full rounded-full transition-all duration-500"
                                  style={{ width: `${result.probabilities?.fraud * 100}%` }}
                                ></div>
                              </div>
                            </div>

                            <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4">
                              <div className="flex justify-between items-center mb-3">
                                <span className="text-sm font-bold text-gray-700">Legitimacy Score</span>
                                <span className="text-lg font-black text-green-600">
                                  {(result.probabilities?.no_fraud * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                                <div
                                  className="bg-gradient-to-r from-green-500 to-emerald-500 h-full rounded-full transition-all duration-500"
                                  style={{ width: `${result.probabilities?.no_fraud * 100}%` }}
                                ></div>
                              </div>
                            </div>

                            <div className={`rounded-lg p-4 border-2 ${result.fraud_detected
                              ? 'bg-red-50 border-red-300'
                              : 'bg-green-50 border-green-300'
                              }`}>
                              <p className="text-sm font-semibold text-gray-700 mb-2">Final Decision</p>
                              <p className={`text-xl font-black ${result.fraud_detected ? 'text-red-700' : 'text-green-700'
                                }`}>
                                {result.fraud_detected ? '⚠️ FRAUD DETECTED' : '✅ LEGITIMATE CLAIM'}
                              </p>
                            </div>
                          </div>
                        </CalculationCard>

                        {/* Recommended Actions */}
                        <CalculationCard title="📋 Recommended Actions" className="border-2 border-blue-200">
                          <div className="space-y-4">
                            <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-5 border-2 border-blue-200">
                              <p className="text-sm font-semibold text-gray-600 mb-3">Primary Action</p>
                              <div className="flex items-center gap-3 mb-3">
                                <div className={`w-12 h-12 rounded-full flex items-center justify-center ${result.recommended_action?.action === 'APPROVE' ? 'bg-green-500' :
                                  result.recommended_action?.action === 'REJECT' ? 'bg-red-500' :
                                    result.recommended_action?.action === 'INVESTIGATE' ? 'bg-orange-500' :
                                      'bg-blue-500'
                                  }`}>
                                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    {result.recommended_action?.action === 'APPROVE' ? (
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                                    ) : result.recommended_action?.action === 'REJECT' ? (
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    ) : (
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                    )}
                                  </svg>
                                </div>
                                <div>
                                  <p className="text-3xl font-black text-blue-700">
                                    {result.recommended_action?.action || 'REVIEW'}
                                  </p>
                                  <p className="text-xs text-gray-600 mt-1">
                                    {result.recommended_action?.message || 'Manual review required'}
                                  </p>
                                </div>
                              </div>
                            </div>

                            {result.recommended_action?.next_steps && result.recommended_action.next_steps.length > 0 && (
                              <div className="bg-white rounded-lg p-4 border border-blue-200">
                                <p className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                  </svg>
                                  Next Steps:
                                </p>
                                <ul className="space-y-2">
                                  {result.recommended_action.next_steps.map((step: string, idx: number) => (
                                    <li key={idx} className="flex items-start gap-3 text-sm text-gray-700">
                                      <span className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 text-blue-700 font-bold text-xs flex items-center justify-center mt-0.5">
                                        {idx + 1}
                                      </span>
                                      <span className="flex-1">{step}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                        </CalculationCard>
                      </div>

                      {/* AI Models Performance */}
                      <CalculationCard title="🤖 AI Models Performance" className="border-2 border-green-200">
                        <div className="grid md:grid-cols-3 gap-6">
                          {/* YOLO Detection */}
                          <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 border border-blue-200">
                            <div className="flex items-center gap-3 mb-4">
                              <div className="bg-blue-500 rounded-lg p-2">
                                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                                </svg>
                              </div>
                              <h4 className="font-bold text-blue-900">YOLO Detection</h4>
                            </div>
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Parts Detected:</span>
                                <span className="font-bold text-blue-700">
                                  {result.yolo_detection_results?.total_parts_detected || 0}
                                </span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Damages Found:</span>
                                <span className="font-bold text-red-600">
                                  {result.yolo_detection_results?.total_damages_detected || 0}
                                </span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Assignments:</span>
                                <span className="font-bold text-green-600">
                                  {result.yolo_detection_results?.total_assignments || 0}
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* CNN Analysis */}
                          <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-6 border border-purple-200">
                            <div className="flex items-center gap-3 mb-4">
                              <div className="bg-purple-500 rounded-lg p-2">
                                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                              </div>
                              <h4 className="font-bold text-purple-900">CNN Analysis</h4>
                            </div>
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Damage Areas:</span>
                                <span className="font-bold text-purple-700">
                                  {result.annotated_images?.reduce((sum: number, img: any) => sum + (img.total_damage_areas || 0), 0) || 0}
                                </span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Avg Coverage:</span>
                                <span className="font-bold text-orange-600">
                                  {result.annotated_images?.length > 0
                                    ? (result.annotated_images.reduce((sum: number, img: any) => sum + (img.damage_percentage || 0), 0) / result.annotated_images.length).toFixed(1)
                                    : 0}%
                                </span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Avg Confidence:</span>
                                <span className="font-bold text-green-600">
                                  {result.annotated_images?.length > 0
                                    ? (result.annotated_images.reduce((sum: number, img: any) => sum + (img.average_confidence || 0), 0) / result.annotated_images.length * 100).toFixed(0)
                                    : 0}%
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Tabular Model */}
                          <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 border border-green-200">
                            <div className="flex items-center gap-3 mb-4">
                              <div className="bg-green-500 rounded-lg p-2">
                                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                              </div>
                              <h4 className="font-bold text-green-900">XGBoost Model</h4>
                            </div>
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Model Type:</span>
                                <span className="font-bold text-green-700">
                                  {result.detailed_calculations?.tabular_analysis?.model_type?.split('Classifier')[0] || 'XGBoost'}
                                </span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Calibrated:</span>
                                <span className="font-bold text-blue-600">
                                  {result.detailed_calculations?.tabular_analysis?.is_calibrated ? '✓ Yes' : '✗ No'}
                                </span>
                              </div>
                              <div className="flex justify-between text-sm">
                                <span className="text-gray-600">Features:</span>
                                <span className="font-bold text-purple-600">
                                  {result.detailed_calculations?.tabular_analysis?.raw_features_shape?.[1] || 'N/A'}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </CalculationCard>

                      {/* Financial Summary */}
                      {result.calculated_claim_amount && (
                        <CalculationCard title="💰 Financial Summary" className="border-2 border-emerald-200 bg-gradient-to-br from-emerald-50 to-green-50">
                          <div className="grid md:grid-cols-4 gap-4">
                            <div className="bg-white rounded-xl p-5 shadow-md border border-emerald-200 text-center">
                              <p className="text-xs font-semibold text-gray-600 uppercase mb-2">Calculated Amount</p>
                              <p className="text-3xl font-black text-green-700">
                                {formatCurrency(result.calculated_claim_amount)}
                              </p>
                            </div>
                            <div className="bg-white rounded-xl p-5 shadow-md border border-blue-200 text-center">
                              <p className="text-xs font-semibold text-gray-600 uppercase mb-2">YOLO Base</p>
                              <p className="text-2xl font-bold text-blue-700">
                                {formatCurrency(result.claim_calculation_details?.yolo_base_amount || 0)}
                              </p>
                            </div>
                            <div className="bg-white rounded-xl p-5 shadow-md border border-purple-200 text-center">
                              <p className="text-xs font-semibold text-gray-600 uppercase mb-2">Parts Damaged</p>
                              <p className="text-2xl font-bold text-purple-700">
                                {result.claim_calculation_details?.total_damaged_parts || 0}
                              </p>
                            </div>
                            <div className="bg-white rounded-xl p-5 shadow-md border border-orange-200 text-center">
                              <p className="text-xs font-semibold text-gray-600 uppercase mb-2">CNN Multiplier</p>
                              <p className="text-2xl font-bold text-orange-700">
                                ×{result.claim_calculation_details?.cnn_multiplier?.toFixed(2) || 1}
                              </p>
                            </div>
                          </div>
                        </CalculationCard>
                      )}

                      {/* Analysis Timestamp */}
                      <div className="text-center text-sm text-gray-500 bg-gray-50 rounded-lg p-4 border border-gray-200">
                        <svg className="w-4 h-4 inline-block mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Analysis completed on {new Date().toLocaleString('en-IN', {
                          dateStyle: 'full',
                          timeStyle: 'short',
                          timeZone: 'Asia/Kolkata'
                        })}
                      </div>
                    </div>
                  )}

                  {/* ========================================== */}
                  {/* NEW: YOLO DETECTION TAB */}
                  {/* ========================================== */}
                  {activeTab === "yolo" && result.yolo_detection_results && (
                    <div className="space-y-6">
                      {/* Overall YOLO Stats */}
                      <CalculationCard title="YOLO Detection Summary">
                        <div className="grid md:grid-cols-4 gap-4">
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
                            <h5 className="text-sm font-medium text-blue-900">Total Parts Detected</h5>
                            <p className="text-3xl font-bold text-blue-700">
                              {result.yolo_detection_results.total_parts_detected || 0}
                            </p>
                          </div>
                          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
                            <h5 className="text-sm font-medium text-red-900">Total Damages Detected</h5>
                            <p className="text-3xl font-bold text-red-700">
                              {result.yolo_detection_results.total_damages_detected || 0}
                            </p>
                          </div>
                          <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
                            <h5 className="text-sm font-medium text-green-900">Damage Assignments</h5>
                            <p className="text-3xl font-bold text-green-700">
                              {result.yolo_detection_results.total_assignments || 0}
                            </p>
                          </div>
                          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 text-center">
                            <h5 className="text-sm font-medium text-purple-900">Images Analyzed</h5>
                            <p className="text-3xl font-bold text-purple-700">
                              {result.yolo_detection_results.all_images?.length || 0}
                            </p>
                          </div>
                        </div>
                      </CalculationCard>

                      {/* Image Selector */}
                      <CalculationCard title="Select Image for YOLO Analysis">
                        <div className="flex space-x-2 overflow-x-auto pb-2">
                          {result.yolo_detection_results.all_images?.map((img: any, idx: number) => (
                            <button
                              key={idx}
                              onClick={() => setSelectedImageIndex(idx)}
                              className={`px-4 py-2 rounded-lg border-2 transition-all whitespace-nowrap ${selectedImageIndex === idx
                                ? 'border-blue-500 bg-blue-50 text-blue-700 font-semibold'
                                : 'border-gray-200 bg-white text-gray-600 hover:border-blue-300'
                                }`}
                            >
                              Image #{img.image_index}
                              <span className="ml-2 text-xs">
                                ({img.parts_detected?.length || 0} parts, {img.damages_detected?.length || 0} damages)
                              </span>
                            </button>
                          ))}
                        </div>
                      </CalculationCard>

                      {/* Selected Image YOLO Details */}
                      {result.yolo_detection_results.all_images && result.yolo_detection_results.all_images[selectedImageIndex] && (
                        <div className="space-y-6">
                          <div className="grid md:grid-cols-2 gap-6">
                            {/* Parts Detected */}
                            <CalculationCard title={`🔧 Car Parts Detected (${result.yolo_detection_results.all_images[selectedImageIndex].parts_detected?.length || 0})`}>
                              {result.yolo_detection_results.all_images[selectedImageIndex].parts_detected &&
                                result.yolo_detection_results.all_images[selectedImageIndex].parts_detected.length > 0 ? (
                                <div className="space-y-2 max-h-96 overflow-y-auto">
                                  {result.yolo_detection_results.all_images[selectedImageIndex].parts_detected.map((part: any, pidx: number) => (
                                    <div key={pidx} className="flex justify-between items-center p-3 bg-blue-50 rounded-lg border border-blue-200">
                                      <div className="flex items-center">
                                        <div className="w-3 h-3 bg-blue-500 rounded-full mr-3"></div>
                                        <span className="font-medium text-gray-900 capitalize">
                                          {part.name?.replace(/_/g, ' ')}
                                        </span>
                                      </div>
                                      <div className="text-sm">
                                        <span className="text-blue-600 font-semibold">
                                          {(part.confidence * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <p className="text-gray-500 text-center py-8">No parts detected in this image</p>
                              )}
                            </CalculationCard>

                            {/* Damages Detected */}
                            <CalculationCard title={`⚠️ Damages Detected (${result.yolo_detection_results.all_images[selectedImageIndex].damages_detected?.length || 0})`}>
                              {result.yolo_detection_results.all_images[selectedImageIndex].damages_detected &&
                                result.yolo_detection_results.all_images[selectedImageIndex].damages_detected.length > 0 ? (
                                <div className="space-y-2 max-h-96 overflow-y-auto">
                                  {result.yolo_detection_results.all_images[selectedImageIndex].damages_detected.map((damage: any, didx: number) => (
                                    <div key={didx} className="flex justify-between items-center p-3 bg-red-50 rounded-lg border border-red-200">
                                      <div className="flex items-center">
                                        <div className="w-3 h-3 bg-red-500 rounded-full mr-3 animate-pulse"></div>
                                        <span className="font-medium text-gray-900 capitalize">
                                          {damage.name?.replace(/_/g, ' ')}
                                        </span>
                                      </div>
                                      <div className="text-sm">
                                        <span className="text-red-600 font-semibold">
                                          {(damage.confidence * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <p className="text-gray-500 text-center py-8">No damages detected in this image</p>
                              )}
                            </CalculationCard>
                          </div>

                          {/* Damage-to-Part Assignments */}
                          <CalculationCard title={`🔗 Damage → Part Assignments (${result.yolo_detection_results.all_images[selectedImageIndex].assignments?.length || 0})`}>
                            {result.yolo_detection_results.all_images[selectedImageIndex].assignments &&
                              result.yolo_detection_results.all_images[selectedImageIndex].assignments.length > 0 ? (
                              <div className="space-y-3">
                                {result.yolo_detection_results.all_images[selectedImageIndex].assignments.map((assignment: any, aidx: number) => (
                                  <div key={aidx} className="flex items-center justify-between p-4 bg-gradient-to-r from-orange-50 to-yellow-50 rounded-lg border border-orange-200">
                                    <div className="flex items-center flex-1">
                                      <div className="flex items-center min-w-0">
                                        <svg className="w-5 h-5 text-red-500 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                        </svg>
                                        <span className="font-semibold text-gray-900 capitalize truncate">
                                          {assignment.damage_type?.replace(/_/g, ' ')}
                                        </span>
                                      </div>

                                      <svg className="w-6 h-6 text-gray-400 mx-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                      </svg>

                                      <div className="flex items-center min-w-0">
                                        <svg className="w-5 h-5 text-blue-500 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                          <path d="M10.394 2.08a1 1 0 00-.788 0l-7 3a1 1 0 000 1.84L5.25 8.051a.999.999 0 01.356-.257l4-1.714a1 1 0 11.788 1.838L7.667 9.088l1.94.831a1 1 0 00.787 0l7-3a1 1 0 000-1.838l-7-3zM3.31 9.397L5 10.12v4.102a8.969 8.969 0 00-1.05-.174 1 1 0 01-.89-.89 11.115 11.115 0 01.25-3.762zM9.3 16.573A9.026 9.026 0 007 14.935v-3.957l1.818.78a3 3 0 002.364 0l5.508-2.361a11.026 11.026 0 01.25 3.762 1 1 0 01-.89.89 8.968 8.968 0 00-5.35 2.524 1 1 0 01-1.4 0zM6 18a1 1 0 001-1v-2.065a8.935 8.935 0 00-2-.712V17a1 1 0 001 1z" />
                                        </svg>
                                        <span className={`font-semibold capitalize truncate ${assignment.assigned_part ? 'text-blue-700' : 'text-gray-500 italic'
                                          }`}>
                                          {assignment.assigned_part?.replace(/_/g, ' ') || 'Unassigned'}
                                        </span>
                                      </div>
                                    </div>

                                    <div className="ml-4 text-right flex-shrink-0">
                                      <div className="text-xs text-gray-600">Confidence</div>
                                      <div className="text-sm font-bold text-orange-600">
                                        {(assignment.damage_confidence * 100).toFixed(1)}%
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p className="text-gray-500 text-center py-8">No damage assignments for this image</p>
                            )}
                          </CalculationCard>
                        </div>
                      )}
                    </div>
                  )}


                  {/* ========================================== */}
                  {/* CLAIM BREAKDOWN TAB */}
                  {/* ========================================== */}
                  {activeTab === "claim-breakdown" && result.claim_calculation_details && (
                    <div className="space-y-6 animate-fade-in">
                      <CalculationCard title="💰 Claim Amount Calculation Details" className="border-2 border-blue-200">
                        <div className="space-y-6">
                          {/* Summary Cards */}
                          <div className="grid md:grid-cols-3 gap-4">
                            <div className="bg-gradient-to-br from-blue-50 to-blue-100 border-2 border-blue-300 rounded-xl p-6 text-center shadow-lg hover:shadow-xl transition-all">
                              <div className="flex justify-center mb-3">
                                <div className="bg-blue-500 rounded-full p-3">
                                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                                  </svg>
                                </div>
                              </div>
                              <h5 className="text-sm font-bold text-blue-900 mb-2">YOLO Base Amount</h5>
                              <p className="text-4xl font-black text-blue-700 mb-2">
                                {formatCurrency(result.claim_calculation_details.yolo_base_amount || 0)}
                              </p>
                              <p className="text-xs text-blue-600 bg-blue-200 rounded-full px-3 py-1 inline-block">
                                Sum of part damages
                              </p>
                            </div>

                            <div className="bg-gradient-to-br from-purple-50 to-purple-100 border-2 border-purple-300 rounded-xl p-6 text-center shadow-lg hover:shadow-xl transition-all">
                              <div className="flex justify-center mb-3">
                                <div className="bg-purple-500 rounded-full p-3">
                                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                  </svg>
                                </div>
                              </div>
                              <h5 className="text-sm font-bold text-purple-900 mb-2">CNN Multiplier</h5>
                              <p className="text-4xl font-black text-purple-700 mb-2">
                                ×{result.claim_calculation_details.cnn_multiplier?.toFixed(2) || 1}
                              </p>
                              <p className="text-xs text-purple-600 bg-purple-200 rounded-full px-3 py-1 inline-block">
                                {result.claim_calculation_details.cnn_damage_percentage?.toFixed(1) || 0}% damage severity
                              </p>
                            </div>

                            <div className="bg-gradient-to-br from-green-50 to-green-100 border-2 border-green-300 rounded-xl p-6 text-center shadow-lg hover:shadow-xl transition-all">
                              <div className="flex justify-center mb-3">
                                <div className="bg-green-500 rounded-full p-3">
                                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                  </svg>
                                </div>
                              </div>
                              <h5 className="text-sm font-bold text-green-900 mb-2">Final Amount</h5>
                              <p className="text-4xl font-black text-green-700 mb-2">
                                {formatCurrency(result.claim_calculation_details.final_calculated_amount || 0)}
                              </p>
                              <p className="text-xs text-green-600 bg-green-200 rounded-full px-3 py-1 inline-block">
                                Base × Multiplier
                              </p>
                            </div>
                          </div>

                          {/* Calculation Formula */}
                          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border-2 border-indigo-300 rounded-xl p-6">
                            <div className="flex items-center gap-3 mb-4">
                              <div className="bg-indigo-500 rounded-lg p-2">
                                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                                </svg>
                              </div>
                              <h5 className="font-bold text-indigo-900 text-lg">Calculation Formula</h5>
                            </div>
                            <div className="bg-white rounded-lg p-6 border-2 border-indigo-200">
                              <p className="font-mono text-base text-center text-gray-800 font-semibold">
                                {result.claim_calculation_details.calculation_formula ||
                                  `${formatCurrency(result.claim_calculation_details.yolo_base_amount || 0)} × ${result.claim_calculation_details.cnn_multiplier?.toFixed(2) || 1} = ${formatCurrency(result.claim_calculation_details.final_calculated_amount || 0)}`}
                              </p>
                            </div>
                          </div>

                          {/* Detailed Breakdown */}
                          {result.claim_calculation_details.detailed_breakdown &&
                            result.claim_calculation_details.detailed_breakdown.length > 0 && (
                              <div className="bg-white border-2 border-gray-200 rounded-xl overflow-hidden">
                                <div className="bg-gradient-to-r from-gray-700 to-gray-900 px-6 py-4">
                                  <h5 className="font-bold text-white text-lg flex items-center gap-2">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                                    </svg>
                                    Itemized Damage Breakdown ({result.claim_calculation_details.total_damaged_parts || 0} parts)
                                  </h5>
                                </div>

                                <div className="p-6">
                                  <div className="overflow-x-auto">
                                    <table className="w-full">
                                      <thead>
                                        <tr className="bg-gradient-to-r from-gray-100 to-gray-200 border-b-2 border-gray-300">
                                          <th className="px-4 py-3 text-left text-sm font-bold text-gray-700">Image #</th>
                                          <th className="px-4 py-3 text-left text-sm font-bold text-gray-700">Part Name</th>
                                          <th className="px-4 py-3 text-left text-sm font-bold text-gray-700">Damage Type</th>
                                          <th className="px-4 py-3 text-right text-sm font-bold text-gray-700">Part Price</th>
                                          <th className="px-4 py-3 text-center text-sm font-bold text-gray-700">Severity</th>
                                          <th className="px-4 py-3 text-right text-sm font-bold text-gray-700">Damage Cost</th>
                                          <th className="px-4 py-3 text-center text-sm font-bold text-gray-700">Confidence</th>
                                        </tr>
                                      </thead>
                                      <tbody className="divide-y divide-gray-200">
                                        {result.claim_calculation_details.detailed_breakdown.map((item: any, idx: number) => (
                                          <tr key={idx} className="hover:bg-blue-50 transition-colors">
                                            <td className="px-4 py-3 text-sm">
                                              <span className="bg-blue-100 text-blue-800 font-semibold px-3 py-1 rounded-full text-xs">
                                                #{item.image_index}
                                              </span>
                                            </td>
                                            <td className="px-4 py-3">
                                              <div className="flex items-center gap-2">
                                                <div className={`w-3 h-3 rounded-full ${item.part === 'UNASSIGNED' ? 'bg-gray-400' : 'bg-blue-500'
                                                  }`}></div>
                                                <span className={`font-semibold text-sm capitalize ${item.part === 'UNASSIGNED' ? 'text-gray-500 italic' : 'text-gray-900'
                                                  }`}>
                                                  {item.part?.replace(/_/g, ' ') || 'Unknown'}
                                                </span>
                                              </div>
                                            </td>
                                            <td className="px-4 py-3">
                                              <span className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-xs font-semibold capitalize">
                                                {item.damage_type?.replace(/_/g, ' ')}
                                              </span>
                                            </td>
                                            <td className="px-4 py-3 text-right font-semibold text-gray-700">
                                              {formatCurrency(item.part_price || 0)}
                                            </td>
                                            <td className="px-4 py-3 text-center">
                                              <div className="flex items-center justify-center gap-1">
                                                <div className="bg-orange-200 rounded-full px-3 py-1">
                                                  <span className="text-orange-800 font-bold text-xs">
                                                    ×{item.severity_multiplier?.toFixed(2) || 1}
                                                  </span>
                                                </div>
                                              </div>
                                            </td>
                                            <td className="px-4 py-3 text-right">
                                              <span className="font-bold text-green-700 text-base">
                                                {formatCurrency(item.damage_cost || 0)}
                                              </span>
                                            </td>
                                            <td className="px-4 py-3 text-center">
                                              <div className="flex items-center justify-center">
                                                <div className="bg-purple-100 rounded-full px-3 py-1">
                                                  <span className="text-purple-800 font-semibold text-xs">
                                                    {(item.confidence * 100).toFixed(0)}%
                                                  </span>
                                                </div>
                                              </div>
                                            </td>
                                          </tr>
                                        ))}
                                      </tbody>
                                      <tfoot>
                                        <tr className="bg-gradient-to-r from-green-100 to-emerald-100 border-t-2 border-green-300">
                                          <td colSpan={5} className="px-4 py-4 text-right font-bold text-gray-800 text-base">
                                            Total YOLO Base Amount:
                                          </td>
                                          <td className="px-4 py-4 text-right">
                                            <span className="font-black text-green-700 text-xl">
                                              {formatCurrency(result.claim_calculation_details.yolo_base_amount || 0)}
                                            </span>
                                          </td>
                                          <td></td>
                                        </tr>
                                      </tfoot>
                                    </table>
                                  </div>

                                  {/* Summary Stats */}
                                  <div className="mt-6 grid md:grid-cols-4 gap-4">
                                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-4 text-center border border-blue-200">
                                      <p className="text-xs text-blue-700 font-semibold mb-1">Total Items</p>
                                      <p className="text-2xl font-black text-blue-900">
                                        {result.claim_calculation_details.detailed_breakdown.length}
                                      </p>
                                    </div>
                                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-4 text-center border border-purple-200">
                                      <p className="text-xs text-purple-700 font-semibold mb-1">Avg Confidence</p>
                                      <p className="text-2xl font-black text-purple-900">
                                        {(result.claim_calculation_details.detailed_breakdown.reduce((sum: number, item: any) => sum + (item.confidence || 0), 0) / result.claim_calculation_details.detailed_breakdown.length * 100).toFixed(0)}%
                                      </p>
                                    </div>
                                    <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg p-4 text-center border border-orange-200">
                                      <p className="text-xs text-orange-700 font-semibold mb-1">Avg Severity</p>
                                      <p className="text-2xl font-black text-orange-900">
                                        {(result.claim_calculation_details.detailed_breakdown.reduce((sum: number, item: any) => sum + (item.severity_multiplier || 0), 0) / result.claim_calculation_details.detailed_breakdown.length).toFixed(2)}×
                                      </p>
                                    </div>
                                    <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-4 text-center border border-green-200">
                                      <p className="text-xs text-green-700 font-semibold mb-1">Unassigned</p>
                                      <p className="text-2xl font-black text-green-900">
                                        {result.claim_calculation_details.detailed_breakdown.filter((item: any) => item.part === 'UNASSIGNED').length}
                                      </p>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}

                          {/* Empty State */}
                          {(!result.claim_calculation_details.detailed_breakdown ||
                            result.claim_calculation_details.detailed_breakdown.length === 0) && (
                              <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-xl p-12 text-center">
                                <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                <p className="text-gray-600 font-semibold mb-2">No Detailed Breakdown Available</p>
                                <p className="text-gray-500 text-sm">The itemized damage breakdown could not be generated for this claim.</p>
                              </div>
                            )}
                        </div>
                      </CalculationCard>

                      {/* Additional Info Card */}
                      <CalculationCard title="ℹ️ Calculation Method" className="bg-gradient-to-br from-indigo-50 to-blue-50 border-2 border-indigo-200">
                        <div className="space-y-4">
                          <div className="flex items-start gap-4">
                            <div className="bg-indigo-500 rounded-lg p-3 flex-shrink-0">
                              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                              </svg>
                            </div>
                            <div className="flex-1">
                              <h5 className="font-bold text-indigo-900 mb-2">How is the claim amount calculated?</h5>
                              <ul className="space-y-2 text-sm text-gray-700">
                                <li className="flex items-start gap-2">
                                  <span className="text-blue-500 font-bold mt-1">1.</span>
                                  <span><strong className="text-blue-700">YOLO Detection:</strong> Identifies damaged car parts and damage types from uploaded images</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <span className="text-blue-500 font-bold mt-1">2.</span>
                                  <span><strong className="text-blue-700">Part Pricing:</strong> Each detected part is assigned a replacement/repair cost from our database</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <span className="text-blue-500 font-bold mt-1">3.</span>
                                  <span><strong className="text-blue-700">Severity Multiplier:</strong> Damage type affects the cost (scratch: 0.2×, dent: 0.4×, broken: 1.0×)</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <span className="text-blue-500 font-bold mt-1">4.</span>
                                  <span><strong className="text-blue-700">CNN Adjustment:</strong> Overall damage percentage adds a multiplier to account for total severity</span>
                                </li>
                                <li className="flex items-start gap-2">
                                  <span className="text-blue-500 font-bold mt-1">5.</span>
                                  <span><strong className="text-blue-700">Final Calculation:</strong> Base amount × CNN multiplier = Final claim amount</span>
                                </li>
                              </ul>
                            </div>
                          </div>
                        </div>
                      </CalculationCard>
                    </div>
                  )}

                  {/* NEW: Multi-Images Tab */}
                  {activeTab === "multi-images" && result.multi_image_analysis && (
                    <div className="space-y-6">
                      {/* Aggregated Metrics Overview */}
                      {result.multi_image_analysis.aggregated_metrics && (
                        <CalculationCard title="Multi-Image Analysis Summary">
                          <div className="space-y-4">
                            {/* Key Metrics */}
                            <div className="grid md:grid-cols-4 gap-4">
                              <div className="p-4 bg-blue-50 rounded-lg text-center">
                                <h5 className="text-sm font-medium text-blue-900">Total Images</h5>
                                <p className="text-2xl font-bold text-blue-700">
                                  {result.multi_image_analysis.aggregated_metrics.total_images}
                                </p>
                              </div>
                              <div className="p-4 bg-purple-50 rounded-lg text-center">
                                <h5 className="text-sm font-medium text-purple-900">Avg Fraud Prob</h5>
                                <p className="text-2xl font-bold text-purple-700">
                                  {(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.mean * 100).toFixed(1)}%
                                </p>
                              </div>
                              <div className="p-4 bg-red-50 rounded-lg text-center">
                                <h5 className="text-sm font-medium text-red-900">Max Fraud Prob</h5>
                                <p className="text-2xl font-bold text-red-700">
                                  {(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.max * 100).toFixed(1)}%
                                </p>
                              </div>
                              <div className="p-4 bg-orange-50 rounded-lg text-center">
                                <h5 className="text-sm font-medium text-orange-900">Overall Severity</h5>
                                <p className="text-2xl font-bold text-orange-700">
                                  {result.multi_image_analysis.aggregated_metrics.severity_analysis.overall_severity}
                                </p>
                              </div>
                            </div>

                            {/* Aggregation Details */}
                            <div className="p-4 bg-gray-50 rounded-lg">
                              <h5 className="font-medium text-gray-900 mb-3">Aggregation Components</h5>
                              <div className="grid md:grid-cols-2 gap-3 text-sm">
                                <div className="flex justify-between">
                                  <span>Max Fraud Image:</span>
                                  <span className="font-semibold">
                                    Image #{result.multi_image_analysis.aggregated_metrics.aggregation_components.max_fraud_image_index}
                                    ({(result.multi_image_analysis.aggregated_metrics.aggregation_components.max_fraud_probability * 100).toFixed(1)}%)
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span>Weighted Average:</span>
                                  <span className="font-semibold">
                                    {(result.multi_image_analysis.aggregated_metrics.aggregation_components.weighted_average * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span>Top-2 Average:</span>
                                  <span className="font-semibold">
                                    {(result.multi_image_analysis.aggregated_metrics.aggregation_components.top_k_average * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span>Damage-Weighted:</span>
                                  <span className="font-semibold">
                                    {(result.multi_image_analysis.aggregated_metrics.aggregation_components.damage_weighted * 100).toFixed(1)}%
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* Fraud Probability Distribution */}
                            <div className="p-4 bg-yellow-50 rounded-lg">
                              <h5 className="font-medium text-yellow-900 mb-3">Fraud Probability Distribution</h5>
                              <div className="space-y-2">
                                <div className="flex items-center">
                                  <div className="w-32 text-sm text-gray-700">Range:</div>
                                  <div className="flex-1 bg-white rounded-full h-6 overflow-hidden flex items-center px-2">
                                    <div className="flex justify-between w-full text-xs">
                                      <span>{(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.min * 100).toFixed(1)}%</span>
                                      <span className="font-semibold">{(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.median * 100).toFixed(1)}%</span>
                                      <span>{(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.max * 100).toFixed(1)}%</span>
                                    </div>
                                  </div>
                                </div>
                                <div className="flex items-center text-sm">
                                  <div className="w-32 text-gray-700">Std Deviation:</div>
                                  <div className="font-semibold">{(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.std * 100).toFixed(1)}%</div>
                                  <div className="ml-4 text-xs text-gray-600">
                                    (Consistency: {result.multi_image_analysis.aggregated_metrics.recommendation.consistency})
                                  </div>
                                </div>
                              </div>
                            </div>

                            {/* Images Requiring Attention */}
                            {result.multi_image_analysis.aggregated_metrics.recommendation.images_requiring_attention.length > 0 && (
                              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                                <h5 className="font-medium text-red-900 mb-2">⚠️ High-Risk Images Detected</h5>
                                <p className="text-sm text-red-800">
                                  Images requiring special attention:
                                  <span className="font-semibold ml-2">
                                    {result.multi_image_analysis.aggregated_metrics.recommendation.images_requiring_attention.map(idx => `#${idx}`).join(', ')}
                                  </span>
                                </p>
                              </div>
                            )}
                          </div>
                        </CalculationCard>
                      )}

                      {/* Individual Image Results */}
                      {result.multi_image_analysis.individual_images && (
                        <CalculationCard title="Individual Image Analysis">
                          <div className="space-y-4">
                            {/* Image Selector */}
                            <div className="flex space-x-2 overflow-x-auto pb-2">
                              {result.multi_image_analysis.individual_images.map((img: any, idx: number) => (
                                <button
                                  key={idx}
                                  onClick={() => setSelectedImageIndex(idx)}
                                  className={`px-4 py-2 rounded-lg border-2 transition-all whitespace-nowrap ${selectedImageIndex === idx
                                    ? 'border-blue-500 bg-blue-50 text-blue-700 font-semibold'
                                    : 'border-gray-200 bg-white text-gray-600 hover:border-blue-300'
                                    }`}
                                >
                                  Image #{img.image_index}
                                  <span className="ml-2 text-xs">
                                    ({(img.image_fraud_probability * 100).toFixed(1)}%)
                                  </span>
                                </button>
                              ))}
                            </div>

                            {/* Selected Image Details */}
                            {result.multi_image_analysis.individual_images[selectedImageIndex] && (
                              <div className="space-y-4">
                                <div className="grid md:grid-cols-3 gap-4">
                                  <div className="p-4 bg-blue-50 rounded-lg">
                                    <h5 className="text-sm font-medium text-blue-900">Fraud Probability</h5>
                                    <p className="text-2xl font-bold text-blue-700">
                                      {(result.multi_image_analysis.individual_images[selectedImageIndex].image_fraud_probability * 100).toFixed(1)}%
                                    </p>
                                  </div>
                                  <div className="p-4 bg-purple-50 rounded-lg">
                                    <h5 className="text-sm font-medium text-purple-900">Confidence</h5>
                                    <p className="text-2xl font-bold text-purple-700">
                                      {(result.multi_image_analysis.individual_images[selectedImageIndex].image_confidence * 100).toFixed(1)}%
                                    </p>
                                  </div>
                                  <div className="p-4 bg-orange-50 rounded-lg">
                                    <h5 className="text-sm font-medium text-orange-900">Damage %</h5>
                                    <p className="text-2xl font-bold text-orange-700">
                                      {result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.damage_percentage.toFixed(2)}%
                                    </p>
                                  </div>
                                </div>

                                {/* Damage Details */}
                                <div className="p-4 bg-gray-50 rounded-lg">
                                  <h5 className="font-medium text-gray-900 mb-3">Damage Analysis</h5>
                                  <div className="grid md:grid-cols-2 gap-4 text-sm">
                                    <div>
                                      <span className="text-gray-600">Severity Level:</span>
                                      <span className={`ml-2 font-semibold ${result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.severity_level === 'HIGH' ? 'text-red-600' :
                                        result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.severity_level === 'MEDIUM' ? 'text-orange-600' :
                                          'text-green-600'
                                        }`}>
                                        {result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.severity_level}
                                      </span>
                                    </div>
                                    <div>
                                      <span className="text-gray-600">Detections:</span>
                                      <span className="ml-2 font-semibold">
                                        {result.multi_image_analysis.individual_images[selectedImageIndex].detection_results.high_confidence_detections} areas
                                      </span>
                                    </div>
                                    <div>
                                      <span className="text-gray-600">Total Damage Area:</span>
                                      <span className="ml-2 font-semibold">
                                        {result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.total_damage_area_pixels.toFixed(0)} px²
                                      </span>
                                    </div>
                                    <div>
                                      <span className="text-gray-600">Severity Score:</span>
                                      <span className="ml-2 font-semibold">
                                        {(result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.severity_score * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                  </div>
                                </div>

                                {/* Damage Regions */}
                                {result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.damage_regions &&
                                  result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.damage_regions.length > 0 && (
                                    <div className="p-4 bg-green-50 rounded-lg">
                                      <h5 className="font-medium text-green-900 mb-3">
                                        Damage Regions ({result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.damage_regions.length})
                                      </h5>
                                      <div className="space-y-2 max-h-60 overflow-y-auto">
                                        {result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.damage_regions.map((region: any, idx: number) => (
                                          <div key={idx} className="flex justify-between items-center p-2 bg-white rounded border text-sm">
                                            <span className="font-medium">Region #{region.region_id}</span>
                                            <div className="text-xs text-gray-600">
                                              <span>Area: {region.area_pixels.toFixed(0)}px²</span>
                                              <span className="ml-2">({(region.relative_size * 100).toFixed(2)}%)</span>
                                              <span className="ml-2 text-blue-600">Conf: {(region.confidence * 100).toFixed(1)}%</span>
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                              </div>
                            )}
                          </div>
                        </CalculationCard>
                      )}
                    </div>
                  )}



                  {/* Tabular Analysis Tab */}
                  {activeTab === "tabular" && result.detailed_calculations?.tabular_analysis && (
                    <div className="space-y-6">
                      <CalculationCard title="Model Information">
                        <div className="space-y-4">
                          <div className="grid md:grid-cols-2 gap-4">
                            <div className="p-4 bg-blue-50 rounded-lg">
                              <h5 className="font-medium text-blue-900">Model Type</h5>
                              <p className="text-lg font-semibold">{result.detailed_calculations.tabular_analysis.model_type}</p>
                              {result.detailed_calculations.tabular_analysis.is_calibrated && (
                                <p className="text-sm text-blue-600 mt-1">✓ Calibrated (Platt Scaling)</p>
                              )}
                            </div>
                            <div className="p-4 bg-green-50 rounded-lg">
                              <h5 className="font-medium text-green-900">Features Shape</h5>
                              <p className="text-lg font-semibold">
                                {result.detailed_calculations.tabular_analysis.raw_features_shape?.join(' × ')}
                              </p>
                            </div>
                          </div>
                        </div>
                      </CalculationCard>

                      <CalculationCard title="Fraud Predictions">
                        <div className="space-y-4">
                          <div className="grid md:grid-cols-2 gap-4">
                            <div className="p-4 bg-blue-50 rounded-lg">
                              <h5 className="font-medium text-blue-900">No Fraud Probability</h5>
                              <p className="text-2xl font-bold text-blue-600">
                                {(result.detailed_calculations.tabular_analysis.probabilities.no_fraud * 100).toFixed(1)}%
                              </p>
                            </div>
                            <div className="p-4 bg-red-50 rounded-lg">
                              <h5 className="font-medium text-red-900">Fraud Probability</h5>
                              <p className="text-2xl font-bold text-red-600">
                                {(result.detailed_calculations.tabular_analysis.probabilities.fraud * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>

                          {/* Threshold Strategies */}
                          {result.detailed_calculations.tabular_analysis.threshold_strategies && (
                            <div>
                              <h5 className="font-medium text-gray-700 mb-3">Threshold Strategies</h5>
                              <div className="space-y-2">
                                {Object.entries(result.detailed_calculations.tabular_analysis.threshold_strategies).map(([strategyName, strategy]: [string, any]) => (
                                  <div key={strategyName} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                                    <div>
                                      <span className="font-medium">{strategyName.replace(/_/g, ' ').toUpperCase()}</span>
                                      <span className="text-xs text-gray-500 ml-2">
                                        (threshold: {(strategy.threshold * 100).toFixed(1)}%)
                                      </span>
                                    </div>
                                    <div className="text-sm">
                                      <span className={strategy.prediction === 1 ? 'text-red-600 font-semibold' : 'text-green-600'}>
                                        {strategy.prediction === 1 ? 'FRAUD' : 'NO FRAUD'}
                                      </span>
                                      <span className="text-gray-500 ml-2">
                                        ({strategy.distance_from_threshold > 0 ? '+' : ''}{(strategy.distance_from_threshold * 100).toFixed(1)}%)
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Probability Analysis */}
                          {result.detailed_calculations.tabular_analysis.probability_analysis && (
                            <div className="p-4 bg-purple-50 rounded-lg">
                              <h5 className="font-medium text-purple-900 mb-2">Probability Quality</h5>
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <span className="text-gray-600">Range:</span>
                                  <span className="font-semibold ml-2">
                                    {result.detailed_calculations.tabular_analysis.probability_analysis.probability_range}
                                  </span>
                                </div>
                                <div>
                                  <span className="text-gray-600">Separation:</span>
                                  <span className="font-semibold ml-2">
                                    {result.detailed_calculations.tabular_analysis.probability_analysis.separation_quality}
                                  </span>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </CalculationCard>

                      {/* Primary Prediction Strategy */}
                      {result.detailed_calculations.tabular_analysis.primary_prediction && (
                        <CalculationCard title="Primary Prediction" className="border-2 border-blue-200">
                          <div className="space-y-3">
                            <div className="flex justify-between items-center">
                              <span className="font-medium">Strategy Used:</span>
                              <span className="text-blue-600 font-semibold">
                                {result.detailed_calculations.tabular_analysis.primary_prediction.strategy.replace(/_/g, ' ').toUpperCase()}
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="font-medium">Threshold:</span>
                              <span>{(result.detailed_calculations.tabular_analysis.primary_prediction.threshold * 100).toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="font-medium">Confidence:</span>
                              <span className="font-semibold">{(result.detailed_calculations.tabular_analysis.primary_prediction.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div className={`p-3 rounded-lg text-center ${result.detailed_calculations.tabular_analysis.primary_prediction.fraud_detected
                              ? 'bg-red-100 text-red-800'
                              : 'bg-green-100 text-green-800'
                              }`}>
                              <span className="font-bold text-lg">
                                {result.detailed_calculations.tabular_analysis.primary_prediction.fraud_detected ? '🚨 FRAUD DETECTED' : '✅ NO FRAUD'}
                              </span>
                            </div>
                          </div>
                        </CalculationCard>
                      )}
                    </div>
                  )}

                  {/* Damage Detection Tab */}
                  {activeTab === "damage" && (
                    <div className="space-y-6">
                      {result.annotated_images && result.annotated_images.length > 0 ? (
                        <>
                          {/* Overall Damage Summary */}
                          <CalculationCard title="Overall Damage Summary">
                            <div className="grid md:grid-cols-4 gap-4">
                              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
                                <h5 className="text-sm font-medium text-blue-900">Total Images</h5>
                                <p className="text-3xl font-bold text-blue-700">{result.annotated_images.length}</p>
                              </div>

                              <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
                                <h5 className="text-sm font-medium text-red-900">Total Damage Areas</h5>
                                <p className="text-3xl font-bold text-red-700">
                                  {result.annotated_images.reduce((sum: number, img: any) => sum + (img.total_damage_areas || 0), 0)}
                                </p>
                              </div>

                              <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 text-center">
                                <h5 className="text-sm font-medium text-orange-900">Avg Damage %</h5>
                                <p className="text-3xl font-bold text-orange-700">
                                  {(result.annotated_images.reduce((sum: number, img: any) => sum + (img.damage_percentage || 0), 0) / result.annotated_images.length).toFixed(1)}%
                                </p>
                              </div>

                              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 text-center">
                                <h5 className="text-sm font-medium text-purple-900">Avg Confidence</h5>
                                <p className="text-3xl font-bold text-purple-700">
                                  {(result.annotated_images.reduce((sum: number, img: any) => sum + (img.average_confidence || 0), 0) / result.annotated_images.length * 100).toFixed(1)}%
                                </p>
                              </div>
                            </div>
                          </CalculationCard>

                          {/* Image Selector */}
                          <CalculationCard title="Select Image for Detailed Analysis">
                            <div className="space-y-4">
                              <div className="flex space-x-2 overflow-x-auto pb-2">
                                {result.annotated_images.map((img: any, idx: number) => (
                                  <button
                                    key={idx}
                                    onClick={() => setSelectedImageIndex(idx)}
                                    className={`relative flex-shrink-0 w-32 h-32 rounded-lg border-2 transition-all ${selectedImageIndex === idx
                                      ? 'border-blue-500 shadow-lg ring-2 ring-blue-300'
                                      : 'border-gray-200 hover:border-blue-300'
                                      }`}
                                  >
                                    {img.annotated_image_base64 ? (
                                      <>
                                        <img
                                          src={`data:image/jpeg;base64,${img.annotated_image_base64}`}
                                          alt={`Thumbnail ${idx + 1}`}
                                          className="w-full h-full object-cover rounded-lg"
                                        />
                                        {selectedImageIndex === idx && (
                                          <div className="absolute inset-0 bg-blue-500 bg-opacity-20 rounded-lg flex items-center justify-center">
                                            <svg className="w-8 h-8 text-white drop-shadow-lg" fill="currentColor" viewBox="0 0 20 20">
                                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                            </svg>
                                          </div>
                                        )}
                                        {/* Damage indicator badge */}
                                        <div className={`absolute top-2 right-2 px-2 py-1 rounded text-xs font-bold ${img.severity === 'HIGH' ? 'bg-red-500 text-white' :
                                          img.severity === 'MEDIUM' ? 'bg-orange-500 text-white' :
                                            'bg-green-500 text-white'
                                          }`}>
                                          {img.severity}
                                        </div>
                                      </>
                                    ) : (
                                      <div className="w-full h-full flex items-center justify-center bg-gray-100 rounded-lg">
                                        <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                        </svg>
                                      </div>
                                    )}
                                    <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white text-xs p-1 text-center rounded-b-lg">
                                      Image #{img.image_index || idx + 1}
                                    </div>
                                  </button>
                                ))}
                              </div>
                            </div>
                          </CalculationCard>

                          {/* Selected Image Detailed Analysis */}
                          {result.annotated_images[selectedImageIndex] && (
                            <>
                              {/* Annotated Image Display */}
                              <CalculationCard title={`Image #${result.annotated_images[selectedImageIndex].image_index || selectedImageIndex + 1} - Damage Detection with Masking`}>
                                <div className="space-y-4">
                                  {/* Large annotated image */}
                                  <div className="border-4 border-blue-300 rounded-lg overflow-hidden bg-gray-900 shadow-xl">
                                    <img
                                      src={`data:image/jpeg;base64,${result.annotated_images[selectedImageIndex].annotated_image_base64}`}
                                      alt={`Damage detection for image ${selectedImageIndex + 1}`}
                                      className="w-full h-auto"
                                    />
                                  </div>

                                  {/* Detection Metrics */}
                                  <div className="grid md:grid-cols-4 gap-4">
                                    <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
                                      <h5 className="text-sm font-medium text-red-900 mb-1">Damage Areas</h5>
                                      <p className="text-3xl font-bold text-red-700">
                                        {result.annotated_images[selectedImageIndex].total_damage_areas}
                                      </p>
                                      <p className="text-xs text-red-600 mt-1">detected regions</p>
                                    </div>

                                    <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 text-center">
                                      <h5 className="text-sm font-medium text-orange-900 mb-1">Coverage</h5>
                                      <p className="text-3xl font-bold text-orange-700">
                                        {result.annotated_images[selectedImageIndex].damage_percentage.toFixed(1)}%
                                      </p>
                                      <p className="text-xs text-orange-600 mt-1">of vehicle surface</p>
                                    </div>

                                    <div className={`border-2 rounded-lg p-4 text-center ${result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'bg-red-50 border-red-300' :
                                      result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'bg-yellow-50 border-yellow-300' :
                                        'bg-green-50 border-green-300'
                                      }`}>
                                      <h5 className={`text-sm font-medium mb-1 ${result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'text-red-900' :
                                        result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'text-yellow-900' :
                                          'text-green-900'
                                        }`}>Severity</h5>
                                      <p className={`text-3xl font-bold ${result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'text-red-700' :
                                        result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'text-yellow-700' :
                                          'text-green-700'
                                        }`}>
                                        {result.annotated_images[selectedImageIndex].severity}
                                      </p>
                                    </div>

                                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
                                      <h5 className="text-sm font-medium text-blue-900 mb-1">Avg Confidence</h5>
                                      <p className="text-3xl font-bold text-blue-700">
                                        {(result.annotated_images[selectedImageIndex].average_confidence * 100).toFixed(0)}%
                                      </p>
                                      <p className="text-xs text-blue-600 mt-1">detection quality</p>
                                    </div>
                                  </div>

                                  {/* Image Dimensions */}
                                  <div className="p-4 bg-gray-50 rounded-lg">
                                    <h5 className="font-medium text-gray-900 mb-2">Image Properties</h5>
                                    <div className="grid grid-cols-3 gap-4 text-sm">
                                      <div>
                                        <span className="text-gray-600">Width:</span>
                                        <span className="ml-2 font-semibold">{result.annotated_images[selectedImageIndex].original_dimensions.width}px</span>
                                      </div>
                                      <div>
                                        <span className="text-gray-600">Height:</span>
                                        <span className="ml-2 font-semibold">{result.annotated_images[selectedImageIndex].original_dimensions.height}px</span>
                                      </div>
                                      <div>
                                        <span className="text-gray-600">Filename:</span>
                                        <span className="ml-2 font-semibold text-xs">{result.annotated_images[selectedImageIndex].image_filename || `image_${selectedImageIndex + 1}.jpg`}</span>
                                      </div>
                                    </div>
                                  </div>

                                  {/* Detected Damage Areas List */}
                                  {result.annotated_images[selectedImageIndex].damage_areas &&
                                    result.annotated_images[selectedImageIndex].damage_areas.length > 0 && (
                                      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                                        <h5 className="font-medium text-red-900 mb-3 flex items-center">
                                          <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                          </svg>
                                          Detected Damage Regions ({result.annotated_images[selectedImageIndex].damage_areas.length})
                                        </h5>
                                        <div className="space-y-2 max-h-60 overflow-y-auto">
                                          {result.annotated_images[selectedImageIndex].damage_areas.map((area: any, areaIdx: number) => (
                                            <div key={areaIdx} className="flex justify-between items-center p-3 bg-white rounded border border-red-100 hover:border-red-300 transition-colors">
                                              <div className="flex items-center">
                                                <div className="w-4 h-4 bg-red-500 rounded-full mr-3 flex-shrink-0 animate-pulse"></div>
                                                <div>
                                                  <span className="font-medium text-gray-900">{area.label || `Damage Region ${areaIdx + 1}`}</span>
                                                  <p className="text-xs text-gray-500 mt-1">
                                                    Position: ({area.bbox[0].toFixed(0)}, {area.bbox[1].toFixed(0)}) to ({area.bbox[2].toFixed(0)}, {area.bbox[3].toFixed(0)})
                                                  </p>
                                                </div>
                                              </div>
                                              <div className="text-sm text-right">
                                                <div className="font-semibold text-gray-900">
                                                  {area.area.toFixed(0)} px²
                                                </div>
                                                <div className="text-xs text-blue-600 font-semibold mt-1">
                                                  {(area.confidence * 100).toFixed(1)}% confidence
                                                </div>
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    )}

                                  {/* No Damage Detected Message */}
                                  {(!result.annotated_images[selectedImageIndex].damage_areas ||
                                    result.annotated_images[selectedImageIndex].damage_areas.length === 0) && (
                                      <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
                                        <svg className="mx-auto h-12 w-12 text-green-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        <p className="text-green-900 font-medium">No significant damage detected in this image</p>
                                        <p className="text-sm text-green-700 mt-1">All detection scores were below the confidence threshold</p>
                                      </div>
                                    )}
                                </div>
                              </CalculationCard>

                              {/* Technical Details */}
                              <CalculationCard title="Detection Technical Details">
                                <div className="space-y-4">
                                  <div className="grid md:grid-cols-2 gap-4">
                                    <div className="p-4 bg-blue-50 rounded-lg">
                                      <h5 className="font-medium text-blue-900 mb-2">Detection Statistics</h5>
                                      <ul className="text-sm text-blue-800 space-y-1">
                                        <li>• Confidence Threshold: 50%</li>
                                        <li>• Detections Found: {result.annotated_images[selectedImageIndex].total_damage_areas}</li>
                                        <li>• Average Confidence: {(result.annotated_images[selectedImageIndex].average_confidence * 100).toFixed(1)}%</li>
                                        <li>• Model: Mask R-CNN</li>
                                      </ul>
                                    </div>

                                    <div className="p-4 bg-purple-50 rounded-lg">
                                      <h5 className="font-medium text-purple-900 mb-2">Damage Calculation</h5>
                                      <div className="text-sm text-purple-800 space-y-2">
                                        <p className="font-mono bg-white p-2 rounded border">
                                          Damage % = (Σ Area) / Total Pixels × 100
                                        </p>
                                        <p>
                                          = {result.annotated_images[selectedImageIndex].damage_percentage.toFixed(2)}%
                                        </p>
                                      </div>
                                    </div>
                                  </div>

                                  {/* Severity Explanation */}
                                  <div className={`p-4 rounded-lg border-2 ${result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'bg-red-50 border-red-300' :
                                    result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'bg-yellow-50 border-yellow-300' :
                                      'bg-green-50 border-green-300'
                                    }`}>
                                    <h5 className={`font-medium mb-2 ${result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'text-red-900' :
                                      result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'text-yellow-900' :
                                        'text-green-900'
                                      }`}>
                                      Severity Level: {result.annotated_images[selectedImageIndex].severity}
                                    </h5>
                                    <p className={`text-sm ${result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'text-red-800' :
                                      result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'text-yellow-800' :
                                        'text-green-800'
                                      }`}>
                                      {result.annotated_images[selectedImageIndex].severity === 'HIGH'
                                        ? '⚠️ High severity damage detected (>15% coverage). Extensive repairs likely required.'
                                        : result.annotated_images[selectedImageIndex].severity === 'MEDIUM'
                                          ? '⚡ Medium severity damage detected (5-15% coverage). Moderate repairs needed.'
                                          : '✅ Low severity damage detected (<5% coverage). Minor repairs sufficient.'}
                                    </p>
                                  </div>
                                </div>
                              </CalculationCard>
                            </>
                          )}
                        </>
                      ) : (
                        /* No annotated images available */
                        <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
                          <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                          <p className="text-lg font-medium text-gray-900 mb-2">No damage detection data available</p>
                          <p className="text-sm text-gray-500">
                            Damage detection visualization could not be generated for the submitted images.
                          </p>
                        </div>
                      )}
                    </div>
                  )}


                  {/* Fusion Process Tab */}
                  {activeTab === "fusion" && result.detailed_calculations?.fusion_analysis && (
                    <div className="space-y-6">

                      {/* Input Probabilities */}
                      <CalculationCard title="Input Probabilities">
                        <div className="grid md:grid-cols-3 gap-4">
                          <div className="p-4 bg-blue-50 rounded-lg">
                            <h5 className="font-medium text-blue-900 mb-2">Tabular Model</h5>
                            <p className="text-sm">Fraud Probability: <span className="font-bold">{(result.detailed_calculations.fusion_analysis.input_probabilities.tabular_fraud_probability * 100).toFixed(1)}%</span></p>
                            <p className="text-sm">Confidence: <span className="font-bold">{(result.detailed_calculations.fusion_analysis.input_probabilities.tabular_confidence * 100).toFixed(1)}%</span></p>
                          </div>

                          <div className="p-4 bg-purple-50 rounded-lg">
                            <h5 className="font-medium text-purple-900 mb-2">Image Model</h5>
                            <p className="text-sm">Fraud Probability: <span className="font-bold">{(result.detailed_calculations.fusion_analysis.input_probabilities.image_fraud_probability * 100).toFixed(1)}%</span></p>
                            <p className="text-sm">Confidence: <span className="font-bold">{(result.detailed_calculations.fusion_analysis.input_probabilities.image_confidence * 100).toFixed(1)}%</span></p>
                          </div>

                          <div className="p-4 bg-green-50 rounded-lg">
                            <h5 className="font-medium text-green-900 mb-2">Verification Layer</h5>
                            <p className="text-sm">Combined Reliability: <span className="font-bold">{(result.detailed_calculations.fusion_analysis.input_probabilities.verification_reliability * 100).toFixed(1)}%</span></p>
                            <p className="text-sm text-gray-600">Influences overall fraud likelihood</p>
                          </div>
                        </div>
                      </CalculationCard>

                      <CalculationCard title="Weight Calculation">
                        {result.detailed_calculations?.fusion_analysis?.weight_calculation ? (
                          <div className="space-y-4">
                            <div className="p-4 bg-gray-50 border rounded-lg">
                              <h5 className="font-medium text-gray-900 mb-2">Formula</h5>
                              <div className="text-sm font-mono bg-white p-3 rounded border">
                                {result.detailed_calculations.fusion_analysis.weight_calculation.weight_formula || "N/A"}
                              </div>
                            </div>
                            <div className="grid md:grid-cols-4 gap-4">
                              <div className="text-center">
                                <p className="text-sm text-gray-600">Total Confidence</p>
                                <p className="text-lg font-bold">
                                  {result.detailed_calculations.fusion_analysis.weight_calculation.total_confidence?.toFixed(3) || "—"}
                                </p>
                              </div>
                              <div className="text-center">
                                <p className="text-sm text-gray-600">Tabular Weight</p>
                                <p className="text-lg font-bold text-blue-600">
                                  {result.detailed_calculations.fusion_analysis.weight_calculation.tabular_weight?.toFixed(3) || "—"}
                                </p>
                              </div>
                              <div className="text-center">
                                <p className="text-sm text-gray-600">Image Weight</p>
                                <p className="text-lg font-bold text-purple-600">
                                  {result.detailed_calculations.fusion_analysis.weight_calculation.image_weight?.toFixed(3) || "—"}
                                </p>
                              </div>
                              <div className="text-center">
                                <p className="text-sm text-gray-600">Verification γ</p>
                                <p className="text-lg font-bold text-green-600">
                                  {result.detailed_calculations.fusion_analysis.final_fusion?.gamma
                                    ? (result.detailed_calculations.fusion_analysis.final_fusion.gamma * 100).toFixed(0)
                                    : "—"}%
                                </p>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="p-4 text-gray-500 italic">Weight calculation data unavailable.</div>
                        )}
                      </CalculationCard>


                      {/* Fusion Methods */}
                      <CalculationCard title="Fusion Methods Comparison">
                        <div className="space-y-4">
                          {Object.entries(result.detailed_calculations.fusion_analysis.fusion_methods).map(([method, data]: [string, any]) => (
                            <div key={method} className="p-4 border rounded-lg">
                              <div className="flex justify-between items-center mb-2">
                                <h5 className="font-medium text-gray-900 capitalize">{method.replace('_', ' ')}</h5>
                                <span className="text-lg font-bold">{(data.score * 100).toFixed(1)}%</span>
                              </div>
                              <div className="text-sm text-gray-600 font-mono bg-gray-50 p-2 rounded">
                                {data.formula}
                              </div>
                            </div>
                          ))}
                        </div>
                      </CalculationCard>

                      {/* Verification Breakdown */}
                      <CalculationCard title="Verification Reliability Breakdown">
                        <div className="grid md:grid-cols-3 gap-4">
                          <div className="p-4 bg-yellow-50 rounded-lg">
                            <h5 className="font-medium text-yellow-900 mb-2">Driving License (DL)</h5>
                            <p className="text-sm">Valid: <span className="font-bold">{result.detailed_calculations.fusion_analysis.verification_details.dl.valid ? 'Yes' : 'No'}</span></p>
                            <p className="text-sm">DL Score: <span className="font-bold">{(result.detailed_calculations.fusion_analysis.verification_details.dl.dl_score * 100).toFixed(1)}%</span></p>
                          </div>
                          <div className="p-4 bg-blue-50 rounded-lg">
                            <h5 className="font-medium text-blue-900 mb-2">RTO Vehicle Info</h5>
                            <p className="text-sm">Valid: <span className="font-bold">{result.detailed_calculations.fusion_analysis.verification_details.rto.valid ? 'Yes' : 'No'}</span></p>
                            <p className="text-sm">RTO Score: <span className="font-bold">{(result.detailed_calculations.fusion_analysis.verification_details.rto.rto_score * 100).toFixed(1)}%</span></p>
                          </div>
                          <div className="p-4 bg-red-50 rounded-lg">
                            <h5 className="font-medium text-red-900 mb-2">FIR Records</h5>
                            <p className="text-sm">Exists: <span className="font-bold">{result.detailed_calculations.fusion_analysis.verification_details.fir.exists ? 'Yes' : 'No'}</span></p>
                            <p className="text-sm">FIR Score: <span className="font-bold">{(result.detailed_calculations.fusion_analysis.verification_details.fir.fir_score * 100).toFixed(1)}%</span></p>
                          </div>
                        </div>
                      </CalculationCard>

                      {/* Final Fusion */}
                      <CalculationCard title="Final Fusion Result" className="border-2 border-green-200">
                        <div className="space-y-4">
                          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                            <h5 className="font-medium text-green-900 mb-2">Final Calculation</h5>
                            <div className="text-sm font-mono bg-white p-3 rounded border">
                              {result.detailed_calculations.fusion_analysis.final_fusion.calculation}
                            </div>
                            <div className="mt-2 text-sm">
                              Where α = {result.detailed_calculations.fusion_analysis.final_fusion.alpha} (Weighted Avg) <br />
                              β = {result.detailed_calculations.fusion_analysis.final_fusion.beta} (Geometric Mean) <br />
                              γ = {result.detailed_calculations.fusion_analysis.final_fusion.gamma} (Verification Influence)
                            </div>
                          </div>
                          <div className="grid md:grid-cols-3 gap-4">
                            <div className="text-center p-4 bg-yellow-50 rounded-lg">
                              <p className="text-sm text-gray-600">Final Score</p>
                              <p className="text-2xl font-bold">{(result.detailed_calculations.fusion_analysis.final_fusion.final_score * 100).toFixed(1)}%</p>
                            </div>
                            <div className="text-center p-4 bg-blue-50 rounded-lg">
                              <p className="text-sm text-gray-600">Threshold</p>
                              <p className="text-2xl font-bold">{(result.detailed_calculations.fusion_analysis.final_fusion.threshold * 100).toFixed(0)}%</p>
                            </div>
                            <div className={`text-center p-4 rounded-lg ${result.fraud_detected ? 'bg-red-50' : 'bg-green-50'}`}>
                              <p className="text-sm text-gray-600">Decision</p>
                              <p className={`text-2xl font-bold ${result.fraud_detected ? 'text-red-600' : 'text-green-600'}`}>
                                {result.detailed_calculations.fusion_analysis.final_fusion.prediction === 1 ? 'FRAUD' : 'NO FRAUD'}
                              </p>
                            </div>
                          </div>
                        </div>
                      </CalculationCard>
                    </div>
                  )}

                </div>
              </div>
            </div>
          )}

          {!result && (
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="username">
                    Policyholder Username
                  </label>
                  <input
                    id="username"
                    name="username"
                    type="text"
                    value={formData.username}
                    disabled
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-gray-50 text-gray-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="accident_date">
                    Accident Date <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="accident_date"
                    name="accident_date"
                    type="date"
                    value={formData.accident_date}
                    onChange={handleChange}
                    required
                    max={new Date().toISOString().split('T')[0]}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="claim_description">
                  Claim Description <span className="text-red-500">*</span>
                </label>
                <textarea
                  id="claim_description"
                  name="claim_description"
                  value={formData.claim_description}
                  onChange={handleChange}
                  required
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={4}
                  placeholder="Describe the accident, damage, and circumstances in detail..."
                />
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                {/* Vehicle Make */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="vehicle_make">
                    Vehicle Make <span className="text-red-500">*</span>
                  </label>
                  <select
                    id="vehicle_make"
                    name="vehicle_make"
                    value={formData.vehicle_make || ""}
                    onChange={handleChange}
                    required
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Select Vehicle Make</option>
                    <option value="Maruti Suzuki">Maruti Suzuki</option>
                    <option value="Hyundai">Hyundai</option>
                    <option value="Tata">Tata Motors</option>
                    <option value="Mahindra">Mahindra</option>
                    <option value="Honda">Honda</option>
                    <option value="Toyota">Toyota</option>
                    <option value="Kia">Kia</option>
                    <option value="Renault">Renault</option>
                    <option value="Nissan">Nissan</option>
                    <option value="Volkswagen">Volkswagen</option>
                    <option value="Skoda">Skoda</option>
                    <option value="Ford">Ford</option>
                    <option value="Jeep">Jeep</option>
                    <option value="MG">MG Motor</option>
                    <option value="Citroen">Citroen</option>
                    <option value="Other">Other</option>
                  </select>
                  <p className="text-sm text-gray-500 mt-1">
                    {formData.vehicle_make ? '✓ Selected' : 'Required for accurate pricing'}
                  </p>
                </div>

                {/* Vehicle Model */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="vehicle_model">
                    Vehicle Model <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="vehicle_model"
                    name="vehicle_model"
                    type="text"
                    value={formData.vehicle_model || ""}
                    onChange={handleChange}
                    placeholder="e.g. Swift, Creta, Nexon"
                    required
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <p className="text-sm text-gray-500 mt-1">
                    {formData.vehicle_model ? '✓ Entered' : 'Enter specific model name'}
                  </p>
                </div>
              </div>

              {/* --- NEW FIELDS START HERE --- */}
              <div className="grid md:grid-cols-3 gap-6">
                {/* DL Number */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="dl_number">
                    Driving License Number <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="dl_number"
                    name="dl_number"
                    type="text"
                    value={formData.dl_number || ""}
                    onChange={handleChange}
                    placeholder="e.g. MH14 20201234567"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    required
                  />
                  <p className="text-sm text-gray-500 mt-1">Format: State code + Year + Serial (e.g. MH14 20201234567)</p>
                </div>

                {/* Vehicle Registration Number */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="vehicle_reg_no">
                    Vehicle Registration No. <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="vehicle_reg_no"
                    name="vehicle_reg_no"
                    type="text"
                    value={formData.vehicle_reg_no || ""}
                    onChange={handleChange}
                    placeholder="e.g. MH12AB1234"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    required
                  />
                  <p className="text-sm text-gray-500 mt-1">Enter the vehicle’s official RTO number</p>
                </div>

                {/* FIR Number */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="fir_number">
                    FIR / Police Report No.
                  </label>
                  <input
                    id="fir_number"
                    name="fir_number"
                    type="text"
                    value={formData.fir_number || ""}
                    onChange={handleChange}
                    placeholder="e.g. FIR2025-123456"
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <p className="text-sm text-gray-500 mt-1">Optional — if an FIR was registered for the accident</p>
                </div>
              </div>
              {/* --- NEW FIELDS END HERE --- */}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="car_images">
                  Vehicle Damage Images <span className="text-red-500">*</span>
                </label>
                <input
                  id="car_images"
                  name="car_images"
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleImageChange}
                  required
                  className="w-full"
                />

                {imagePreviews.length > 0 && (
                  <div className="mt-4 space-y-4">
                    <p className="text-sm font-medium text-gray-700">
                      {imagePreviews.length} image(s) selected
                    </p>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {imagePreviews.map((preview, index) => (
                        <div key={index} className="relative border rounded-lg overflow-hidden">
                          <img
                            src={preview}
                            alt={`Vehicle damage preview ${index + 1}`}
                            className="w-full h-48 object-cover"
                          />
                          <button
                            type="button"
                            onClick={() => removeImage(index)}
                            className="absolute top-2 right-2 bg-red-600 text-white rounded-full p-1 hover:bg-red-700 transition-colors"
                            title="Remove image"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                          <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-2">
                            Image {index + 1}: {imageFiles[index]?.name}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <p className="text-sm text-gray-500 mt-1">
                  Upload 1-10 clear images showing vehicle damage. Max size per image: 10MB
                </p>
              </div>

              <div className="text-center pt-4">
                <Button
                  type="submit"
                  disabled={loading}
                  className="px-8 py-3 text-lg font-semibold bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? (
                    <div className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Processing Claim...
                    </div>
                  ) : (
                    "Submit Claim for Analysis"
                  )}
                </Button>
              </div>

              {loading && (
                <div className="text-center text-gray-600 bg-blue-50 p-4 rounded-lg">
                  <p className="font-medium">Analyzing your claim...</p>
                  <p className="text-sm mt-1">
                    Our AI is processing the vehicle image, performing damage detection, and calculating fusion scores. This may take a few moments.
                  </p>
                </div>
              )}
            </form>
          )}

          {result && (
            <div className="text-center pt-4">
              <Button
                onClick={() => {
                  setResult(null);
                  setFormData({
                    username: username,
                    claim_description: "",
                    accident_date: "",
                    claim_amount: "",
                    dl_number: "",
                    vehicle_reg_no: "",
                    fir_number: "",
                  });
                  setImageFiles([]);
                  setImagePreviews([]);
                  setActiveTab("results");
                  setSelectedImageIndex(0);
                }}
                variant="outline"
                className="mr-4"
              >
                Submit Another Claim
              </Button>
              <Button
                onClick={() => router.push("/customer")}
                className="bg-green-600 hover:bg-green-700"
              >
                Back to Dashboard
              </Button>
            </div>
          )}
        </div>
      </div>
      {/* Toast Notification - ADD THIS */}
      {toast && (
        <ToastNotification
          type={toast.type}
          message={toast.message}
          claimNumber={toast.claimNumber}
          onClose={() => setToast(null)}
        />
      )}
    </div>


  );
}