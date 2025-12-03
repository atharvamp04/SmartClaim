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

  const [formData, setFormData] = useState({
    username: username,
    claim_description: "",
    accident_date: "",
    claim_amount: "",
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



  
  // ... (previous form handling code remains the same) ...
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

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

const removeImage = (indexToRemove: number) => {
  setImageFiles(prev => prev.filter((_, index) => index !== indexToRemove));
  setImagePreviews(prev => prev.filter((_, index) => index !== indexToRemove));
  
  // Adjust selected index if needed
  if (selectedImageIndex >= indexToRemove && selectedImageIndex > 0) {
    setSelectedImageIndex(selectedImageIndex - 1);
  }
};

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
    
    if (!formData.claim_amount || parseFloat(formData.claim_amount) <= 0) {
      setError("Valid claim amount is required");
      return false;
    }
    
    const claimAmount = parseFloat(formData.claim_amount);
    if (claimAmount < 1000) {
      setError("Claim amount must be at least ₹1,000");
      return false;
    }
    
    if (claimAmount > 10000000) {
      setError("Claim amount cannot exceed ₹1,00,00,000");
      return false;
    }
    
// Replace the imageFile validation with:
  if (imageFiles.length === 0) {
    setError("Please upload at least one image of the damaged vehicle");
    return false;
    }
    
    return true;
  };

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
      data.append("claim_amount", formData.claim_amount);
      imageFiles.forEach((file) => {data.append("car_images", file);});
      data.append("dl_number", formData.dl_number.trim());
      data.append("vehicle_reg_no", formData.vehicle_reg_no.trim());
      data.append("fir_number", formData.fir_number.trim());

      const res = await fetch("http://127.0.0.1:8000/api/detection/predict-claim/", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: data,
      });

      const resData = await res.json();
      console.log("Response from server:", resData);

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
      console.error("Error:", err);
      setError(`Network error: ${err instanceof Error ? err.message : 'Something went wrong'}`);
      setLoading(false);
    }
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
      className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
        isActive
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
      <h1 className="text-3xl font-bold">Insurance Claim Submission</h1>
      <p className="text-blue-100 mt-2">Submit your claim with AI-powered fraud detection analysis</p>
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
          {/* Results Summary */}
          <div className="bg-white border rounded-lg shadow-sm mb-6">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Fraud Detection Results</h3>
              {result.multi_image_analysis && (
                <p className="text-sm text-gray-600 mt-1">
                  Analysis based on {result.total_images_submitted || result.multi_image_analysis.aggregated_metrics?.total_images || 1} image(s)
                </p>
              )}
            </div>
            
            <div className="p-6">
              <div className="grid md:grid-cols-3 gap-6 mb-6">
                {/* Final Result */}
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

                {/* Damage/Multi-Image Summary */}
                {result.multi_image_analysis?.aggregated_metrics ? (
                  <div className={`p-4 rounded-lg border ${getRiskColor(result.multi_image_analysis.aggregated_metrics.severity_analysis.overall_severity)}`}>
                    <div className="text-center">
                      <p className="font-semibold">Overall Damage</p>
                      <p className="text-2xl font-bold mt-1">{result.multi_image_analysis.aggregated_metrics.severity_analysis.overall_severity}</p>
                      <p className="text-sm mt-1">
                        Avg: {result.multi_image_analysis.aggregated_metrics.damage_summary.avg_damage_percentage.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                ) : result.damage_detection ? (
                  <div className={`p-4 rounded-lg border ${getRiskColor(result.damage_detection.severity)}`}>
                    <div className="text-center">
                      <p className="font-semibold">Damage Severity</p>
                      <p className="text-2xl font-bold mt-1">{result.damage_detection.severity}</p>
                      <p className="text-sm mt-1">{result.damage_detection.total_damage_areas} areas detected</p>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          {/* Detailed Analysis Tabs */}
          <div className="bg-white border rounded-lg shadow-sm">
            <div className="px-6 py-4 border-b border-gray-200">
              <div className="flex space-x-2 overflow-x-auto">
                <TabButton tabKey="results" label="Summary" isActive={activeTab === "results"} onClick={setActiveTab} />
                {result.multi_image_analysis && (
                  <TabButton 
                    tabKey="multi-images" 
                    label={`Images (${result.total_images_submitted || result.multi_image_analysis?.aggregated_metrics?.total_images || 1})`} 
                    isActive={activeTab === "multi-images"} 
                    onClick={setActiveTab} 
                  />
                )}
                <TabButton tabKey="tabular" label="Tabular Analysis" isActive={activeTab === "tabular"} onClick={setActiveTab} />
                <TabButton tabKey="image" label="Image Analysis" isActive={activeTab === "image"} onClick={setActiveTab} />
                <TabButton tabKey="fusion" label="Fusion Process" isActive={activeTab === "fusion"} onClick={setActiveTab} />
                <TabButton tabKey="damage" label="Damage Detection" isActive={activeTab === "damage"} onClick={setActiveTab} />
              </div>
            </div>

            <div className="p-6">
              {/* Summary Tab */}
              {activeTab === "results" && (
                <div className="space-y-6">
                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Claim Information</h4>
                      <dl className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <dt className="text-gray-600">Claim Amount:</dt>
                          <dd className="font-medium">{formatCurrency(result.claim_amount)}</dd>
                        </div>
                        {result.multi_image_analysis && (
                          <div className="flex justify-between">
                            <dt className="text-gray-600">Images Submitted:</dt>
                            <dd className="font-medium">{result.total_images_submitted || result.multi_image_analysis.aggregated_metrics?.total_images || 1}</dd>
                          </div>
                        )}
                        <div className="flex justify-between">
                          <dt className="text-gray-600">Final Score:</dt>
                          <dd className="font-medium">{(result.confidence * 100).toFixed(1)}%</dd>
                        </div>
                      </dl>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-gray-900 mb-3">Analysis Summary</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                          <span className="text-sm font-medium">Tabular Analysis</span>
                          <span className="text-sm">
                            {result.detailed_calculations?.tabular_analysis?.probabilities?.fraud ? 
                              (result.detailed_calculations.tabular_analysis.probabilities.fraud * 100).toFixed(1) + '%' : 
                              result.detailed_calculations?.tabular_analysis?.ensemble_probabilities?.fraud ?
                              (result.detailed_calculations.tabular_analysis.ensemble_probabilities.fraud * 100).toFixed(1) + '%' :
                              'N/A'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                          <span className="text-sm font-medium">Image Analysis</span>
                          <span className="text-sm">
                            {result.multi_image_analysis?.aggregated_metrics?.final_image_fraud_probability !== undefined ? 
                              (result.multi_image_analysis.aggregated_metrics.final_image_fraud_probability * 100).toFixed(1) + '%' :
                              result.detailed_calculations?.image_analysis?.image_fraud_probability !== undefined ? 
                              (result.detailed_calculations.image_analysis.image_fraud_probability * 100).toFixed(1) + '%' : 'N/A'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                          <span className="text-sm font-medium">Fusion Result</span>
                          <span className="text-sm font-semibold">
                            {(result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
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
                              className={`px-4 py-2 rounded-lg border-2 transition-all whitespace-nowrap ${
                                selectedImageIndex === idx
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
                                  <span className={`ml-2 font-semibold ${
                                    result.multi_image_analysis.individual_images[selectedImageIndex].damage_analysis.severity_level === 'HIGH' ? 'text-red-600' :
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
          <div className={`p-3 rounded-lg text-center ${
            result.detailed_calculations.tabular_analysis.primary_prediction.fraud_detected 
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
                  className={`relative flex-shrink-0 w-32 h-32 rounded-lg border-2 transition-all ${
                    selectedImageIndex === idx
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
                      <div className={`absolute top-2 right-2 px-2 py-1 rounded text-xs font-bold ${
                        img.severity === 'HIGH' ? 'bg-red-500 text-white' :
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
                  
                  <div className={`border-2 rounded-lg p-4 text-center ${
                    result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'bg-red-50 border-red-300' :
                    result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'bg-yellow-50 border-yellow-300' :
                    'bg-green-50 border-green-300'
                  }`}>
                    <h5 className={`text-sm font-medium mb-1 ${
                      result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'text-red-900' :
                      result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'text-yellow-900' :
                      'text-green-900'
                    }`}>Severity</h5>
                    <p className={`text-3xl font-bold ${
                      result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'text-red-700' :
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
                <div className={`p-4 rounded-lg border-2 ${
                  result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'bg-red-50 border-red-300' :
                  result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'bg-yellow-50 border-yellow-300' :
                  'bg-green-50 border-green-300'
                }`}>
                  <h5 className={`font-medium mb-2 ${
                    result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'text-red-900' :
                    result.annotated_images[selectedImageIndex].severity === 'MEDIUM' ? 'text-yellow-900' :
                    'text-green-900'
                  }`}>
                    Severity Level: {result.annotated_images[selectedImageIndex].severity}
                  </h5>
                  <p className={`text-sm ${
                    result.annotated_images[selectedImageIndex].severity === 'HIGH' ? 'text-red-800' :
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
                <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="claim_amount">
                  Estimated Claim Amount (₹) <span className="text-red-500">*</span>
                </label>
                <input
                  id="claim_amount"
                  name="claim_amount"
                  type="number"
                  min={1000}
                  max={10000000}
                  step={100}
                  value={formData.claim_amount}
                  onChange={handleChange}
                  required
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter estimated repair/replacement cost"
                />
                <p className="text-sm text-gray-500 mt-1">
                  Minimum: ₹1,000 | Maximum: ₹1,00,00,000
                </p>
              </div>

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
                onClick={() => router.push("/")}
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