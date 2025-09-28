"use client";

//import jsPDF from "jspdf";
//import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import "jspdf-autotable";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";

export default function ClaimPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const username = searchParams.get("username") || "";

  const [formData, setFormData] = useState({
    username: username,
    claim_description: "",
    accident_date: "",
    claim_amount: "",
  });
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>("");
  
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<string>("results");

  // ... (previous form handling code remains the same) ...
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      
      if (file.size > 10 * 1024 * 1024) {
        setError("Image file size must be less than 10MB");
        return;
      }

      if (!file.type.startsWith('image/')) {
        setError("Please upload a valid image file");
        return;
      }

      setImageFile(file);
      setError("");
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
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
    
    if (!imageFile) {
      setError("Please upload an image of the damaged vehicle");
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
      data.append("car_image", imageFile!);

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

const generatePDF = async (formData: {
  username: string;
  accident_date: string;
  claim_amount: string;
  claim_description?: string;
}, result: any) => {
  if (!result) return;

  // Dynamic imports
  const { default: jsPDF } = await import("jspdf");
  await import("jspdf-autotable"); // patch autoTable

  const doc = new jsPDF("p", "pt", "a4");
  const pageWidth = doc.internal.pageSize.getWidth();
  let yOffset = 40;

  // Title
  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  doc.text("Insurance Claim Report", pageWidth / 2, yOffset, { align: "center" });
  yOffset += 30;

  // Policyholder info
  doc.setFontSize(12);
  doc.setFont("helvetica", "normal");
  doc.text(`Policyholder: ${formData.username}`, 40, yOffset);
  yOffset += 20;
  doc.text(`Accident Date: ${formData.accident_date}`, 40, yOffset);
  yOffset += 20;
  doc.text(`Claim Amount: ${formData.claim_amount}`, 40, yOffset);
  yOffset += 30;

  // Fraud detection summary
  doc.setFont("helvetica", "bold");
  doc.text("Fraud Detection Summary", 40, yOffset);
  doc.setFont("helvetica", "normal");
  yOffset += 20;
  doc.text(`Fraud Detected: ${result.fraud_detected ? "YES" : "NO"}`, 40, yOffset);
  yOffset += 20;
  doc.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, 40, yOffset);
  yOffset += 30;

  // Damage image (if available)
  if (result.damage_detection?.annotated_image_base64) {
    const imgData = "data:image/jpeg;base64," + result.damage_detection.annotated_image_base64;
    try {
      const imgProps = await new Promise<HTMLImageElement>((resolve, reject) => {
        const img = new Image();
        img.src = imgData;
        img.onload = () => resolve(img);
        img.onerror = reject;
      });

      const imgWidth = pageWidth - 80; // margin
      const scale = imgWidth / imgProps.width;
      const imgHeight = imgProps.height * scale;
      doc.addImage(imgData, "JPEG", 40, yOffset, imgWidth, imgHeight);
      yOffset += imgHeight + 20;
    } catch (err) {
      console.error("Error loading image for PDF:", err);
    }
  }

  // Tabular data
  doc.setFont("helvetica", "bold");
  doc.text("Tabular Analysis", 40, yOffset);
  yOffset += 20;
  doc.setFont("helvetica", "normal");

  const tableData = result.detailed_calculations?.tabular_analysis?.individual_model_predictions || [];
  if (tableData.length > 0) {
    doc.autoTable({
      startY: yOffset,
      head: [["Model Name", "No Fraud %", "Fraud %"]],
      body: tableData.map((m: any) => [
        m.model_name,
        (m.no_fraud_probability * 100).toFixed(1),
        (m.fraud_probability * 100).toFixed(1),
      ]),
      theme: "grid",
      headStyles: { fillColor: [30, 144, 255] },
      styles: { fontSize: 10 },
      margin: { left: 40, right: 40 },
    });
  }

  // Save PDF
  doc.save(`Claim_Report_${formData.username}.pdf`);
};




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

                    {/* Damage Analysis */}
                    {result.damage_detection && (
                      <div className={`p-4 rounded-lg border ${getRiskColor(result.damage_detection.severity)}`}>
                        <div className="text-center">
                          <p className="font-semibold">Damage Severity</p>
                          <p className="text-2xl font-bold mt-1">{result.damage_detection.severity}</p>
                          <p className="text-sm mt-1">{result.damage_detection.total_damage_areas} areas detected</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Detailed Analysis Tabs */}
              <div className="bg-white border rounded-lg shadow-sm">
                <div className="px-6 py-4 border-b border-gray-200">
                  <div className="flex space-x-2">
                    <TabButton tabKey="results" label="Summary" isActive={activeTab === "results"} onClick={setActiveTab} />
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
                                {result.detailed_calculations?.tabular_analysis?.ensemble_probabilities?.fraud ? 
                                  (result.detailed_calculations.tabular_analysis.ensemble_probabilities.fraud * 100).toFixed(1) + '%' : 'N/A'}
                              </span>
                            </div>
                            <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                              <span className="text-sm font-medium">Image Analysis</span>
                              <span className="text-sm">
                                {result.detailed_calculations?.image_analysis?.image_fraud_probability ? 
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

                  {/* Tabular Analysis Tab */}
                  {activeTab === "tabular" && result.detailed_calculations?.tabular_analysis && (
                    <div className="space-y-6">
                      <CalculationCard title="Feature Processing">
                        <div className="space-y-4">
                          <div className="grid md:grid-cols-2 gap-4">
                            <div>
                              <h5 className="font-medium text-gray-700 mb-2">Raw Features</h5>
                              <p className="text-sm text-gray-600">Shape: {result.detailed_calculations.tabular_analysis.raw_features_shape?.join(' × ')}</p>
                              <div className="mt-2 p-3 bg-gray-50 rounded text-xs font-mono">
                                Sample: [{result.detailed_calculations.tabular_analysis.raw_features_sample?.map(f => f.toFixed(3)).join(', ')}...]
                              </div>
                            </div>
                            <div>
                              <h5 className="font-medium text-gray-700 mb-2">CNN Features</h5>
                              <p className="text-sm text-gray-600">Shape: {result.detailed_calculations.tabular_analysis.cnn_features_shape?.join(' × ')}</p>
                              <div className="mt-2 p-3 bg-gray-50 rounded text-xs font-mono">
                                Sample: [{result.detailed_calculations.tabular_analysis.cnn_features_sample?.map(f => f.toFixed(3)).join(', ')}...]
                              </div>
                            </div>
                          </div>
                        </div>
                      </CalculationCard>

                      <CalculationCard title="Ensemble Model Predictions">
                        <div className="space-y-4">
                          <div className="grid md:grid-cols-2 gap-4">
                            <div className="p-4 bg-blue-50 rounded-lg">
                              <h5 className="font-medium text-blue-900">No Fraud Probability</h5>
                              <p className="text-2xl font-bold text-blue-600">
                                {(result.detailed_calculations.tabular_analysis.ensemble_probabilities.no_fraud * 100).toFixed(1)}%
                              </p>
                            </div>
                            <div className="p-4 bg-red-50 rounded-lg">
                              <h5 className="font-medium text-red-900">Fraud Probability</h5>
                              <p className="text-2xl font-bold text-red-600">
                                {(result.detailed_calculations.tabular_analysis.ensemble_probabilities.fraud * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                          
                          {result.detailed_calculations.tabular_analysis.individual_model_predictions?.length > 0 && (
                            <div>
                              <h5 className="font-medium text-gray-700 mb-3">Individual Model Contributions</h5>
                              <div className="space-y-2">
                                {result.detailed_calculations.tabular_analysis.individual_model_predictions.map((model, idx) => (
                                  <div key={idx} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                                    <span className="font-medium">{model.model_name}</span>
                                    <div className="text-sm">
                                      <span className="text-green-600">{(model.no_fraud_probability * 100).toFixed(1)}% No Fraud</span>
                                      {' | '}
                                      <span className="text-red-600">{(model.fraud_probability * 100).toFixed(1)}% Fraud</span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </CalculationCard>
                    </div>
                  )}

                  {/* Image Analysis Tab */}
                  {activeTab === "image" && result.detailed_calculations?.image_analysis && (
                    <div className="space-y-6">
                      <CalculationCard title="Image Processing">
                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <h5 className="font-medium text-gray-700 mb-2">Image Properties</h5>
                            <ul className="text-sm text-gray-600 space-y-1">
                              <li>Width: {result.detailed_calculations.image_analysis.image_dimensions.width}px</li>
                              <li>Height: {result.detailed_calculations.image_analysis.image_dimensions.height}px</li>
                              <li>Total Pixels: {result.detailed_calculations.image_analysis.image_dimensions.total_pixels.toLocaleString()}</li>
                            </ul>
                          </div>
                          <div>
                            <h5 className="font-medium text-gray-700 mb-2">Detection Results</h5>
                            <ul className="text-sm text-gray-600 space-y-1">
                              <li>Total Detections: {result.detailed_calculations.image_analysis.detection_results.total_detections}</li>
                              <li>High Confidence: {result.detailed_calculations.image_analysis.detection_results.high_confidence_detections}</li>
                              <li>Threshold: {result.detailed_calculations.image_analysis.detection_results.confidence_threshold}</li>
                            </ul>
                          </div>
                        </div>
                      </CalculationCard>

                      <CalculationCard title="Damage Analysis Calculation">
                        <div className="space-y-4">
                          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                            <h5 className="font-medium text-yellow-900 mb-2">Damage Coverage Formula</h5>
                            <div className="text-sm font-mono bg-white p-3 rounded border">
                              Damage % = (Total Damage Area ÷ Total Image Area) × 100
                            </div>
                            <div className="mt-2 text-sm">
                              = ({result.detailed_calculations.image_analysis.damage_analysis.total_damage_area_pixels.toFixed(0)} ÷ {result.detailed_calculations.image_analysis.image_dimensions.total_pixels.toLocaleString()}) × 100
                              = {result.detailed_calculations.image_analysis.damage_analysis.damage_percentage.toFixed(2)}%
                            </div>
                          </div>

                          <div className="grid md:grid-cols-3 gap-4">
                            <div className="p-4 bg-red-50 rounded-lg text-center">
                              <h5 className="font-medium text-red-900">Damage Coverage</h5>
                              <p className="text-2xl font-bold text-red-600">
                                {result.detailed_calculations.image_analysis.damage_analysis.damage_percentage.toFixed(2)}%
                              </p>
                            </div>
                            <div className="p-4 bg-orange-50 rounded-lg text-center">
                              <h5 className="font-medium text-orange-900">Severity Score</h5>
                              <p className="text-2xl font-bold text-orange-600">
                                {(result.detailed_calculations.image_analysis.damage_analysis.severity_score * 100).toFixed(1)}%
                              </p>
                            </div>
                            <div className="p-4 bg-purple-50 rounded-lg text-center">
                              <h5 className="font-medium text-purple-900">Weighted Score</h5>
                              <p className="text-2xl font-bold text-purple-600">
                                {(result.detailed_calculations.image_analysis.damage_analysis.weighted_damage_score * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>

                          {result.detailed_calculations.image_analysis.damage_analysis.damage_regions?.length > 0 && (
                            <div>
                              <h5 className="font-medium text-gray-700 mb-3">Individual Damage Regions</h5>
                              <div className="space-y-2">
                                {result.detailed_calculations.image_analysis.damage_analysis.damage_regions.map((region, idx) => (
                                  <div key={idx} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                                    <span className="font-medium">Region {region.region_id}</span>
                                    <div className="text-sm">
                                      <span>Area: {region.area_pixels.toFixed(0)}px² </span>
                                      <span>({(region.relative_size * 100).toFixed(2)}%) </span>
                                      <span className="text-blue-600">Conf: {(region.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </CalculationCard>
                    </div>
                  )}

                  {/* Fusion Process Tab */}
                  {activeTab === "fusion" && result.detailed_calculations?.fusion_analysis && (
                    <div className="space-y-6">
                      <CalculationCard title="Input Probabilities">
                        <div className="grid md:grid-cols-2 gap-4">
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
                        </div>
                      </CalculationCard>

                      <CalculationCard title="Weight Calculation">
                        <div className="space-y-4">
                          <div className="p-4 bg-gray-50 border rounded-lg">
                            <h5 className="font-medium text-gray-900 mb-2">Formula</h5>
                            <div className="text-sm font-mono bg-white p-3 rounded border">
                              {result.detailed_calculations.fusion_analysis.weight_calculation.weight_formula}
                            </div>
                          </div>
                          <div className="grid md:grid-cols-3 gap-4">
                            <div className="text-center">
                              <p className="text-sm text-gray-600">Total Confidence</p>
                              <p className="text-lg font-bold">{result.detailed_calculations.fusion_analysis.weight_calculation.total_confidence.toFixed(3)}</p>
                            </div>
                            <div className="text-center">
                              <p className="text-sm text-gray-600">Tabular Weight</p>
                              <p className="text-lg font-bold text-blue-600">{result.detailed_calculations.fusion_analysis.weight_calculation.tabular_weight.toFixed(3)}</p>
                            </div>
                            <div className="text-center">
                              <p className="text-sm text-gray-600">Image Weight</p>
                              <p className="text-lg font-bold text-purple-600">{result.detailed_calculations.fusion_analysis.weight_calculation.image_weight.toFixed(3)}</p>
                            </div>
                          </div>
                        </div>
                      </CalculationCard>

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

                      <CalculationCard title="Final Fusion Result" className="border-2 border-green-200">
                        <div className="space-y-4">
                          <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                            <h5 className="font-medium text-green-900 mb-2">Final Calculation</h5>
                            <div className="text-sm font-mono bg-white p-3 rounded border">
                              {result.detailed_calculations.fusion_analysis.final_fusion.calculation}
                            </div>
                            <div className="mt-2 text-sm">
                              Where α = {result.detailed_calculations.fusion_analysis.final_fusion.alpha} (weighted average weight)
                              <br />
                              β = {result.detailed_calculations.fusion_analysis.final_fusion.beta} (geometric mean weight)
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

                  {/* Damage Detection Tab */}
                  {activeTab === "damage" && result.damage_detection && (
                    <div className="space-y-6">
                      <CalculationCard title="Visual Damage Analysis">
                        {result.damage_detection.annotated_image_base64 ? (
                          <div className="space-y-4">
                            <div className="border rounded-lg overflow-hidden">
                              <img 
                                src={`data:image/jpeg;base64,${result.damage_detection.annotated_image_base64}`}
                                alt="Damage Detection Results"
                                className="w-full h-auto"
                              />
                            </div>
                            
                            <div className="grid md:grid-cols-2 gap-4">
                              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                                <h5 className="font-medium text-blue-900 mb-2">Detection Summary</h5>
                                <ul className="text-sm text-blue-800 space-y-1">
                                  <li>• {result.damage_detection.total_damage_areas} damage area(s) detected</li>
                                  <li>• {result.damage_detection.damage_percentage.toFixed(2)}% of vehicle surface affected</li>
                                  <li>• Damage severity: {result.damage_detection.severity}</li>
                                  <li>• Image dimensions: {result.damage_detection.original_dimensions.width} × {result.damage_detection.original_dimensions.height}px</li>
                                </ul>
                              </div>

                              {result.damage_detection.damage_areas && result.damage_detection.damage_areas.length > 0 && (
                                <div className="bg-gray-50 rounded-lg p-4">
                                  <h5 className="font-medium text-gray-900 mb-3">Detected Damage Areas</h5>
                                  <div className="space-y-2">
                                    {result.damage_detection.damage_areas.map((area, index) => (
                                      <div key={index} className="flex justify-between items-center p-2 bg-white rounded border text-sm">
                                        <span className="font-medium">Area {index + 1}</span>
                                        <span className="text-gray-600">
                                          Confidence: {(area.confidence * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        ) : (
                          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                            <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 14v20c0 4.418 7.163 8 16 8 1.381 0 2.721-.087 4-.252M8 14c0 4.418 7.163 8 16 8s16-3.582 16-8M8 14c0-4.418 7.163-8 16-8s16 3.582 16 8m0 0v14m-16-4L24 30l8-8" />
                            </svg>
                            <p className="mt-2 text-sm text-gray-500">
                              Damage detection visualization not available
                            </p>
                          </div>
                        )}
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
                <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="car_image">
                  Vehicle Damage Image <span className="text-red-500">*</span>
                </label>
                <input
                  id="car_image"
                  name="car_image"
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  required
                  className="w-full"
                />
                
                {imagePreview && (
                  <div className="mt-4 border rounded-lg overflow-hidden max-w-md">
                    <img 
                      src={imagePreview}
                      alt="Vehicle damage preview"
                      className="w-full h-auto"
                    />
                  </div>
                )}
                
                {imageFile && (
                  <div className="mt-2 p-3 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-sm text-green-800">
                      📸 Selected: {imageFile.name} ({(imageFile.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                  </div>
                )}
                <p className="text-sm text-gray-500 mt-1">
                  Upload a clear image showing vehicle damage. Max size: 10MB
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
              <button onClick={() => generatePDF(formData, result)} className="btn btn-primary">
                 Generate PDF
              </button>   
              <Button 
                onClick={() => {
                  setResult(null);
                  setFormData({
                    username: username,
                    claim_description: "",
                    accident_date: "",
                    claim_amount: "",
                  });
                  setImageFile(null);
                  setImagePreview("");
                  setActiveTab("results");
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
    </div>
  );
}
