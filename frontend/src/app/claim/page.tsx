"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { AlertTriangle, CheckCircle, XCircle, Loader2, ArrowLeft, X, TrendingUp, Home, DollarSign, AlertCircle } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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
  const [activeTab, setActiveTab] = useState<string>("summary");
  const [selectedImageIndex, setSelectedImageIndex] = useState<number>(0);



  // Handle select dropdown changes
  const handleSelectChange = (value: string) => {
    setFormData((prev) => ({
      ...prev,
      vehicle_make: value,
    }));
  };

  // Handle image uploads
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const files = Array.from(e.target.files);

      if (files.length > 10) {
        setError("Maximum 10 images allowed");
        return;
      }

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

  // Pre-fill vehicle info
  useEffect(() => {
    const fetchPolicyholderData = async () => {
      if (!username) return;

      try {
        const token = localStorage.getItem("access_token");
        const res = await fetch(`http://127.0.0.1:8000/api/policyholders/${username}/`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        });

        if (res.ok) {
          const data = await res.json();
          setFormData(prev => ({
            ...prev,
            vehicle_make: data.vehicle_make || "",
            vehicle_model: data.vehicle_model || ""
          }));
        }
      } catch (error) {
        console.error("Failed to fetch policyholder data:", error);
      }
    };

    fetchPolicyholderData();
  }, [username]);

  // Remove image
  const removeImage = (indexToRemove: number) => {
    setImageFiles(prev => prev.filter((_, index) => index !== indexToRemove));
    setImagePreviews(prev => prev.filter((_, index) => index !== indexToRemove));

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

    // Try to parse as JSON
    let resData;
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

      const res = await fetch("http://127.0.0.1:8000/api/detection/predict-claim/", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: data,
      });

      const responseData = await res.json();

      if (res.ok) {
        setResult(responseData);
      } else {
        setError(responseData.detail || "Failed to submit claim");
      }
    } catch (err: any) {
      setError(err.message || "An error occurred while submitting the claim");
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
    }).format(value);
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'HIGH': return 'border-red-200 bg-red-50';
      case 'MEDIUM': return 'border-amber-200 bg-amber-50';
      case 'LOW': return 'border-green-200 bg-green-50';
      default: return 'border-gray-200 bg-gray-50';
    }
  };

  if (loading && !result) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
            <h2 className="text-lg font-semibold">Processing Claim...</h2>
            <p className="text-sm text-muted-foreground mt-2 text-center">
              Our AI is analyzing vehicle damage and detecting fraud. This may take a few moments.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="bg-primary text-primary-foreground border-b">
        <div className="max-w-7xl mx-auto px-4 lg:px-8 py-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold">Insurance Claim Submission</h1>
              <p className="text-primary-foreground/90 mt-2">Submit your claim with AI-powered fraud detection analysis</p>
            </div>
            <Button
              variant="secondary"
              onClick={() => router.push("/customer")}
            >
              <Home className="h-4 w-4 mr-2" />
              My Account
            </Button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 lg:px-8 py-8">
        {!result ? (
          <Card>
            <CardHeader>
              <CardTitle>Claim Details</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-6">
                {error && (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {/* Username */}
                <div className="space-y-2">
                  <Label htmlFor="username">Username <span className="text-red-600">*</span></Label>
                  <Input
                    id="username"
                    name="username"
                    value={formData.username}
                    onChange={handleChange}
                    placeholder="Enter your username"
                    disabled={!!username}
                  />
                </div>

                {/* Accident Date */}
                <div className="space-y-2">
                  <Label htmlFor="accident_date">Accident Date <span className="text-red-600">*</span></Label>
                  <Input
                    id="accident_date"
                    type="date"
                    name="accident_date"
                    value={formData.accident_date}
                    onChange={handleChange}
                  />
                </div>

                {/* Claim Description */}
                <div className="space-y-2">
                  <Label htmlFor="claim_description">What happened? <span className="text-red-600">*</span></Label>
                  <Textarea
                    id="claim_description"
                    name="claim_description"
                    value={formData.claim_description}
                    onChange={handleChange}
                    rows={4}
                    placeholder="Briefly describe the incident..."
                  />
                </div>

                <Separator />

                {/* Vehicle Information */}
                <div>
                  <h3 className="font-semibold mb-4">Vehicle Information</h3>
                  <div className="grid md:grid-cols-2 gap-6">
                    {/* Vehicle Make */}
                    <div className="space-y-2">
                      <Label htmlFor="vehicle_make">Vehicle Make <span className="text-red-600">*</span></Label>
                      <Select value={formData.vehicle_make} onValueChange={handleSelectChange}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select vehicle make" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="Maruti">Maruti</SelectItem>
                          <SelectItem value="Hyundai">Hyundai</SelectItem>
                          <SelectItem value="Honda">Honda</SelectItem>
                          <SelectItem value="Toyota">Toyota</SelectItem>
                          <SelectItem value="Mahindra">Mahindra</SelectItem>
                          <SelectItem value="Tata">Tata</SelectItem>
                          <SelectItem value="Kia">Kia</SelectItem>
                          <SelectItem value="MG">MG</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Vehicle Model */}
                    <div className="space-y-2">
                      <Label htmlFor="vehicle_model">Vehicle Model <span className="text-red-600">*</span></Label>
                      <Input
                        id="vehicle_model"
                        name="vehicle_model"
                        value={formData.vehicle_model}
                        onChange={handleChange}
                        placeholder="e.g. Swift, City, Fortuner"
                      />
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Documentation */}
                <div>
                  <h3 className="font-semibold mb-4">Documentation</h3>
                  <div className="grid md:grid-cols-3 gap-6">
                    {/* DL Number */}
                    <div className="space-y-2">
                      <Label htmlFor="dl_number">Driving License Number <span className="text-red-600">*</span></Label>
                      <Input
                        id="dl_number"
                        name="dl_number"
                        value={formData.dl_number}
                        onChange={handleChange}
                        placeholder="e.g. MH14 20201234567"
                      />
                      <p className="text-xs text-muted-foreground">Format: State code + Year + Serial</p>
                    </div>

                    {/* Vehicle Registration Number */}
                    <div className="space-y-2">
                      <Label htmlFor="vehicle_reg_no">Vehicle Registration No. <span className="text-red-600">*</span></Label>
                      <Input
                        id="vehicle_reg_no"
                        name="vehicle_reg_no"
                        value={formData.vehicle_reg_no}
                        onChange={handleChange}
                        placeholder="e.g. MH12AB1234"
                      />
                      <p className="text-xs text-muted-foreground">RTO registration number</p>
                    </div>

                    {/* FIR Number */}
                    <div className="space-y-2">
                      <Label htmlFor="fir_number">FIR / Police Report No.</Label>
                      <Input
                        id="fir_number"
                        name="fir_number"
                        value={formData.fir_number}
                        onChange={handleChange}
                        placeholder="e.g. FIR2025-123456"
                      />
                      <p className="text-xs text-muted-foreground">Optional</p>
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Image Upload */}
                <div className="space-y-4">
                  <Label htmlFor="car_images">Vehicle Damage Images <span className="text-red-600">*</span></Label>
                  <Input
                    id="car_images"
                    name="car_images"
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={handleImageChange}
                  />
                  <p className="text-xs text-muted-foreground">Upload 1-10 images (Max 10MB per image)</p>

                  {imagePreviews.length > 0 && (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      {imagePreviews.map((preview, index) => (
                        <div key={index} className="relative rounded-lg overflow-hidden border bg-muted">
                          <img
                            src={preview}
                            alt={`Preview ${index + 1}`}
                            className="w-full h-48 object-cover"
                          />
                          <Button
                            type="button"
                            variant="destructive"
                            size="icon"
                            className="absolute top-2 right-2 h-7 w-7"
                            onClick={() => removeImage(index)}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                          <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-xs p-2">
                            Image {index + 1}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full py-6 text-lg font-semibold"
                >
                  {loading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Processing Claim...
                    </>
                  ) : (
                    "Submit Claim for Analysis"
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        ) : (
          /* Results Display */
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid md:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">Claim Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    {result.fraud_detected ? (
                      <>
                        <XCircle className="h-6 w-6 text-red-600" />
                        <span className="text-lg font-bold text-red-600">FLAGGED</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-6 w-6 text-green-600" />
                        <span className="text-lg font-bold text-green-600">APPROVED</span>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">Risk Level</CardTitle>
                </CardHeader>
                <CardContent>
                  <Badge variant={result.risk_level === 'HIGH' ? 'destructive' : result.risk_level === 'MEDIUM' ? 'secondary' : 'default'}>
                    {result.risk_level}
                  </Badge>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">Confidence</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{(result.confidence * 100).toFixed(1)}%</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm font-medium">Images Analyzed</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{result.total_images_submitted || 1}</p>
                </CardContent>
              </Card>
            </div>

            {/* Claim Amount */}
            {result.calculated_claim_amount && (
              <Card className="border-2 border-green-200 bg-green-50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <DollarSign className="h-5 w-5" />
                    AI-Calculated Claim Amount
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-4xl font-bold text-green-700">
                    {formatCurrency(result.calculated_claim_amount)}
                  </p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Based on YOLO parts detection and damage analysis
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Tabs for detailed analysis */}
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 lg:grid-cols-8 mb-6 gap-1">
                <TabsTrigger value="summary" className="text-xs lg:text-sm">Summary</TabsTrigger>
                <TabsTrigger value="images" className="text-xs lg:text-sm">Images</TabsTrigger>
                <TabsTrigger value="annotated" className="text-xs lg:text-sm">Annotated</TabsTrigger>
                <TabsTrigger value="yolo" className="text-xs lg:text-sm">YOLO</TabsTrigger>
                <TabsTrigger value="breakdown" className="text-xs lg:text-sm">Breakdown</TabsTrigger>
                <TabsTrigger value="fraud" className="text-xs lg:text-sm">Fraud</TabsTrigger>
                <TabsTrigger value="fusion" className="text-xs lg:text-sm">Fusion</TabsTrigger>
                <TabsTrigger value="metrics" className="text-xs lg:text-sm">Metrics</TabsTrigger>
              </TabsList>

              <TabsContent value="summary" className="space-y-4 mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>AI Analysis Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {result.fraud_detected ? (
                      <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription>
                          ⚠️ Fraud indicators detected - {result.risk_level} RISK
                        </AlertDescription>
                      </Alert>
                    ) : (
                      <Alert>
                        <CheckCircle className="h-4 w-4" />
                        <AlertDescription>
                          ✅ Claim appears legitimate - {result.risk_level} RISK
                        </AlertDescription>
                      </Alert>
                    )}

                    <div className="grid md:grid-cols-3 gap-4">
                      <div className="p-4 border rounded-lg bg-blue-50">
                        <h4 className="text-sm font-semibold text-blue-900">Fraud Score</h4>
                        <p className="text-3xl font-bold text-blue-700 mt-2">
                          {(result.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="p-4 border rounded-lg bg-purple-50">
                        <h4 className="text-sm font-semibold text-purple-900">Images</h4>
                        <p className="text-3xl font-bold text-purple-700 mt-2">
                          {result.total_images_submitted}
                        </p>
                      </div>
                      <div className="p-4 border rounded-lg bg-green-50">
                        <h4 className="text-sm font-semibold text-green-900">Detections</h4>
                        <p className="text-3xl font-bold text-green-700 mt-2">
                          {(result.yolo_detection_results?.total_assignments || 0) + (result.yolo_detection_results?.total_parts_detected || 0)}
                        </p>
                      </div>
                    </div>

                    {result.recommended_action && (
                      <Card className="border-2 border-amber-200 bg-amber-50">
                        <CardHeader>
                          <CardTitle className="text-sm">Recommended Action</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-2">
                          <Badge className="bg-amber-600">{result.recommended_action.action}</Badge>
                          <p className="text-sm font-medium">{result.recommended_action.message}</p>
                          {result.recommended_action.next_steps && (
                            <ul className="text-sm list-disc pl-5 text-muted-foreground">
                            {result.recommended_action.next_steps.map((step: any, idx: any) => (
                                <li key={idx}>{step}</li>
                              ))}
                            </ul>
                          )}
                        </CardContent>
                      </Card>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="images" className="space-y-4 mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Original Uploaded Vehicle Images</CardTitle>
                    <p className="text-sm text-muted-foreground mt-2">
                      Total Images Submitted: {imagePreviews.length > 0 ? imagePreviews.length : result.total_images_submitted || 0}
                    </p>
                  </CardHeader>
                  <CardContent>
                    {imagePreviews.length > 0 ? (
                      <div className="space-y-4">
                        <div className="flex gap-2 overflow-x-auto pb-2 mb-4">
                          {imagePreviews.map((preview, idx) => (
                            <button
                              key={idx}
                              onClick={() => setSelectedImageIndex(idx)}
                              className={`flex-shrink-0 h-20 w-20 rounded border-2 overflow-hidden transition ${
                                selectedImageIndex === idx
                                  ? 'border-blue-500 shadow-lg'
                                  : 'border-gray-200 hover:border-gray-400'
                              }`}
                            >
                              <img
                                src={preview}
                                alt={`Thumb ${idx + 1}`}
                                className="w-full h-full object-cover"
                              />
                            </button>
                          ))}
                        </div>

                        <div className="space-y-2">
                          <div className="relative rounded-lg overflow-hidden border-2 border-gray-200 bg-gray-100 flex items-center justify-center min-h-[500px]">
                            <img
                              src={imagePreviews[selectedImageIndex]}
                              alt={`Full Image ${selectedImageIndex + 1}`}
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                            <span className="text-sm font-semibold">
                              Image {selectedImageIndex + 1} of {imagePreviews.length}
                            </span>
                            <div className="flex gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setSelectedImageIndex(Math.max(0, selectedImageIndex - 1))}
                                disabled={selectedImageIndex === 0}
                              >
                                ←
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setSelectedImageIndex(Math.min(imagePreviews.length - 1, selectedImageIndex + 1))}
                                disabled={selectedImageIndex === imagePreviews.length - 1}
                              >
                                →
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-muted-foreground">No original images available in current session</p>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="annotated" className="space-y-4 mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>AI Damage Visualization - Annotated Images</CardTitle>
                    <p className="text-sm text-muted-foreground mt-2">
                      Bounding boxes and damage areas detected by CNN model
                    </p>
                  </CardHeader>
                  <CardContent>
                    {result.annotated_images && result.annotated_images.length > 0 ? (
                      <div className="space-y-4">
                        {result.annotated_images.map((img: any, idx: any) => (
                          <Card key={idx} className="border overflow-hidden">
                            <CardHeader className="pb-3 bg-gradient-to-r from-cyan-50 to-blue-50">
                              <CardTitle className="text-base flex justify-between items-center">
                                <span>Damage Detection - Image {img.image_index || idx + 1}</span>
                                {img.damage_percentage !== undefined && (
                                  <Badge className={
                                    img.damage_percentage > 15 ? 'bg-red-500' : 
                                    img.damage_percentage > 5 ? 'bg-orange-500' :
                                    'bg-green-500'
                                  }>
                                    {img.damage_percentage.toFixed(1)}% Damaged
                                  </Badge>
                                )}
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="pt-4 space-y-3">
                              {/* Annotated Image Display */}
                              <div className="relative rounded-lg overflow-hidden border-2 border-gray-200 bg-gray-100 flex items-center justify-center">
                                {img.annotated_image_base64 ? (
                                  <img
                                    src={`data:image/jpeg;base64,${img.annotated_image_base64}`}
                                    alt={`Annotated ${idx + 1}`}
                                    className="w-full h-auto max-h-[400px] object-contain"
                                  />
                                ) : (
                                  <div className="min-h-[300px] flex items-center justify-center">
                                    <div className="text-center space-y-2">
                                      <p className="text-muted-foreground">CNN Model Image Not Available</p>
                                      <p className="text-xs text-muted-foreground">
                                        (CNN model may not be loaded or no damage detected)
                                      </p>
                                      {/* Debug: Show fields that ARE available */}
                                      {img.damage_percentage !== undefined && (
                                        <div className="text-xs mt-4 p-3 bg-blue-50 rounded">
                                          <p>But found damage data: {img.damage_percentage.toFixed(1)}% damage</p>
                                        </div>
                                      )}
                                    </div>
                                  </div>
                                )}
                              </div>

                              {/* Damage Analysis Details */}
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                <div className="p-3 border rounded-lg bg-blue-50">
                                  <p className="text-xs text-muted-foreground">Damage Areas</p>
                                  <p className="text-lg font-bold text-blue-700">{img.total_damage_areas || 0}</p>
                                </div>
                                <div className="p-3 border rounded-lg bg-red-50">
                                  <p className="text-xs text-muted-foreground">Damage %</p>
                                  <p className="text-lg font-bold text-red-700">{img.damage_percentage?.toFixed(1)}%</p>
                                </div>
                                <div className="p-3 border rounded-lg bg-purple-50">
                                  <p className="text-xs text-muted-foreground">Severity</p>
                                  <p className="text-lg font-bold text-purple-700">{img.severity || 'N/A'}</p>
                                </div>
                                <div className="p-3 border rounded-lg bg-orange-50">
                                  <p className="text-xs text-muted-foreground">Avg Confidence</p>
                                  <p className="text-lg font-bold text-orange-700">{(img.average_confidence * 100).toFixed(1)}%</p>
                                </div>
                              </div>

                              {/* Damage Regions List */}
                              {img.damage_areas && img.damage_areas.length > 0 && (
                                <div>
                                  <h5 className="font-semibold text-sm mb-2">Detected Damage Regions</h5>
                                  <div className="space-y-1 max-h-[200px] overflow-y-auto">
                                    {img.damage_areas.map((region: any, ridx: any) => (
                                      <div key={ridx} className="text-xs p-2 border rounded bg-gray-50 font-mono hover:bg-gray-100">
                                        <div className="flex justify-between items-center">
                                          <span><strong>Region {ridx + 1}:</strong> {(region.confidence * 100).toFixed(1)}%</span>
                                          <span className="text-muted-foreground">{region.area?.toFixed(0)} px²</span>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription className="space-y-2">
                          <p>No annotated damage visualizations available</p>
                          <p className="text-xs text-muted-foreground mt-2">
                            This could be because:
                            <ul className="list-disc list-inside mt-1">
                              <li>CNN damage detection model is not loaded</li>
                              <li>No damage regions detected in the images</li>
                              <li>Check backend logs for model loading status</li>
                            </ul>
                          </p>
                        </AlertDescription>
                      </Alert>
                    )}

                    {/* Fallback: Show damage analysis from multi_image_analysis if annotated images empty */}
                    {(!result.annotated_images || result.annotated_images.length === 0) && 
                     result.multi_image_analysis?.individual_images && result.multi_image_analysis.individual_images.length > 0 && (
                      <Card className="mt-4 border-2 border-blue-200 bg-blue-50">
                        <CardHeader>
                          <CardTitle className="text-base">CNN Damage Analysis (Fallback View)</CardTitle>
                          <p className="text-sm text-muted-foreground mt-1">Detailed damage data without annotated images</p>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3">
                            {result.multi_image_analysis.individual_images.map((img: any, idx: any) => (
                              <div key={idx} className="border rounded-lg p-4 bg-white hover:shadow-md transition">
                                <div className="flex justify-between items-start mb-3">
                                  <h5 className="font-semibold text-sm">
                                    Image {img.image_index || idx + 1}
                                    {img.image_filename && <span className="text-xs text-muted-foreground ml-2">({img.image_filename})</span>}
                                  </h5>
                                  {img.damage_analysis?.damage_percentage !== undefined && (
                                    <Badge className={
                                      img.damage_analysis.damage_percentage > 15 ? 'bg-red-500' : 
                                      img.damage_analysis.damage_percentage > 5 ? 'bg-orange-500' :
                                      'bg-green-500'
                                    }>
                                      {img.damage_analysis.damage_percentage.toFixed(1)}% Damaged
                                    </Badge>
                                  )}
                                </div>

                                {img.damage_analysis && (
                                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 bg-gray-50 p-3 rounded">
                                    <div>
                                      <p className="text-xs text-muted-foreground">Damage %</p>
                                      <p className="text-lg font-bold text-orange-600">
                                        {img.damage_analysis.damage_percentage?.toFixed(1)}%
                                      </p>
                                    </div>
                                    <div>
                                      <p className="text-xs text-muted-foreground">Severity</p>
                                      <p className="text-lg font-bold text-red-600">
                                        {img.damage_analysis.severity_level || 'Unknown'}
                                      </p>
                                    </div>
                                    <div>
                                      <p className="text-xs text-muted-foreground">Severity Score</p>
                                      <p className="text-lg font-bold text-purple-600">
                                        {img.damage_analysis.severity_score?.toFixed(2) || 'N/A'}
                                      </p>
                                    </div>
                                    <div>
                                      <p className="text-xs text-muted-foreground">Weighted Score</p>
                                      <p className="text-lg font-bold text-blue-600">
                                        {img.damage_analysis.weighted_damage_score?.toFixed(2) || 'N/A'}
                                      </p>
                                    </div>
                                  </div>
                                )}

                                {img.detection_results && (
                                  <div className="mt-3 p-2 bg-blue-50 rounded text-sm">
                                    <p className="text-muted-foreground">
                                      Detections: {img.detection_results.total_detections} (High confidence: {img.detection_results.high_confidence_detections})
                                    </p>
                                  </div>
                                )}

                                {img.damage_analysis?.damage_regions && img.damage_analysis.damage_regions.length > 0 && (
                                  <div className="mt-3">
                                    <p className="text-xs font-semibold text-muted-foreground mb-2">
                                      Damage Regions: {img.damage_analysis.damage_regions.length}
                                    </p>
                                    <div className="space-y-1 max-h-[150px] overflow-y-auto text-xs">
                                      {img.damage_analysis.damage_regions.map((region: any, ridx: any) => (
                                        <div key={ridx} className="p-2 bg-gray-50 rounded font-mono">
                                          Region {region.region_id}: {(region.confidence * 100).toFixed(1)}% confidence, 
                                          {(region.relative_size * 100).toFixed(1)}% of image
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="yolo" className="space-y-4 mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>YOLO Detection - Complete Per Image Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {result.yolo_detection_results?.all_images && result.yolo_detection_results.all_images.length > 0 ? (
                      <div className="space-y-6">
                        {result.yolo_detection_results.all_images.map((yoloImg: any, imgIdx: any) => (
                          <Card key={imgIdx} className="border-2 overflow-hidden">
                            <CardHeader className="pb-3 bg-gradient-to-r from-blue-50 to-purple-50">
                              <CardTitle className="text-base flex justify-between items-center">
                                <span>Image {imgIdx + 1}</span>
                                {!yoloImg.detection_successful && <Badge variant="destructive">Detection Failed</Badge>}
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4 pt-4">
                              {yoloImg.detection_successful ? (
                                <>
                                  {/* Parts Detected */}
                                  <div>
                                    <h5 className="font-semibold text-sm mb-3 flex items-center gap-2">
                                      <span className="text-lg">🔧</span>
                                      Parts Detected: {yoloImg.parts_detected?.length || 0}
                                    </h5>
                                    {yoloImg.parts_detected && yoloImg.parts_detected.length > 0 ? (
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-[300px] overflow-y-auto pr-2">
                                        {yoloImg.parts_detected.map((part: any, pidx: any) => (
                                          <div key={pidx} className="text-sm p-3 bg-blue-50 rounded border border-blue-200 hover:bg-blue-100 transition">
                                            <div className="flex justify-between items-start gap-2">
                                              <div className="flex-1">
                                                <Badge variant="outline" className="bg-blue-100 text-blue-900 mb-2">{part.name || 'Unknown'}</Badge>
                                                <div className="text-xs mt-1">
                                                  <p className="text-muted-foreground">Confidence:</p>
                                                  <p className="font-semibold text-blue-700">{(part.confidence * 100).toFixed(1)}%</p>
                                                </div>
                                              </div>
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    ) : (
                                      <p className="text-sm text-muted-foreground">No parts detected</p>
                                    )}
                                  </div>

                                  {/* Damages Detected */}
                                  <Separator />
                                  <div>
                                    <h5 className="font-semibold text-sm mb-3 flex items-center gap-2">
                                      <span className="text-lg">⚠️</span>
                                      Damages Detected: {yoloImg.damages_detected?.length || 0}
                                    </h5>
                                    {yoloImg.damages_detected && yoloImg.damages_detected.length > 0 ? (
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-[300px] overflow-y-auto pr-2">
                                        {yoloImg.damages_detected.map((dmg: any, didx: any) => (
                                          <div key={didx} className="text-sm p-3 bg-red-50 rounded border border-red-200 hover:bg-red-100 transition">
                                            <div className="flex justify-between items-start gap-2">
                                              <div className="flex-1">
                                                <Badge variant="destructive" className="mb-2">{dmg.name || 'Unknown'}</Badge>
                                                <div className="text-xs mt-1">
                                                  <p className="text-muted-foreground">Severity:</p>
                                                  <p className="font-semibold text-red-700">{(dmg.confidence * 100).toFixed(1)}%</p>
                                                </div>
                                              </div>
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    ) : (
                                      <p className="text-sm text-muted-foreground">No damages detected</p>
                                    )}
                                  </div>

                                  {/* Part-to-Damage Assignments */}
                                  {yoloImg.assignments && yoloImg.assignments.length > 0 && (
                                    <>
                                      <Separator />
                                      <div>
                                        <h5 className="font-semibold text-sm mb-3 flex items-center gap-2">
                                          <span className="text-lg">🔗</span>
                                          Part-to-Damage Assignments: {yoloImg.assignments.length}
                                        </h5>
                                        <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                                          {yoloImg.assignments.map((assign: any, aidx: any) => (
                                            <div key={aidx} className="p-3 border rounded bg-white hover:bg-gray-50 transition">
                                              <div className="flex items-center gap-3">
                                                <Badge variant="outline" className="bg-blue-50">{assign.assigned_part || 'UNASSIGNED'}</Badge>
                                                <span className="text-muted-foreground">→</span>
                                                <Badge variant="secondary" className="bg-red-50">{assign.damage_type || 'Unknown'}</Badge>
                                                <Badge className="ml-auto bg-orange-100 text-orange-900">
                                                  {(assign.damage_confidence * 100).toFixed(1)}%
                                                </Badge>
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    </>
                                  )}
                                </>
                              ) : (
                                <Alert variant="destructive">
                                  <AlertTriangle className="h-4 w-4" />
                                  <AlertDescription>
                                    Detection failed for Image {imgIdx + 1}
                                    {yoloImg.error && `: ${yoloImg.error}`}
                                  </AlertDescription>
                                </Alert>
                              )}
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>No YOLO detection data available</AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="breakdown" className="space-y-4 mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Detailed Claim Amount Breakdown - All Items</CardTitle>
                    <p className="text-sm text-muted-foreground mt-2">
                      Total Items: {result.claim_calculation_details?.detailed_breakdown?.length || 0}
                    </p>
                  </CardHeader>
                  <CardContent>
                    {result.claim_calculation_details?.detailed_breakdown && result.claim_calculation_details.detailed_breakdown.length > 0 ? (
                      <div className="space-y-3">
                        {/* Summary Statistics */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4 p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border">
                          <div>
                            <p className="text-xs text-muted-foreground">Base Amount</p>
                            <p className="text-2xl font-bold text-green-700">₹{formatCurrency(result.claim_calculation_details.yolo_base_amount || 0)}</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground">CNN Multiplier</p>
                            <p className="text-2xl font-bold text-blue-700">{(result.claim_calculation_details.cnn_multiplier || 1).toFixed(2)}x</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground">Final Amount</p>
                            <p className="text-2xl font-bold text-purple-700">₹{formatCurrency(result.claim_calculation_details.final_calculated_amount || 0)}</p>
                          </div>
                        </div>

                        {/* All Items - No Limit */}
                        <div className="max-h-[700px] overflow-y-auto pr-2">
                          {result.claim_calculation_details.detailed_breakdown.map((item: any, idx: any) => (
                            <div key={idx} className="border-l-4 border-l-orange-400 rounded-lg p-4 mb-3 bg-white hover:shadow-md transition">
                              {/* Header with Part & Damage */}
                              <div className="flex justify-between items-start mb-3">
                                <div className="flex-1">
                                  <h4 className="font-semibold text-sm flex items-center gap-2">
                                    <Badge variant="outline" className="bg-blue-50">Img {item.image_index}</Badge>
                                    <span>{item.part?.replace(/_/g, ' ') || 'Unknown Part'}</span>
                                  </h4>
                                  <p className="text-xs text-muted-foreground mt-1">Damage: {item.damage_type?.replace(/_/g, ' ') || 'Unknown'}</p>
                                </div>
                                <div className="text-right">
                                  <p className="text-2xl font-bold text-green-700">₹{formatCurrency(item.cost || item.damage_cost || 0)}</p>
                                  <p className="text-xs text-muted-foreground">Total Cost</p>
                                </div>
                              </div>

                              {/* Calculation Details */}
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 bg-gray-50 rounded">
                                <div>
                                  <p className="text-xs text-muted-foreground">Part Price</p>
                                  <p className="font-semibold text-sm">₹{formatCurrency(item.part_price || 0)}</p>
                                </div>
                                <div>
                                  <p className="text-xs text-muted-foreground">Damage %</p>
                                  <p className="font-semibold text-sm">{(item.damage_percentage || 0).toFixed(1)}%</p>
                                </div>
                                <div>
                                  <p className="text-xs text-muted-foreground">Multiplier</p>
                                  <p className="font-semibold text-sm">{(item.severity_multiplier || 1).toFixed(3)}x</p>
                                </div>
                                <div>
                                  <p className="text-xs text-muted-foreground">Confidence</p>
                                  <p className="font-semibold text-sm">{(item.confidence * 100).toFixed(1)}%</p>
                                </div>
                              </div>

                              <p className="text-xs text-muted-foreground mt-2">Source: {item.price_source || 'fallback'}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>No breakdown available</AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="fraud" className="space-y-4 mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Fraud Detection Analysis - Detailed</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="p-4 border rounded-lg bg-red-50">
                        <h5 className="font-semibold text-sm text-red-900">Fraud Probability</h5>
                        <p className="text-3xl font-bold text-red-700 mt-2">
                          {(result.probabilities?.fraud * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="p-4 border rounded-lg bg-green-50">
                        <h5 className="font-semibold text-sm text-green-900">Legitimacy</h5>
                        <p className="text-3xl font-bold text-green-700 mt-2">
                          {(result.probabilities?.no_fraud * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    {result.detailed_calculations?.fusion_analysis?.yolo_claim_calculation && (
                      <Card className="bg-blue-50 border-blue-200">
                        <CardHeader>
                          <CardTitle className="text-base">Claim Amount Analysis</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-2">
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                            <div>
                              <p className="text-xs text-muted-foreground">YOLO Calculated</p>
                              <p className="font-semibold">₹{formatCurrency(result.detailed_calculations.fusion_analysis.yolo_claim_calculation.yolo_calculated_amount || 0)}</p>
                            </div>
                            <div>
                              <p className="text-xs text-muted-foreground">Damaged Parts</p>
                              <p className="font-semibold">{result.detailed_calculations.fusion_analysis.yolo_claim_calculation.total_parts_damaged || 0}</p>
                            </div>
                            <div>
                              <p className="text-xs text-muted-foreground">Unique Parts</p>
                              <p className="font-semibold">{(result.claim_calculation_details?.detailed_breakdown || []).length}</p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    )}

                    {result.detailed_calculations?.fusion_analysis && (
                      <>
                        <Separator />
                        <div>
                          <h5 className="font-semibold text-sm mb-3">Tabular Analysis Probabilities</h5>
                          <div className="space-y-2">
                            {result.detailed_calculations.fusion_analysis.input_probabilities && (
                              <>
                                <div className="p-3 bg-gray-50 rounded border">
                                  <p className="text-xs text-muted-foreground">Tabular Fraud Score</p>
                                  <p className="text-lg font-bold text-red-600">
                                    {(result.detailed_calculations.fusion_analysis.input_probabilities.tabular_fraud_probability * 100).toFixed(1)}%
                                  </p>
                                </div>
                                <div className="p-3 bg-gray-50 rounded border">
                                  <p className="text-xs text-muted-foreground">CNN Fraud Score</p>
                                  <p className="text-lg font-bold text-orange-600">
                                    {(result.detailed_calculations.fusion_analysis.input_probabilities.cnn_fraud_probability * 100).toFixed(1)}%
                                  </p>
                                </div>
                              </>
                            )}
                          </div>
                        </div>
                      </>
                    )}

                    {result.detailed_calculations?.tabular_analysis?.threshold_strategies && (
                      <>
                        <Separator />
                        <div>
                          <h5 className="font-semibold text-sm mb-3">Available Threshold Strategies</h5>
                          <div className="space-y-2 max-h-[300px] overflow-y-auto">
                            {Object.entries(result.detailed_calculations.tabular_analysis.threshold_strategies).map(([strategyName, strategyData]: [string, any]) => (
                              <div key={strategyName} className="p-3 border rounded bg-white hover:bg-gray-50">
                                <div className="flex justify-between items-start mb-2">
                                  <div>
                                    <h6 className="font-semibold text-sm capitalize">{strategyName.replace(/_/g, ' ')}</h6>
                                    <p className="text-xs text-muted-foreground">Threshold: {(strategyData.threshold * 100).toFixed(1)}%</p>
                                  </div>
                                  <Badge variant={strategyData.prediction === 1 ? 'destructive' : 'secondary'}>
                                    {strategyData.prediction === 1 ? 'Fraud' : 'Legitimate'}
                                  </Badge>
                                </div>
                                <div className="grid grid-cols-3 gap-2 text-xs">
                                  <div><span className="text-muted-foreground">Precision:</span> {(strategyData.expected_precision * 100).toFixed(1)}%</div>
                                  <div><span className="text-muted-foreground">Recall:</span> {(strategyData.expected_recall * 100).toFixed(1)}%</div>
                                  <div><span className="text-muted-foreground">F1:</span> {(strategyData.expected_f1 * 100).toFixed(1)}%</div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="fusion" className="space-y-4 mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Fusion Analysis Results</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {result.detailed_calculations?.fusion_analysis ? (
                      <>
                        <div className="p-4 border-2 border-purple-300 rounded-lg bg-gradient-to-r from-blue-50 to-purple-50">
                          <h5 className="font-semibold text-sm mb-2">Final Fusion Score</h5>
                          <p className="text-4xl font-bold text-purple-700">
                            {((result.detailed_calculations.fusion_analysis.final_score || 0) * 100).toFixed(1)}%
                          </p>
                        </div>

                        {result.detailed_calculations.fusion_analysis.component_scores && (
                          <div className="space-y-2">
                            <h5 className="font-semibold text-sm">Component Analysis</h5>
                            {Object.entries(result.detailed_calculations.fusion_analysis.component_scores).map(([key, value]: [string, any]) => (
                              <div key={key} className="flex justify-between items-center p-2 border rounded bg-gray-50">
                                <span className="text-sm capitalize">{key.replace(/_/g, ' ')}</span>
                                <Badge variant="outline">{((value as number) * 100).toFixed(1)}%</Badge>
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    ) : (
                      <p className="text-muted-foreground">No fusion analysis available</p>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="metrics" className="space-y-4 mt-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Multi-Image Metrics & Aggregation - Complete Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {result.multi_image_analysis?.aggregated_metrics ? (
                      <div className="space-y-6">
                        {/* Overall Metrics */}
                        {result.multi_image_analysis.aggregated_metrics.damage_summary && (
                          <div>
                            <h5 className="font-semibold text-sm mb-3">Overall Damage Summary</h5>
                            <div className="grid md:grid-cols-4 gap-3">
                              <div className="p-3 border rounded-lg bg-blue-50">
                                <p className="text-xs text-muted-foreground mb-1">Average Damage</p>
                                <p className="text-2xl font-bold text-blue-700">
                                  {result.multi_image_analysis.aggregated_metrics.damage_summary.avg_damage_percentage?.toFixed(1)}%
                                </p>
                              </div>
                              <div className="p-3 border rounded-lg bg-red-50">
                                <p className="text-xs text-muted-foreground mb-1">Max Damage</p>
                                <p className="text-2xl font-bold text-red-700">
                                  {result.multi_image_analysis.aggregated_metrics.damage_summary.max_damage_percentage?.toFixed(1)}%
                                </p>
                              </div>
                              <div className="p-3 border rounded-lg bg-purple-50">
                                <p className="text-xs text-muted-foreground mb-1">Min Damage</p>
                                <p className="text-2xl font-bold text-purple-700">
                                  {result.multi_image_analysis.aggregated_metrics.damage_summary.min_damage_percentage?.toFixed(1)}%
                                </p>
                              </div>
                              <div className="p-3 border rounded-lg bg-orange-50">
                                <p className="text-xs text-muted-foreground mb-1">Severity</p>
                                <Badge className="mt-1" variant={
                                  result.multi_image_analysis.aggregated_metrics.damage_summary.overall_severity === 'HIGH' ? 'destructive' : 'secondary'
                                }>
                                  {result.multi_image_analysis.aggregated_metrics.damage_summary.overall_severity}
                                </Badge>
                              </div>
                            </div>
                            <div className="grid md:grid-cols-2 gap-3 mt-3">
                              <div className="p-3 border rounded-lg bg-indigo-50">
                                <p className="text-xs text-muted-foreground">Total Detections All Images</p>
                                <p className="text-2xl font-bold text-indigo-700">
                                  {result.multi_image_analysis.aggregated_metrics.damage_summary.total_detections_all_images || 0}
                                </p>
                              </div>
                              <div className="p-3 border rounded-lg bg-cyan-50">
                                <p className="text-xs text-muted-foreground">Total Images Analyzed</p>
                                <p className="text-2xl font-bold text-cyan-700">
                                  {result.multi_image_analysis.aggregated_metrics.total_images || 0}
                                </p>
                              </div>
                            </div>
                          </div>
                        )}

                        <Separator />

                        {/* Fraud Probability Distribution */}
                        {result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution && (
                          <div>
                            <h5 className="font-semibold text-sm mb-3">Fraud Probability Distribution</h5>
                            <div className="grid md:grid-cols-3 gap-3">
                              <div className="p-3 border rounded bg-gray-50">
                                <p className="text-xs text-muted-foreground">Min</p>
                                <p className="text-xl font-bold">{(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.min * 100).toFixed(1)}%</p>
                              </div>
                              <div className="p-3 border rounded bg-gray-50">
                                <p className="text-xs text-muted-foreground">Max</p>
                                <p className="text-xl font-bold">{(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.max * 100).toFixed(1)}%</p>
                              </div>
                              <div className="p-3 border rounded bg-gray-50">
                                <p className="text-xs text-muted-foreground">Mean</p>
                                <p className="text-xl font-bold">{(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.mean * 100).toFixed(1)}%</p>
                              </div>
                              <div className="p-3 border rounded bg-gray-50">
                                <p className="text-xs text-muted-foreground">Median</p>
                                <p className="text-xl font-bold">{(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.median * 100).toFixed(1)}%</p>
                              </div>
                              <div className="p-3 border rounded bg-gray-50">
                                <p className="text-xs text-muted-foreground">Std Dev</p>
                                <p className="text-xl font-bold">{(result.multi_image_analysis.aggregated_metrics.fraud_probability_distribution.std * 100).toFixed(1)}%</p>
                              </div>
                            </div>
                          </div>
                        )}

                        <Separator />

                        {/* Per-Image Detailed Metrics */}
                        {result.multi_image_analysis.individual_images && (
                          <div>
                            <h5 className="font-semibold text-sm mb-3">Per-Image Detailed Analysis</h5>
                            <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
                              {result.multi_image_analysis.individual_images.map((img: any, idx: any) => (
                                <Card key={idx} className="border">
                                  <CardContent className="pt-4 space-y-3">
                                    <div className="flex justify-between items-center mb-2">
                                      <h6 className="font-semibold text-sm">Image {img.image_index} {img.image_filename && `- ${img.image_filename}`}</h6>
                                      <Badge variant="outline">{img.detection_results?.high_confidence_detections || 0} detections</Badge>
                                    </div>

                                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
                                      <div className="p-2 bg-blue-50 rounded">
                                        <p className="text-muted-foreground">Damage %</p>
                                        <p className="font-bold text-blue-700">{img.damage_analysis?.damage_percentage?.toFixed(1)}%</p>
                                      </div>
                                      <div className="p-2 bg-red-50 rounded">
                                        <p className="text-muted-foreground">Severity</p>
                                        <p className="font-bold text-red-700">{img.damage_analysis?.severity_level}</p>
                                      </div>
                                      <div className="p-2 bg-purple-50 rounded">
                                        <p className="text-muted-foreground">Confidence</p>
                                        <p className="font-bold text-purple-700">{(img.image_confidence * 100).toFixed(1)}%</p>
                                      </div>
                                      <div className="p-2 bg-orange-50 rounded">
                                        <p className="text-muted-foreground">Fraud Prob</p>
                                        <p className="font-bold text-orange-700">{(img.image_fraud_probability * 100).toFixed(1)}%</p>
                                      </div>
                                      <div className="p-2 bg-green-50 rounded">
                                        <p className="text-muted-foreground">Area (px)</p>
                                        <p className="font-bold">{img.image_dimensions?.total_pixels?.toLocaleString() || 'N/A'}</p>
                                      </div>
                                      <div className="p-2 bg-indigo-50 rounded">
                                        <p className="text-muted-foreground">Detections</p>
                                        <p className="font-bold text-indigo-700">{img.detection_results?.total_detections || 0}</p>
                                      </div>
                                    </div>
                                  </CardContent>
                                </Card>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <Alert>
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>No aggregated metrics available</AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <Button
                onClick={() => {
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
                  setActiveTab("summary");
                  setSelectedImageIndex(0);
                }}
                variant="outline"
              >
                Submit Another Claim
              </Button>
              <Button
                onClick={() => router.push("/customer")}
                className="ml-auto"
              >
                Back to Dashboard
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
