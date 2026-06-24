// app/claim-draft/page.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { AlertTriangle, Loader2, Upload, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";



export default function ClaimDraftPage() {
  const router = useRouter();

  const [formData, setFormData] = useState({
    username: "",
    accident_date: "",
    claim_description: "",
  });

  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // -------------------------------- Handle Text Inputs ---------------------------------
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  // -------------------------------- Handle Image Upload --------------------------------
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;

    const files = Array.from(e.target.files);

    if (files.length > 10) {
      setError("Maximum 10 images allowed.");
      return;
    }

    setImageFiles(files);

    const previews = files.map((file) => URL.createObjectURL(file));
    setImagePreviews(previews);
    setError("");
  };

  const removeImage = (index: number) => {
    setImageFiles((prev) => prev.filter((_, i) => i !== index));
    setImagePreviews((prev) => prev.filter((_, i) => i !== index));
  };

  // -------------------------------- Form Submit --------------------------------
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!formData.username || !formData.accident_date || !formData.claim_description) {
      setError("Please fill all required fields.");
      return;
    }

    if (imageFiles.length === 0) {
      setError("Please upload at least one image.");
      return;
    }

    setLoading(true);

    try {
      const token = localStorage.getItem("access_token");

      const data = new FormData();
      data.append("username", formData.username);
      data.append("accident_date", formData.accident_date);
      data.append("claim_description", formData.claim_description);

      imageFiles.forEach((file) => data.append("car_images", file));

      const res = await fetch(`/api/detection/claims/draft/`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: data,
      });

      const result = await res.json();

      if (!res.ok) {
        throw new Error(result.error || "Submission failed");
      }

      // SUCCESS TOAST


      router.push("/");
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------- UI --------------------------------
  return (
    <div className="max-w-4xl mx-auto p-8">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Submit Accident Claim</CardTitle>
          <p className="text-sm text-muted-foreground">
            Upload images and basic details. A surveyor will review and complete the full claim.
          </p>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">

            {/* ERROR ALERT */}
            {error && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* USERNAME */}
            <div className="space-y-2">
              <Label htmlFor="username">
                Username <span className="text-red-600">*</span>
              </Label>
              <Input
                id="username"
                name="username"
                value={formData.username}
                onChange={handleChange}
                placeholder="Enter your username"
              />
            </div>

            {/* ACCIDENT DATE */}
            <div className="space-y-2">
              <Label htmlFor="accident_date">
                Accident Date <span className="text-red-600">*</span>
              </Label>
              <Input
                id="accident_date"
                type="date"
                name="accident_date"
                value={formData.accident_date}
                onChange={handleChange}
              />
            </div>

            {/* DESCRIPTION */}
            <div className="space-y-2">
              <Label htmlFor="claim_description">
                What happened? <span className="text-red-600">*</span>
              </Label>
              <Textarea
                id="claim_description"
                name="claim_description"
                value={formData.claim_description}
                onChange={handleChange}
                rows={4}
                placeholder="Briefly describe the incident..."
              />
            </div>

            {/* IMAGE UPLOAD */}
            <div className="space-y-2">
              <Label htmlFor="images">
                Upload Images <span className="text-red-600">*</span>
              </Label>
              <Input 
                id="images"
                type="file" 
                accept="image/*" 
                multiple 
                onChange={handleImageChange}
              />
              <p className="text-xs text-muted-foreground">Maximum 10 images allowed</p>
            </div>

            {/* IMAGE PREVIEWS */}
            {imagePreviews.length > 0 && (
              <div className="space-y-3">
                <p className="text-sm font-medium">Image Previews ({imagePreviews.length})</p>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {imagePreviews.map((url, index) => (
                    <div key={index} className="relative aspect-video rounded-lg border overflow-hidden bg-muted">
                      <img src={url} alt={`Preview ${index}`} className="w-full h-full object-cover" />
                      <Button
                        type="button"
                        variant="destructive"
                        size="icon"
                        className="absolute top-2 right-2 h-7 w-7"
                        onClick={() => removeImage(index)}
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* SUBMIT BUTTON */}
            <Button type="submit" disabled={loading} className="w-full">
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {loading ? "Submitting..." : "Submit Draft Claim"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
