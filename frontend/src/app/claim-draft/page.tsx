// app/claim-draft/page.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";



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

      const res = await fetch("http://127.0.0.1:8000/api/detection/claims/draft/", {
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
          <CardTitle className="text-2xl">📸 Submit Accident Images (Draft Claim)</CardTitle>
          <p className="text-sm text-gray-600">
            Upload images and basic details. A surveyor will review and complete the full claim.
          </p>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">

            {/* ERROR BOX */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded p-4 text-red-800">
                {error}
              </div>
            )}

            {/* USERNAME */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Username <span className="text-red-600">*</span>
              </label>
              <Input
                name="username"
                value={formData.username}
                onChange={handleChange}
                placeholder="Enter your username"
              />
            </div>

            {/* ACCIDENT DATE */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Accident Date <span className="text-red-600">*</span>
              </label>
              <Input
                type="date"
                name="accident_date"
                value={formData.accident_date}
                onChange={handleChange}
              />
            </div>

            {/* DESCRIPTION */}
            <div>
              <label className="block text-sm font-medium mb-2">
                What happened? <span className="text-red-600">*</span>
              </label>
              <Textarea
                name="claim_description"
                value={formData.claim_description}
                onChange={handleChange}
                rows={4}
                placeholder="Briefly describe the incident..."
              />
            </div>

            {/* IMAGE UPLOAD */}
            <div>
              <label className="block text-sm font-medium mb-2">
                Upload Images <span className="text-red-600">*</span>
              </label>
              <Input type="file" accept="image/*" multiple onChange={handleImageChange} />
            </div>

            {/* IMAGE PREVIEWS */}
            {imagePreviews.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {imagePreviews.map((url, index) => (
                  <div key={index} className="relative border rounded">
                    <img src={url} className="w-full h-40 object-cover rounded" />

                    <button
                      type="button"
                      onClick={() => removeImage(index)}
                      className="absolute top-2 right-2 bg-red-600 text-white rounded-full p-1"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* SUBMIT BUTTON */}
            <Button type="submit" disabled={loading} className="w-full py-3 text-lg">
              {loading ? "Submitting..." : "Submit Draft Claim"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
