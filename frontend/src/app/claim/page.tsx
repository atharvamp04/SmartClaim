"use client";

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

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [successMsg, setSuccessMsg] = useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData((prev) => ({
      ...prev,
      [e.target.name]: e.target.value,
    }));
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setImageFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccessMsg("");

    if (!imageFile) {
      setError("Please upload an image of the car.");
      setLoading(false);
      return;
    }

    try {
      const token = localStorage.getItem("access_token");
      if (!token) {
        setError("You must be logged in.");
        setLoading(false);
        return;
      }

      // Use the predict_claim endpoint which handles everything internally
      const data = new FormData();
      data.append("username", formData.username);
      data.append("claim_description", formData.claim_description);
      data.append("accident_date", formData.accident_date);
      data.append("claim_amount", formData.claim_amount);
      data.append("car_image", imageFile); // Note: this should match the backend field name

      console.log("Submitting claim with data:", {
        username: formData.username,
        claim_description: formData.claim_description,
        accident_date: formData.accident_date,
        claim_amount: formData.claim_amount,
        image_name: imageFile.name
      });

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
        setError("Failed to submit claim: " + (resData.error || resData.detail || JSON.stringify(resData)));
        setLoading(false);
        return;
      }

      setSuccessMsg(
        `Claim submitted successfully! 
        Fraud Detection: ${resData.fraud_detected ? 'FRAUD DETECTED' : 'No fraud detected'}
        Confidence: ${(resData.confidence * 100).toFixed(2)}%
        Message: ${resData.message}
        
        Claim Details:
        - Username: ${resData.username}
        - Description: ${resData.claim_description}
        - Accident Date: ${resData.accident_date}
        - Claim Amount: ₹${resData.claim_amount}`
      );
      setLoading(false);
    } catch (err) {
      console.error("Error:", err);
      setError("Something went wrong: " + (err instanceof Error ? err.message : String(err)));
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-8 mt-10 bg-white rounded shadow">
      <h1 className="text-3xl font-bold mb-6">Make a Claim</h1>

      {error && <p className="text-red-600 mb-4">{error}</p>}
      {successMsg && (
        <div className="text-green-600 mb-4 p-4 bg-green-50 rounded">
          <pre className="whitespace-pre-wrap">{successMsg}</pre>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="block font-medium mb-1" htmlFor="username">
            Username
          </label>
          <input
            id="username"
            name="username"
            type="text"
            value={formData.username}
            disabled
            className="w-full border border-gray-300 rounded p-2 bg-gray-100"
          />
        </div>

        <div>
          <label className="block font-medium mb-1" htmlFor="claim_description">
            Claim Description
          </label>
          <textarea
            id="claim_description"
            name="claim_description"
            value={formData.claim_description}
            onChange={handleChange}
            required
            className="w-full border border-gray-300 rounded p-2"
            rows={4}
            placeholder="Describe the accident or damage"
          />
        </div>

        <div>
          <label className="block font-medium mb-1" htmlFor="accident_date">
            Accident Date
          </label>
          <input
            id="accident_date"
            name="accident_date"
            type="date"
            value={formData.accident_date}
            onChange={handleChange}
            required
            className="w-full border border-gray-300 rounded p-2"
          />
        </div>

        <div>
          <label className="block font-medium mb-1" htmlFor="claim_amount">
            Claim Amount (in INR)
          </label>
          <input
            id="claim_amount"
            name="claim_amount"
            type="number"
            min={0}
            value={formData.claim_amount}
            onChange={handleChange}
            required
            className="w-full border border-gray-300 rounded p-2"
            placeholder="Enter estimated claim amount"
          />
        </div>

        <div>
          <label className="block font-medium mb-1" htmlFor="car_image">
            Upload Car Image
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
          {imageFile && (
            <p className="text-sm text-gray-600 mt-1">
              Selected: {imageFile.name} ({(imageFile.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}
        </div>

        <div className="text-center">
          <Button type="submit" disabled={loading} className="px-8 py-3 text-lg font-semibold">
            {loading ? "Processing Claim..." : "Submit Claim"}
          </Button>
        </div>

        {loading && (
          <div className="text-center text-gray-600">
            <p>Please wait while we process your claim and analyze the uploaded image...</p>
          </div>
        )}
      </form>
    </div>
  );
}