"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const sections = [
  {
    title: "Personal Information",
    description: "Basic info about yourself and your address",
    imageUrl: "/images/personal_info.jpg", // replace with your image paths
    fields: [
      "email",
      "username",
      "sex",
      "marital_status",
      "age",
      "address_area",
    ],
  },
  {
    title: "Agent & Policy Details",
    description: "Details about your agent and policy",
    imageUrl: "/images/agent_policy.jpg",
    fields: [
      "policy_type",
      "base_policy",
      "number_of_cars",
      "agent_type",
    ],
  },
  {
    title: "Vehicle Information",
    description: "Details about your vehicle",
    imageUrl: "/images/vehicle_info.jpg",
    fields: [
      "vehicle_make",
      "vehicle_category",
      "vehicle_price_category",
      "age_of_vehicle",
      "year_of_vehicle",
      "driver_rating",
      "past_number_of_claims",
    ],
  },
];

export default function PolicyholderForm() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [formData, setFormData] = useState({
    email: "",
    username: "",
    sex: "",
    marital_status: "",
    age: "",
    address_area: "",
    policy_type: "",
    base_policy: "",
    number_of_cars: 1,
    agent_type: "",
    vehicle_make: "",
    vehicle_category: "",
    vehicle_price_category: "",
    age_of_vehicle: "",
    year_of_vehicle: "",
    driver_rating: "",
    past_number_of_claims: 0,
  });

    useEffect(() => {
    const storedUsername = localStorage.getItem("username") || "";
    const storedEmail = localStorage.getItem("email") || "";
    setFormData((prev) => ({
      ...prev,
      username: storedUsername,
      email: storedEmail,
    }));
  }, []);

  const isLastStep = currentStep === sections.length - 1;

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };
  

  
  const handleNext = () => {
    if (currentStep < sections.length - 1) {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const handlePrev = () => {
    if (currentStep > 0) {
      setCurrentStep((prev) => prev - 1);
    }
  };

  const handleSubmit = async () => {
    setError("");
    setLoading(true);

    try {
      const token = localStorage.getItem("access_token");

      const res = await fetch("http://127.0.0.1:8000/api/detection/policyholder/create/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          ...formData,
          age: Number(formData.age),
          number_of_cars: Number(formData.number_of_cars),
          year_of_vehicle: formData.year_of_vehicle ? Number(formData.year_of_vehicle) : null,
          driver_rating: formData.driver_rating ? Number(formData.driver_rating) : null,
          past_number_of_claims: Number(formData.past_number_of_claims),
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        setError(JSON.stringify(data));
        setLoading(false);
        return;
      }

      setLoading(false);
      router.push("/"); // redirect after successful submit
    } catch (err) {
      setError("Something went wrong.");
      setLoading(false);
    }
  };

  function renderInput(field: string) {
    switch (field) {
      case "sex":
        return (
          <select
            name="sex"
            value={formData.sex}
            onChange={handleChange}
            required
            className="input"
          >
            <option value="">Select Sex</option>
            <option value="Female">Female</option>
            <option value="Male">Male</option>
            <option value="Other">Other</option>
          </select>
        );
      case "marital_status":
      case "address_area":
      case "policy_type":
      case "base_policy":
      case "agent_type":
      case "vehicle_make":
      case "vehicle_category":
      case "vehicle_price_category":
      case "age_of_vehicle":
        return (
          <Input
            name={field}
            value={formData[field as keyof typeof formData]}
            onChange={handleChange}
            placeholder={field.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
            className="input"
            required={
              ["policy_type", "base_policy", "vehicle_make", "vehicle_category"].includes(field)
            }
          />
        );
      case "email":
        return (
          <Input
            name="email"
            type="email"
            value={formData.email}
            onChange={handleChange}
            placeholder="Email"
            className="input"
            required
          />
        );
      case "username":
        return (
          <Input
            name="username"
            value={formData.username}
            onChange={handleChange}
            placeholder="Username"
            className="input"
            required
            readOnly
          />
        );
      case "age":
      case "number_of_cars":
      case "year_of_vehicle":
      case "driver_rating":
      case "past_number_of_claims":
        return (
          <Input
            name={field}
            type="number"
            value={formData[field as keyof typeof formData]}
            onChange={handleChange}
            placeholder={field.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
            className="input"
            min={0}
            required={field === "age" || field === "number_of_cars"}
          />
        );
      default:
        return null;
    }
  }

return (
  <div className="min-h-screen flex items-center justify-center bg-gray-100 p-8">
    <div
      className="bg-white rounded-lg shadow-lg flex max-w-5xl w-full overflow-hidden"
      style={{ height: "650px", minHeight: "650px", maxHeight: "650px" }}
    >
      {/* Left side */}
      <div className="w-1/2 p-8 flex flex-col justify-center items-center border-r border-gray-200">
        <h2 className="text-4xl font-extrabold mb-4">{sections[currentStep].title}</h2>
        <p className="mb-6 text-gray-600 text-center">{sections[currentStep].description}</p>
        <p className="mb-8 font-semibold text-gray-700">{`<${currentStep + 1}/${sections.length}>`}</p>
        <img
          src={sections[currentStep].imageUrl}
          alt={sections[currentStep].title}
          className="mb-8 rounded-lg shadow-md max-h-60 object-cover"
        />
        <div className="flex gap-4">
          <Button variant="outline" onClick={handlePrev} disabled={currentStep === 0}>
            Prev
          </Button>
          {!isLastStep ? (
            <Button onClick={handleNext}>Next</Button>
          ) : (
            <Button onClick={handleSubmit} disabled={loading}>
              {loading ? "Submitting..." : "Submit"}
            </Button>
          )}
        </div>
      </div>

      {/* Right side form */}
      <div
        className="w-1/2 p-8 overflow-y-auto"
        style={{ maxHeight: "650px" }}
      >
        <form className="space-y-4">
          {sections[currentStep].fields.map((field) => (
            <div key={field}>
              <label
                htmlFor={field}
                className="block mb-1 font-semibold capitalize"
              >
                {field.replace(/_/g, " ")}
              </label>
              {renderInput(field)}
            </div>
          ))}
          {error && <p className="mt-4 text-red-600">{error}</p>}
        </form>
      </div>
    </div>
  </div>
);

}
