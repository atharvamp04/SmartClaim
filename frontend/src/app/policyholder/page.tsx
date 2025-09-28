"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const sections = [
  {
    title: "Personal Information",
    description: "Basic info about yourself and your address",
    imageUrl: "/images/personal_info.jpg",
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

// Exact values from your training data
const dropdownOptions = {
  sex: ["Male", "Female"],
  marital_status: ["Single", "Married", "Divorced", "Widowed"],
  address_area: ["Urban", "Rural"],
  policy_type: [
    "Sedan - All Perils",
    "Sedan - Collision", 
    "Sedan - Liability",
    "Sport - All Perils",
    "Sport - Collision",
    "Sport - Liability"
  ],
  base_policy: ["All Perils", "Collision", "Liability"],
  agent_type: ["Internal", "External"],
  vehicle_make: [
    "Acura", "BMW", "Chevrolet", "Dodge", "Ford", "Honda", "Jeep", 
    "Mazda", "Mercury", "Nisson", "Pontiac", "Saab", "Saturn", 
    "Suburu", "Toyota", "Volkswagen"
  ],
  vehicle_category: ["Sedan", "Sport", "SUV", "Utility"],
  vehicle_price_category: [
    "less than 20000",
    "20000 to 29000", 
    "30000 to 39000",
    "40000 to 59000",
    "60000 to 69000",
    "more than 69000"
  ],
  age_of_vehicle: [
    "new",
    "1 years", 
    "2 years",
    "3 years", 
    "4 years",
    "5 years",
    "6 years", 
    "7 years",
    "more than 7"
  ],
  past_number_of_claims: ["none", "1", "2 to 4", "more than 4"]
};

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
    past_number_of_claims: "none",
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
          past_number_of_claims: formData.past_number_of_claims,
        }),
      });

      if (!res.ok) {
        const data = await res.json();
        setError(JSON.stringify(data));
        setLoading(false);
        return;
      }

      setLoading(false);
      router.push("/");
    } catch (err) {
      setError("Something went wrong.");
      setLoading(false);
    }
  };

  function renderInput(field: string) {
    // Handle dropdown fields
    if (field in dropdownOptions) {
      return (
        <select
          name={field}
          value={formData[field as keyof typeof formData]}
          onChange={handleChange}
          required
          className="w-full border border-gray-300 rounded p-2 focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select {field.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}</option>
          {dropdownOptions[field as keyof typeof dropdownOptions].map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      );
    }

    // Handle special cases
    switch (field) {
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
        return (
          <Input
            name="age"
            type="number"
            value={formData.age}
            onChange={handleChange}
            placeholder="Age"
            className="input"
            min={16}
            max={100}
            required
          />
        );
      case "number_of_cars":
        return (
          <select
            name="number_of_cars"
            value={formData.number_of_cars}
            onChange={handleChange}
            required
            className="w-full border border-gray-300 rounded p-2 focus:ring-2 focus:ring-blue-500"
          >
            <option value={1}>1 vehicle</option>
            <option value={2}>2 vehicles</option>
            <option value={3}>3 to 4 vehicles</option>
            <option value={4}>More than 4 vehicles</option>
          </select>
        );
      case "year_of_vehicle":
        return (
          <Input
            name="year_of_vehicle"
            type="number"
            value={formData.year_of_vehicle}
            onChange={handleChange}
            placeholder="Vehicle Year (e.g., 2020)"
            className="input"
            min={1990}
            max={new Date().getFullYear()}
            required
          />
        );
      case "driver_rating":
        return (
          <select
            name="driver_rating"
            value={formData.driver_rating}
            onChange={handleChange}
            required
            className="w-full border border-gray-300 rounded p-2 focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Select Driver Rating</option>
            <option value={1}>1 - Excellent</option>
            <option value={2}>2 - Good</option>
            <option value={3}>3 - Average</option>
            <option value={4}>4 - Poor</option>
          </select>
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
          <p className="mb-8 font-semibold text-gray-700">{`Step ${currentStep + 1} of ${sections.length}`}</p>
          <img
            src={sections[currentStep].imageUrl}
            alt={sections[currentStep].title}
            className="mb-8 rounded-lg shadow-md max-h-60 object-cover"
          />
          <div className="flex gap-4">
            <Button variant="outline" onClick={handlePrev} disabled={currentStep === 0}>
              Previous
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
                  className="block mb-2 font-semibold text-gray-700 capitalize"
                >
                  {field.replace(/_/g, " ")}
                  {["policy_type", "base_policy", "vehicle_make", "vehicle_category", "age"].includes(field) && 
                    <span className="text-red-500 ml-1">*</span>
                  }
                </label>
                {renderInput(field)}
              </div>
            ))}
            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded">
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            )}
          </form>
        </div>
      </div>
    </div>
  );
}