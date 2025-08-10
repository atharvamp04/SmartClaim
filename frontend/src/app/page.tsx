"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";

interface PolicyholderData {
  email: string;
  username: string;
  sex: string;
  marital_status: string;
  age: number;
  address_area: string;
  policy_type: string;
  base_policy: string;
  number_of_cars: number;
  agent_type: string;
  vehicle_make: string;
  vehicle_category: string;
  vehicle_price_category: string;
  age_of_vehicle: string;
  year_of_vehicle: number | null;
  driver_rating: number | null;
  past_number_of_claims: number;
}

export default function HomePage() {
  const [loading, setLoading] = useState(true);
  const [userData, setUserData] = useState<PolicyholderData | null>(null);
  const [error, setError] = useState("");
  const router = useRouter();

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    const username = localStorage.getItem("username");

    if (!token || !username) {
      router.push("/login");
      return;
    }

    fetchUserData(username, token);
  }, [router]);

  const fetchUserData = async (username: string, token: string) => {
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/detection/policyholders/${username}/`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!res.ok) {
        setError("Failed to fetch user data.");
        setLoading(false);
        return;
      }

      const data = await res.json();
      setUserData(data);
      setLoading(false);
    } catch (err) {
      setError("Something went wrong.");
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("username");
    router.push("/login");
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-r from-blue-400 to-indigo-600 text-white font-semibold text-lg">
        <p>Loading user data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-3xl mx-auto p-8 text-center text-red-600">
        <p className="mb-4">{error}</p>
        <Button onClick={handleLogout} variant="destructive">
          Logout
        </Button>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto p-8 mt-10">
      <h1 className="text-4xl font-extrabold mb-6 text-gray-800 border-b pb-4">
        Welcome, <span className="text-indigo-600">{userData?.username}</span>
      </h1>

      <div className="bg-white rounded-lg shadow-lg p-8">
        <h2 className="text-2xl font-semibold mb-6 text-gray-700 border-b pb-4">
          Your Policyholder Details
        </h2>

        <dl className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
          {userData &&
            Object.entries(userData).map(([key, value]) => {
              const formattedKey = key
                .replace(/_/g, " ")
                .replace(/\b\w/g, (c) => c.toUpperCase());
              const displayValue = value === null || value === "" ? "N/A" : value;

              return (
                <div key={key}>
                  <dt className="text-sm font-medium text-gray-500">{formattedKey}</dt>
                  <dd className="mt-1 text-lg font-semibold text-gray-900">{displayValue}</dd>
                </div>
              );
            })}
        </dl>
      </div>

      <div className="mt-8 text-center">
        <Button onClick={handleLogout} variant="destructive" className="px-8 py-3 text-lg font-semibold">
          Logout
        </Button>
      </div>
    </div>
  );
}
