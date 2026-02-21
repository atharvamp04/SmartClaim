"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/auth/login-with-role/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      console.log("Login response status:", res.status);

      if (!res.ok) {
        const data = await res.json();
        console.error("Login error:", data);
        setError(data.detail || data.error || "Login failed");
        setLoading(false);
        return;
      }

      const data = await res.json();
      console.log("Login successful. Received data:", data);

      // Store auth data
      localStorage.setItem("access_token", data.access);
      localStorage.setItem("username", username);
      localStorage.setItem("user_role", data.role);
      if (data.employee_id) {
        localStorage.setItem("employee_id", data.employee_id);
      }

      console.log("Stored role in localStorage:", data.role);

      setLoading(false);

      // Route based on role (case-insensitive)
      const role = data.role ? data.role.toLowerCase() : '';
      console.log("Normalized role:", role);

      switch (role) {
        case "admin":
          console.log("Pushing to /admin");
          localStorage.setItem("admin_name", username);
          router.push("/admin");
          break;
        case "surveyor":
          console.log("Pushing to /surveyor");
          router.push("/surveyor");
          break;
        default:
          // Regular customer → account page with notifications
          console.log("Pushing to /customer");
          router.push("/customer");
          break;
      }
    } catch (err) {
      console.error("Catch error:", err);
      setError("An error occurred. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50 p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-center text-2xl font-bold">SmartClaim Login</CardTitle>
          <p className="text-center text-sm text-gray-500 mt-1">
            AI-Powered Insurance Claims
          </p>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="username" className="block mb-1 font-medium">Username</label>
              <Input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                placeholder="Enter your username"
              />
            </div>
            <div>
              <label htmlFor="password" className="block mb-1 font-medium">Password</label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="Enter your password"
              />
            </div>
            <Button type="submit" disabled={loading} className="w-full">
              {loading ? "Logging in..." : "Login"}
            </Button>
          </form>
          {error && <p className="mt-4 text-center text-sm text-red-600">{error}</p>}
        </CardContent>
      </Card>
    </div>
  );
}