import type { NextConfig } from "next";

// Backend is only accessible locally on the same machine.
// Next.js server-side rewrites proxy /api/* to it internally.
const BACKEND_URL = "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Allow the frontend to be accessed from the local network IP
  allowedDevOrigins: ["192.168.34.88"],

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
      {
        source: "/media/:path*",
        destination: `${BACKEND_URL}/media/:path*`,
      },
    ];
  },
};

export default nextConfig;
