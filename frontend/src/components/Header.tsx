"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function Header() {
  const [username, setUsername] = useState<string | null>(null);
  const [isClient, setIsClient] = useState(false);
  const router = useRouter();

  useEffect(() => {
    setIsClient(true);

    // Try to get username from localStorage
    const storedUsername = localStorage.getItem("username");
    if (storedUsername) {
      setUsername(storedUsername);
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("username");
    setUsername(null);
    router.push("/login");
  };

  return (
    <header className="bg-white text-gray-900 p-4 shadow-md">
      <div className="max-w-7xl mx-auto flex justify-between items-center">
        <Link href="/" className="text-xl font-bold hover:text-blue-600 transition">
          SmartClaim
        </Link>
        <nav className="flex items-center space-x-6">
          <Link href="/" className="hover:text-blue-600 hover:underline transition">
            Home
          </Link>


          {isClient && username ? (
            <>
              <span className="font-semibold text-gray-700">
                Hello, {username}
              </span>
              <button
                onClick={handleLogout}
                className="ml-2 rounded bg-red-500 px-3 py-1 text-white hover:bg-red-600 transition duration-200 ease-in-out"
              >
                Logout
              </button>
            </>
          ) : isClient ? (
            <Link href="/login" className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition">
              Login
            </Link>
          ) : null}
        </nav>
      </div>
    </header>
  );
}
