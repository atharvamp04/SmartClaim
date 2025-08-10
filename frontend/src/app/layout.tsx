// src/app/layout.tsx
import ClientLayoutWrapper from "@/app/ClientLayoutWrapper";
import "./globals.css";

export const metadata = {
  title: "SmartClaim",
  description: "Insurance claim system",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {/* Client wrapper handles showing header/footer based on route */}
        <ClientLayoutWrapper>{children}</ClientLayoutWrapper>
      </body>
    </html>
  );
}
