import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Topo-Attention Glass",
  description: "Real-time visualization of TF-A-N topology and attention dynamics",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
