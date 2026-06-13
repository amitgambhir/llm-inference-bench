import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Inference Capacity Planner",
  description: "Size, cost, and validate LLM inference deployments",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 min-h-screen">
        <header className="bg-white border-b border-gray-200 px-6 py-3">
          <div className="max-w-4xl mx-auto flex items-center justify-between">
            <a href="/" className="font-semibold text-brand-700 hover:text-brand-600">
              LLM Inference Planner
            </a>
            <nav className="flex gap-4 text-sm text-gray-500">
              <a href="/" className="hover:text-gray-900">New Scenario</a>
              <a
                href="https://github.com/amitgambhir/llm-inference-bench/blob/main/README.md"
                target="_blank"
                rel="noreferrer"
                className="hover:text-gray-900"
              >
                Docs
              </a>
            </nav>
          </div>
        </header>
        <main className="max-w-4xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
