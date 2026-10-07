import type { NextConfig } from "next";

// NEXT_PUBLIC_STATIC=1 builds a fully static site (GitHub Pages): models run in the
// browser, so there is no server. Otherwise the dashboard talks to the FastAPI service.
const STATIC = process.env.NEXT_PUBLIC_STATIC === "1";

const turbopack = {
  rules: { "*.css": { loaders: ["@tailwindcss/turbopack"], as: "*.css" } },
};

const nextConfig: NextConfig = STATIC
  ? {
      output: "export",
      basePath: process.env.NEXT_PUBLIC_BASE_PATH || "",
      images: { unoptimized: true },
      devIndicators: false,
      turbopack,
    }
  : {
      cacheComponents: true,
      partialPrefetching: true,
      devIndicators: false,
      turbopack,
    };

export default nextConfig;
