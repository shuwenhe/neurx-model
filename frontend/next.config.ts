import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Serve the app under /model so it is reachable at
  // http://<host>:<port>/model
  basePath: "/model",
  assetPrefix: "/model",
};

export default nextConfig;
