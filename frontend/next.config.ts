import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Serve the app under /neurx so it is reachable at
  // http://<host>:<port>/neurx
  basePath: "/neurx",
  assetPrefix: "/neurx",
};

export default nextConfig;
