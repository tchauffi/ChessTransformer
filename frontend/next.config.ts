import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // "standalone" for the docker-compose frontend container; "export" builds a
  // static site that ct-bot serves itself (the Hugging Face Space).
  output: process.env.NEXT_OUTPUT === "export" ? "export" : "standalone",
};

export default nextConfig;
