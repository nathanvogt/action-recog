#!/usr/bin/env ts-node

import "../server/trainDatasetServer.js";

// This script simply imports and runs the server
// The server configuration is handled in the server file itself
console.log("Starting TrainDataset server...");
console.log("Use Ctrl+C to stop the server");

process.on("SIGINT", () => {
  console.log("\nReceived SIGINT. Gracefully shutting down...");
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.log("\nReceived SIGTERM. Gracefully shutting down...");
  process.exit(0);
});
