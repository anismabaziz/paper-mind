import { defineConfig, devices } from "@playwright/test";

// Browser tests run against the real HTTP application (backend/tests/
// browser_server.py) with real Postgres + Qdrant and deterministic model
// adapters. Start the backing services first:
//   POSTGRES_TEST_PASSWORD=... docker compose -f ../backend/compose.test.yaml up -d --wait
// Then: npm run test:e2e. The webServer entries below start the backend and
// the frontend automatically. Tests run serially (workers: 1) because
// stale-index and settings specs mutate server-wide configuration.
//
// The frontend is served from a production build, not the Vite dev server.
// The bundle assertions measure what a user actually downloads: the PDF
// engine staying out of the initial chunk, and no long task while the reader
// mounts. Both are true of the built bundle and neither is measurable through
// an unbundled dev server, which serves hundreds of separate modules.
const backendUrl =
  process.env.BROWSER_BACKEND_URL ?? "http://127.0.0.1:38201";
// Port 5180 keeps e2e runs clear of the default Vite dev port (5173), which a
// developer session may already hold.
const frontendUrl = "http://127.0.0.1:5180";

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  timeout: 240_000,
  expect: {
    timeout: 15_000,
  },
  use: {
    baseURL: frontendUrl,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  outputDir: "./test-results",
  projects: [
    {
      name: "desktop",
      testIgnore: /mobile/,
      use: { viewport: { width: 1440, height: 900 } },
    },
    {
      name: "mobile",
      testMatch: /mobile/,
      use: { ...devices["Pixel 7"] },
    },
  ],
  webServer: [
    {
      command: "uv run python tests/browser_server.py",
      cwd: "../backend",
      env: {
        PAPERMIND_BROWSER_TESTS: "1",
        BROWSER_BACKEND_PORT: "38201",
        BROWSER_FRONTEND_ORIGIN:
          "http://127.0.0.1:5180,http://localhost:5180",
        FULL_STACK_DATABASE_URL:
          process.env.FULL_STACK_DATABASE_URL ??
          "postgresql+psycopg://papermind_test:papermind_test@127.0.0.1:55432/papermind_test",
        FULL_STACK_QDRANT_URL:
          process.env.FULL_STACK_QDRANT_URL ?? "http://127.0.0.1:56333",
      },
      url: `${backendUrl}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
    {
      command: "npm run build && npm run preview -- --port 5180 --strictPort --host 127.0.0.1",
      env: {
        // The build reads this at config time and bakes the API origin into
        // the bundle, so the same value has to reach the build and the server.
        VITE_API_URL: backendUrl,
      },
      url: frontendUrl,
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
  ],
});
