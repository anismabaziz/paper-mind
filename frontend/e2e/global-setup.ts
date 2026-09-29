import { BACKEND_URL, seedWorkingSettings, unfailAll, resetRuntime } from "./fixtures";

async function main() {
  let health = 0;
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    health = response.status;
  } catch {
    health = 0;
  }
  if (health !== 200) {
    throw new Error(
      `browser backend not reachable at ${BACKEND_URL} (status ${health}). ` +
        "Start Postgres + Qdrant first: " +
        "docker compose -f ../backend/compose.test.yaml up -d --wait",
    );
  }
  await unfailAll();
  await resetRuntime();
  // Chat refuses without saved provider settings; every spec starts from one
  // known-good credential verified by the deterministic test adapter.
  await seedWorkingSettings();
}

export default main;
