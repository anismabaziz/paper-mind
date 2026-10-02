// How a settings number or failure reaches the reader. The server's own
// wording is preferred where it sent one: an error it explained itself is more
// useful than anything this app could invent about it.

import { getErrorMessage } from "./api-error";

/**
 * The wording to show for a failed settings request.
 *
 * Kept under this name so settings callers keep reading settings wording.
 * Delegates to the shared reader so backend and network failures read the
 * same way everywhere.
 */
export function errorMessage(e: unknown, fallback: string): string {
  return getErrorMessage(e, fallback);
}

export function formatTokenCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 3,
  }).format(value);
}