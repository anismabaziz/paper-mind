// How a settings number or failure reaches the reader. The server's own
// wording is preferred where it sent one: an error it explained itself is more
// useful than anything this app could invent about it.

export function errorMessage(e: unknown, fallback: string): string {
  if (e instanceof Error && "response" in e) {
    const data = (e as { response?: { data?: { error?: string } } }).response?.data;
    if (data?.error) return data.error;
  }
  return fallback;
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