/**
 * Reading a failure the backend sent.
 *
 * Every failure the app reports comes in one shape, whether it arrived as an
 * HTTP status or as an event inside a stream that had already opened:
 *
 *     { "error": "...", "category": "...", "details": { ... } }
 *
 * `error` is what a reader is shown, `category` is the stable name to branch on,
 * and `details` carries whatever that particular failure has to explain. Read
 * through these rather than reaching into the body, so a new field is added in
 * one place and a caller never has to know which of the two it is looking at.
 */

import { AxiosError } from "axios";

/** One failure as the backend sent it. */
export interface IApiError {
  /** What the reader is shown. */
  message: string;
  /** The stable name to branch on. */
  category: string;
  /** Whatever extra this particular failure had to explain. */
  details: Record<string, unknown>;
}

/**
 * A failure the backend reported, carrying the whole of what it sent.
 *
 * Extends Error so everything that already displays `error.message` shows the
 * server's own wording for free, while `category` and `details` stay reachable
 * for the callers that switch on them.
 */
export class ApiError extends Error {
  readonly category: string;
  readonly details: Record<string, unknown>;

  constructor(failure: IApiError) {
    super(failure.message);
    this.name = "ApiError";
    this.category = failure.category;
    this.details = failure.details;
  }
}

/** The shape every failure body has, or the parts of it that are readable. */
type RawError = Partial<{
  error: unknown;
  category: unknown;
  details: unknown;
}> | null;

/**
 * Read a failure out of an unknown body.
 *
 * Falls back to whatever the caller supplies rather than throwing: a body that
 * did not arrive as JSON — a proxy's HTML error page, an empty response — is
 * still a failure the reader has to be told about, and it is not worth losing
 * the fallback message over.
 */
export function readError(body: unknown, fallback: string): IApiError {
  const raw = body as RawError;
  return {
    message: typeof raw?.error === "string" && raw.error ? raw.error : fallback,
    category: typeof raw?.category === "string" ? raw.category : "",
    details:
      typeof raw?.details === "object" && raw.details !== null
        ? (raw.details as Record<string, unknown>)
        : {},
  };
}

/**
 * Read the failure off a failed response.
 *
 * Used wherever a request is made with `fetch` and the body has to be parsed by
 * hand. Responses that go through the axios client raise an {@link ApiError}
 * instead, carrying the same three fields.
 */
export async function readErrorResponse(
  response: Response,
  fallback: string,
): Promise<IApiError> {
  return readError(await response.json().catch(() => null), fallback);
}

/**
 * Replace an axios failure with one the reader can be shown.
 *
 * Without this, axios reports every failure as "Request failed with status code
 * NNN" and the server's own wording — which says what went wrong and what to do
 * about it — is discarded. Every request that goes through the axios client
 * raises an `ApiError` instead, so the ~18 call sites that display
 * `error.message` show the real reason without each of them having to know
 * where the wording came from.
 *
 * A failure with no response at all — the server was never reached — has no body
 * to read and is passed through as it is, because "Failed to load PDF (0)" would
 * be worse than what axios already says about the network.
 */
export function toApiError(e: unknown): unknown {
  if (!(e instanceof AxiosError) || !e.response) return e;
  return new ApiError(readError(e.response.data, fallbackFor(e.response.status)));
}

/** The wording for a status the server sent no body for. */
function fallbackFor(status: number): string {
  return `Request failed (${status})`;
}