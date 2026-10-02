import client, { apiBaseUrl } from "./client";
import { StreamProtocolError, isRecord, readEventStream } from "./files";
import type { DocumentIndex, File as DbFile } from "@/types/db";
import { isIndexStale, isIngestionActive } from "@/types/db";

/**
 * One Document in a brief's scope.
 *
 * The label is what the brief and the reader both use to talk about it, and
 * the title is what a reader recognises the paper by. The storage filename the
 * server filters on is never part of this: it is not something a reader is
 * shown, or something a model is told.
 */
export interface IBriefDocument {
  label: string;
  title: string;
  document_id: string;
  index_generation: number | null;
}

/** One Passage a brief collected, under the id its claims may cite. */
export interface IBriefEvidence {
  evidence_id: string;
  label: string;
  document_id: string;
  /** The Document's storage key, which is never what a reader is shown. */
  document: string;
  /** The title a reader recognises the Document by. */
  title: string;
  chunk_index: number;
  page: number | null;
  /** Where this Passage landed in the search that found it, counting from 1. */
  rank: number;
  position: number;
  method: string;
  /** What the model was shown: an excerpt until the brief read it in full. */
  content: string;
  /** Whether the brief read this Passage in full rather than as an excerpt. */
  read: boolean;
}

/** What the loop was allowed to spend, and what it spent. */
export interface IBriefBudget {
  turns: number;
  tool_calls: number;
  repeated_calls: number;
  tokens: number;
  elapsed_seconds: number;
  documents: number;
  max_turns: number;
  max_tool_calls: number;
  max_repeated_calls: number;
  max_tokens: number;
  max_seconds: number;
}

/** The ceilings a brief starts under, before anything is spent. */
export type IBriefLimits = Pick<
  IBriefBudget,
  | "documents"
  | "max_turns"
  | "max_tool_calls"
  | "max_repeated_calls"
  | "max_tokens"
  | "max_seconds"
>;

/**
 * How a brief ended.
 *
 * `complete` is the only status that means the model answered on its own. A
 * brief stopped by a limit is `incomplete` even when it collected evidence, and
 * that distinction is the whole point: a partial result is shown as a partial
 * result.
 */
export type BriefStatus = "complete" | "incomplete";

export interface IBriefStart {
  briefId: string;
  documents: IBriefDocument[];
  promptVersion: string | null;
  limits: IBriefLimits;
}

/** One checkable statement, with the evidence on each side kept apart. */
export interface IBriefClaim {
  order: number;
  claim: string;
  supports: string[];
  conflicts: string[];
  status: "supported" | "contested" | "unresolved";
}

/** The structured result: summary, ordered claims, gaps, abstention. */
export interface IBrief {
  summary: string;
  claims: IBriefClaim[];
  gaps: string[];
  abstained: boolean;
}

/** The model that ran a brief, as the terminal event reports it. */
export interface IBriefModel {
  provider: string;
  model: string;
}

export interface IBriefDone {
  status: BriefStatus;
  answer: string;
  /** Only sent on an incomplete brief: the limit that stopped the loop. */
  stoppedBy: string | null;
  /** Only sent on an incomplete brief: what stopped it means for the reader. */
  message: string | null;
  evidence: IBriefEvidence[];
  documents: IBriefDocument[];
  budget: IBriefBudget;
  brief: IBrief | null;
  claims: IBriefClaim[];
  gaps: string[];
  abstained: boolean;
  promptVersion: string | null;
  model: IBriefModel | null;
}

/** Why a brief could not run, as a refused request or a failed turn. */
export type BriefFailureCategory = "provider" | "timeout" | "settings" | "scope";

export interface IBriefProviderError {
  message: string;
  category: "provider" | "timeout";
  status: BriefStatus;
  evidence: IBriefEvidence[];
  documents: IBriefDocument[];
  budget: IBriefBudget;
}

export interface IBriefAbstained {
  message: string;
  reason: "no_evidence";
  documents: IBriefDocument[];
  stoppedBy: string | null;
}



export interface IBriefStreamHandlers {
  onStart?: (start: IBriefStart) => void;
  onDone: (done: IBriefDone) => void;
  onAbstained?: (result: IBriefAbstained) => void;
  onProviderError?: (failure: IBriefProviderError) => void;
}

function readString(data: Record<string, unknown>, key: string, event: string): string {
  const value = data[key];
  if (typeof value !== "string") {
    throw new StreamProtocolError(`The ${event} event has no ${key}.`);
  }
  return value;
}

function readDocuments(data: Record<string, unknown>): IBriefDocument[] {
  const documents = data.documents;
  if (!Array.isArray(documents)) return [];
  return documents.filter(isRecord).map((document) => ({
    label: typeof document.label === "string" ? document.label : "",
    title: typeof document.title === "string" ? document.title : "",
    document_id: typeof document.document_id === "string" ? document.document_id : "",
    index_generation:
      typeof document.index_generation === "number" ? document.index_generation : null,
  }));
}

function readEvidence(data: Record<string, unknown>): IBriefEvidence[] {
  const evidence = data.evidence;
  if (!Array.isArray(evidence)) return [];
  return evidence.filter(isRecord).map((item) => ({
    evidence_id: typeof item.evidence_id === "string" ? item.evidence_id : "",
    label: typeof item.label === "string" ? item.label : "",
    document_id: typeof item.document_id === "string" ? item.document_id : "",
    document: typeof item.document === "string" ? item.document : "",
    title: typeof item.title === "string" ? item.title : "",
    chunk_index: typeof item.chunk_index === "number" ? item.chunk_index : 0,
    page: typeof item.page === "number" ? item.page : null,
    rank: typeof item.rank === "number" ? item.rank : 0,
    position: typeof item.position === "number" ? item.position : 0,
    method: typeof item.method === "string" ? item.method : "",
    content: typeof item.content === "string" ? item.content : "",
    read: item.read === true,
  }));
}

function readBudget(data: Record<string, unknown>): IBriefBudget {
  const budget = isRecord(data.budget) ? data.budget : {};
  const number = (key: string) => (typeof budget[key] === "number" ? (budget[key] as number) : 0);
  return {
    turns: number("turns"),
    tool_calls: number("tool_calls"),
    repeated_calls: number("repeated_calls"),
    tokens: number("tokens"),
    elapsed_seconds: number("elapsed_seconds"),
    documents: number("documents"),
    max_turns: number("max_turns"),
    max_tool_calls: number("max_tool_calls"),
    max_repeated_calls: number("max_repeated_calls"),
    max_tokens: number("max_tokens"),
    max_seconds: number("max_seconds"),
  };
}

function readIds(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function readClaim(item: Record<string, unknown>, order: number): IBriefClaim {
  const supports = readIds(item.supports);
  const conflicts = readIds(item.conflicts);
  // An explicit unresolved stays unresolved; anything else follows the
  // evidence, so a cached claim with nothing behind it never reads supported.
  const derived = supports.length === 0 && conflicts.length === 0 ? "unresolved" : "supported";
  const contested = supports.length > 0 && conflicts.length > 0 ? "contested" : derived;
  const withConflicts = conflicts.length > 0 && supports.length === 0 ? "contested" : contested;
  const status = item.status === "unresolved" ? "unresolved" : withConflicts;
  return {
    order: typeof item.order === "number" ? item.order : order,
    claim: typeof item.claim === "string" ? item.claim : "",
    supports,
    conflicts,
    status,
  };
}

function readClaims(data: Record<string, unknown>): IBriefClaim[] {
  if (isRecord(data.brief) && Array.isArray(data.brief.claims)) {
    return data.brief.claims
      .filter(isRecord)
      .map((item, index) => readClaim(item as Record<string, unknown>, index + 1))
      .filter((claim) => claim.claim.length > 0);
  }
  const direct = data.claims;
  if (Array.isArray(direct)) {
    return direct
      .filter(isRecord)
      .map((item, index) => readClaim(item, index + 1))
      .filter((claim) => claim.claim.length > 0);
  }
  return [];
}

function readGaps(data: Record<string, unknown>): string[] {
  const direct = data.gaps;
  if (Array.isArray(direct)) return direct.filter((item): item is string => typeof item === "string");
  if (isRecord(data.brief) && Array.isArray(data.brief.gaps)) {
    return data.brief.gaps.filter((item): item is string => typeof item === "string");
  }
  return [];
}

function readBrief(data: Record<string, unknown>): IBrief | null {
  if (isRecord(data.brief)) {
    const brief = data.brief;
    return {
      summary:
        typeof brief.summary === "string"
          ? brief.summary
          : typeof data.answer === "string"
            ? data.answer
            : "",
      claims: readClaims(data),
      gaps: readGaps(data),
      abstained: brief.abstained === true || data.abstained === true,
    };
  }
  if (typeof data.answer === "string" || data.claims !== undefined) {
    return {
      summary: typeof data.answer === "string" ? data.answer : "",
      claims: readClaims(data),
      gaps: readGaps(data),
      abstained: data.abstained === true,
    };
  }
  return null;
}

function readModel(data: Record<string, unknown>): IBriefModel | null {
  if (isRecord(data.model)) {
    return {
      provider: typeof data.model.provider === "string" ? data.model.provider : "",
      model: typeof data.model.model === "string" ? data.model.model : "",
    };
  }
  return null;
}

/**
 * Hand one event to its handler, and report whether it ended the stream.
 *
 * An event this protocol does not define, or one whose payload does not match
 * it, is a protocol error. Ignoring it would leave the reader watching a brief
 * that has already stopped, or one that will never finish.
 */
function dispatch(
  event: { name: string; data: Record<string, unknown> },
  handlers: IBriefStreamHandlers,
): boolean {
  const { name, data } = event;
  switch (name) {
    case "start":
      handlers.onStart?.({
        briefId: readString(data, "brief_id", name),
        documents: readDocuments(data),
        promptVersion:
          typeof data.prompt_version === "string" ? data.prompt_version : null,
        limits: readBudget({ budget: data.limits }),
      });
      return false;
    case "done": {
      const status = data.status === "incomplete" ? "incomplete" : "complete";
      handlers.onDone({
        status,
        // An incomplete brief has no answer: the model never wrote one, and an
        // empty string here would render as a blank result rather than as the
        // partial brief the server said it was.
        answer: typeof data.answer === "string" ? data.answer : "",
        stoppedBy: typeof data.stopped_by === "string" ? data.stopped_by : null,
        message: typeof data.message === "string" ? data.message : null,
        evidence: readEvidence(data),
        documents: readDocuments(data),
        budget: readBudget(data),
        brief: readBrief(data),
        claims: readClaims(data),
        gaps: readGaps(data),
        abstained: data.abstained === true,
        promptVersion:
          typeof data.prompt_version === "string" ? data.prompt_version : null,
        model: readModel(data),
      });
      return true;
    }
    case "abstained":
      handlers.onAbstained?.({
        message: readString(data, "message", name),
        reason: "no_evidence",
        documents: readDocuments(data),
        stoppedBy: typeof data.stopped_by === "string" ? data.stopped_by : null,
      });
      return true;
    case "provider_error": {
      const category = data.category === "timeout" ? "timeout" : "provider";
      handlers.onProviderError?.({
        message: readString(data, "error", name),
        category,
        status: "incomplete",
        evidence: readEvidence(data),
        documents: readDocuments(data),
        budget: readBudget(data),
      });
      return true;
    }
    default:
      throw new StreamProtocolError(`The brief sent an unknown ${name} event.`);
  }
}

/**
 * Run one brief and read it as it happens.
 *
 * The whole answer arrives in the terminal event rather than as tokens: a brief
 * takes several turns and writes once, so streaming its prose would show a
 * spinner for the entire run and then the same text. What the reader watches is
 * the scope opening, and the result at the end.
 */
export async function briefStream(
  question: string,
  documents: string[],
  handlers: IBriefStreamHandlers,
  options?: { signal?: AbortSignal },
) {
  const response = await fetch(`${apiBaseUrl}/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, documents }),
    signal: options?.signal,
  });

  if (!response.ok || !response.body) {
    throw await refusalOf(response);
  }

  await readEventStream(
    response.body,
    (event) => dispatch(event, handlers),
    "The brief sent an unreadable event.",
  );
}

/** A brief the server refused before it opened a stream. */
export class BriefRequestError extends Error {
  readonly category: BriefFailureCategory;
  readonly missingCapabilities: string[];
  readonly needsSettings: boolean;

  constructor(
    message: string,
    category: BriefFailureCategory = "scope",
    missingCapabilities: string[] = [],
  ) {
    super(message);
    this.name = "BriefRequestError";
    this.category = category;
    this.missingCapabilities = missingCapabilities;
    this.needsSettings = SETTINGS_CATEGORIES.has(category);
  }
}

const SETTINGS_CATEGORIES: ReadonlySet<string> = new Set([
  "no_provider_configured",
  "incomplete_settings",
  "unsupported_provider",
  "unsupported_model",
  "secrets_resave_required",
]);

const MODEL_CATEGORY = "research_brief_unsupported_model";

async function refusalOf(response: Response): Promise<BriefRequestError> {
  const body = (await response.json().catch(() => null)) as
    | Record<string, unknown>
    | null;
  const message =
    typeof body?.error === "string"
      ? body.error
      : `The brief could not start (${response.status}).`;
  const category = typeof body?.category === "string" ? body.category : "";
  const missing = Array.isArray(body?.missing_capabilities)
    ? body.missing_capabilities.filter((item): item is string => typeof item === "string")
    : [];
  if (category === MODEL_CATEGORY) return new BriefRequestError(message, "scope", missing);
  return new BriefRequestError(
    message,
    SETTINGS_CATEGORIES.has(category) ? "settings" : "scope",
    missing,
  );
}

/**
 * Why a brief may not be started right now, or null when it may.
 *
 * Two questions are answered here, in the order a reader hits them: whether the
 * chosen model can run one at all, and whether the chosen pair can be read.
 * The first is settled by App Settings, which publish each catalog model's tool
 * and structured-output support; the second by each Document's index state,
 * which the server judges and the listing only approximates.
 */
export function briefUnavailableReason(
  model: { tool_use: boolean; structured_output: boolean } | null,
  selected: DbFile[],
  files: DbFile[] | undefined,
): string | null {
  if (!model) return "Choose a model in App Settings before starting a brief.";
  const missing: string[] = [];
  if (!model.tool_use) missing.push("tool use");
  if (!model.structured_output) missing.push("structured output");
  if (missing.length > 0) {
    return `A research brief needs a model that reports ${missing.join(" and ")}. Choose one in App Settings.`;
  }
  if (selected.length !== 2) {
    const count = selected.length;
    return count === 0
      ? "Select two Documents to compare."
      : `Select one more Document — a brief compares exactly two.`;
  }
  const blocked = selected.find((file) => isBriefBlocked(file));
  if (blocked) return blockedReason(blocked, files);
  return null;
}

/** Whether one Document cannot be part of a brief's scope right now. */
export function isBriefBlocked(file: DbFile): boolean {
  return (
    file.deletion_state !== "active" ||
    isIngestionActive(file.ingestion?.state) ||
    isIndexStale(file.index) ||
    !isBriefReadable(file)
  );
}

/** Whether one Document's index is ready to be searched at all. */
export function isBriefReadable(file: DbFile): boolean {
  return file.is_processed && (file.index?.state ?? "pending") === "ready";
}

/** What is wrong with one Document, in the words the reader can act on. */
function blockedReason(file: DbFile, files: DbFile[] | undefined): string {
  const title = displayTitleOf(file, files);
  if (file.deletion_state === "deleting") return `${title} is being removed.`;
  if (file.ingestion?.state === "failed") return `${title} failed to index. Retry it first.`;
  if (isIngestionActive(file.ingestion?.state)) return `${title} is still indexing.`;
  if (isIndexStale(file.index)) return `${title} has a stale index. Reindex it first.`;
  return `${title} is not indexed yet.`;
}

function displayTitleOf(file: DbFile, files: DbFile[] | undefined): string {
  const listed = files?.find((candidate) => candidate.id === file.id);
  return file.title ?? listed?.title ?? "That Document";
}

/**
 * Ask a running brief to stop, so the reader keeps what it found.
 *
 * Aborting the request instead would abandon the brief along with the
 * connection, which is right for a reader who left the page and wrong for one
 * who pressed stop. The server stops the loop at a point where its evidence is
 * still consistent and sends a terminal event carrying it.
 */
export async function cancelBrief(briefId: string) {
  const response = await client.post<{ brief_id: string; cancelled: boolean }>(
    "/research/cancel",
    { brief_id: briefId },
  );
  return response.data;
}

export interface IBriefScope {
  /** The pair, under the labels and titles the brief itself will use. */
  documents: IBriefDocument[];
  document_count: number;
}

/**
 * Ask the server whether it will accept this pair, before a brief is started.
 *
 * The library listing can say a Document is processed; only the server can say
 * whether its index still matches the configuration this build serves. So the
 * scope is confirmed here rather than guessed from the listing, and the labels
 * the brief will use come back with it.
 */
export async function checkBriefScope(documents: string[], signal?: AbortSignal) {
  const params = documents.flatMap((name) => ["documents", name]);
  const response = await client.get<IBriefScope>("/research/scope", { params, signal });
  return response.data;
}

/** A completed brief kept so it survives a reload. */
export interface ISavedBrief {
  question: string;
  documents: string[];
  run: {
    status: "complete" | "incomplete";
    answer: string;
    stoppedBy: string | null;
    message: string | null;
    evidence: IBriefEvidence[];
    budget: IBriefBudget;
    scope: IBriefDocument[];
    brief: IBrief | null;
    claims: IBriefClaim[];
    gaps: string[];
    abstained: boolean;
    promptVersion: string | null;
    model: IBriefModel | null;
  };
  savedAt: string;
}

const SAVED_BRIEF_KEY = "papermind:last-brief-v1";

/** Save the last finished brief so closing the dialog keeps it. */
export function saveBrief(saved: ISavedBrief): void {
  try {
    localStorage.setItem(SAVED_BRIEF_KEY, JSON.stringify(saved));
  } catch {
    // Storage full or unavailable: the brief stays on screen, it just
    // does not survive a reload.
  }
}

/** Load the last finished brief, or null when nothing was saved. */
export function loadBrief(): ISavedBrief | null {
  try {
    const raw = localStorage.getItem(SAVED_BRIEF_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<ISavedBrief>;
    if (!parsed || typeof parsed !== "object" || !parsed.run) return null;
    return parsed as ISavedBrief;
  } catch {
    return null;
  }
}

export type { DocumentIndex };