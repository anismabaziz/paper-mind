export type DeletionState = "active" | "deleting" | "delete_failed";

export type IngestionState =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "failed"
  | "stale"
  | "ready";

export type IngestionStage =
  | "queued"
  | "parsing"
  | "embedding"
  | "indexing"
  | "validating"
  | "cancelling"
  | "cancelled"
  | "ready"
  | "failed"
  | "stale";

export interface IngestionJob {
  id: string;
  file_id: string;
  filename: string;
  generation: number;
  state: IngestionState;
  stage: IngestionStage;
  progress: number;
  attempt: number;
  error_category: string | null;
  error_message: string | null;
  worker_id: string | null;
  created_at: string;
  updated_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  heartbeat_at: string | null;
  cancel_requested_at?: string | null;
  cancelled_at?: string | null;
  cancellation_requested?: boolean;
  limits?: Record<string, number>;
  usage?: Record<string, number>;
}

export interface IndexManifest {
  content_hash: string;
  parser: string;
  parser_version: string;
  chunk_size_tokens: number;
  chunk_overlap_tokens: number;
  embedding_model: string;
  embedding_revision: string;
  vector_dimension: number;
  sparse_method: string;
  sparse_tokenizer_version: string;
  reranker_model: string;
  reranker_revision: string;
  reranker_enabled: boolean;
  collection_name: string;
  collection_schema_version: string;
  index_generation: number;
}

export interface IndexChange {
  field: string;
  label: string;
  indexed: string | number | boolean | null;
  current: string | number | boolean | null;
}

export type IndexState = "pending" | "ready" | "stale";

export interface DocumentIndex {
  state: IndexState;
  manifest: IndexManifest | null;
  runtime_manifest: IndexManifest;
  changes: string[];
  change_details: IndexChange[];
}

export function isIndexStale(index: DocumentIndex | null | undefined): boolean {
  return index?.state === "stale";
}

export function indexStatusLine(index: DocumentIndex | null | undefined): string {
  if (!index || index.state !== "stale") return "";
  const reasons = index.change_details.map((change) => change.label).join(", ");
  return `Index is stale · reindex to update ${reasons}`;
}

export interface File {
  id: string;
  is_processed: boolean;
  index_generation?: number | null;
  index?: DocumentIndex | null;
  last_opened_at: string | null;
  deletion_state: DeletionState;
  deletion_error: string | null;
  deletion_attempts: number;
  ingestion: IngestionJob | null;
  metadata: {
    content_type: string;
    size: number;
  };
  name: string;
  title: string | null;
  original_filename: string | null;
  url: string;
}

export function displayTitle(file: File): string {
  return file.title ?? file.name;
}

const STAGE_LABELS: Record<IngestionStage, string> = {
  queued: "Queued",
  parsing: "Parsing",
  embedding: "Embedding",
  indexing: "Indexing",
  validating: "Validating",
  cancelling: "Cancelling",
  cancelled: "Cancelled",
  ready: "Ready",
  failed: "Failed",
  stale: "Superseded",
};

export function ingestionStageLabel(stage: IngestionStage): string {
  return STAGE_LABELS[stage] ?? stage;
}

export function isIngestionActive(state: IngestionState | undefined): boolean {
  return state === "queued" || state === "running" || state === "cancelling";
}

export function isIngestionCancellable(state: IngestionState | undefined): boolean {
  return state === "queued" || state === "running";
}

export function isIngestionRetryable(state: IngestionState | undefined): boolean {
  return state === "failed" || state === "cancelled";
}
