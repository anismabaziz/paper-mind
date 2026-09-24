export type DeletionState = "active" | "deleting" | "delete_failed";

export interface File {
  id: string;
  is_processed: boolean;
  last_opened_at: string | null;
  deletion_state: DeletionState;
  deletion_error: string | null;
  deletion_attempts: number;
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
