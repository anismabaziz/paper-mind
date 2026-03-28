export interface File {
  created_at: Date;
  id: string;
  last_accessed_at: Date;
  is_processed: boolean;
  metadata: {
    mimetype: string;
    size: number;
  };
  name: string;
  updated_at: Date;
  url: string;
}
