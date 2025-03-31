export interface File {
  created_at: Date;
  id: string;
  last_accessed_at: Date;
  metadata: {
    mimetype: string;
    size: number;
  };
  name: string;
  updated_at: Date;
  url: string;
}
