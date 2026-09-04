export interface File {
  id: string;
  is_processed: boolean;
  metadata: {
    content_type: string;
    size: number;
  };
  name: string;
  url: string;
}
