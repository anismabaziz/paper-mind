import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getFiles,
  uploadFile,
  deleteFile,
  checkIsProcessed,
  retryIngestionJob,
  cancelIngestionJob,
  getMessages,
  getFileMeta,
  markFileOpened,
  reindexFile,
} from "@/services/files";
import type { IDeleteFile } from "@/services/files";
import type { File as DbFile } from "@/types/db";
import { isIngestionActive } from "@/types/db";

// All per-document keys live under the "files" prefix so a single
// invalidateQueries({ queryKey: ["files"] }) after upload/delete/process
// drops every derived entry instead of leaving stale status/messages/meta
// behind. The filename is the backend identity for these routes.
export const fileKeys = {
  status: (name: string) => ["files", name, "is-processed"] as const,
  messages: (name: string) => ["files", name, "messages"] as const,
  meta: (name: string) => ["files", name, "meta"] as const,
};

const ACTIVE_POLL_MS = 2000;

export function useFiles() {
  return useQuery({
    queryKey: ["files"],
    queryFn: getFiles,
    refetchInterval: (q) => {
      const hasActive = q.state.data?.files.some(
        (f) => isIngestionActive(f.ingestion?.state) || (!f.is_processed && !f.ingestion)
      );
      return hasActive ? ACTIVE_POLL_MS : false;
    },
  });
}

// All per-document keys live under the "files" prefix so a single
// invalidateQueries({ queryKey: ["files"] }) after upload/delete/process
// drops every derived entry instead of leaving stale status/messages/meta
// behind. The filename is the backend identity for these routes.
export function useFileStatus(file: Pick<DbFile, "name"> | null | undefined) {
  return useQuery({
    queryKey: fileKeys.status(file?.name ?? ""),
    queryFn: () => checkIsProcessed(file as DbFile),
    enabled: !!file,
    refetchInterval: (q) => {
      if (isIngestionActive(q.state.data?.ingestion?.state)) return ACTIVE_POLL_MS;
      if (q.state.data?.is_processed) return false;
      return q.state.error ? ACTIVE_POLL_MS : false;
    },
  });
}

export function useFileMessages(file: Pick<DbFile, "name"> | null | undefined, enabled: boolean) {
  return useQuery({
    queryKey: fileKeys.messages(file?.name ?? ""),
    queryFn: () => getMessages((file as DbFile).name),
    enabled: !!file && enabled,
  });
}

export function useFileMeta(file: Pick<DbFile, "name"> | null | undefined) {
  return useQuery({
    queryKey: fileKeys.meta(file?.name ?? ""),
    queryFn: () => getFileMeta((file as DbFile).name),
    enabled: !!file,
  });
}

type UploadResult = Awaited<ReturnType<typeof uploadFile>>;

type MutationCallbacks<TData, TVariables> = {
  onSuccess?: (data: TData, variables: TVariables) => void | Promise<void>;
  onError?: (err: Error, variables: TVariables) => void;
};

export function useUploadFile(options?: MutationCallbacks<UploadResult, File>) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: uploadFile,
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      void options?.onSuccess?.(data, variables);
    },
    onError: (err, variables) => {
      options?.onError?.(err, variables);
    },
  });
}

export function useDeleteFile(options?: MutationCallbacks<IDeleteFile, DbFile>) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteFile,
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      void options?.onSuccess?.(data, variables);
    },
    onError: (err, variables) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      options?.onError?.(err, variables);
    },
  });
}

export function useRetryIngestion(
  options?: MutationCallbacks<Awaited<ReturnType<typeof retryIngestionJob>>, string>
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: retryIngestionJob,
    onSuccess: (_data, name) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      queryClient.invalidateQueries({ queryKey: fileKeys.status(name) });
    },
    onError: (err, name) => {
      options?.onError?.(err, name);
    },
  });
}

export function useCancelIngestion(
  options?: MutationCallbacks<Awaited<ReturnType<typeof cancelIngestionJob>>, string>
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: cancelIngestionJob,
    onSuccess: (_data, name) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      queryClient.invalidateQueries({ queryKey: fileKeys.status(name) });
    },
    onError: (err, name) => {
      options?.onError?.(err, name);
    },
  });
}

export function useReindex(
  options?: MutationCallbacks<Awaited<ReturnType<typeof reindexFile>>, string>
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: reindexFile,
    onSuccess: (_data, name) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      queryClient.invalidateQueries({ queryKey: fileKeys.status(name) });
    },
    onError: (err, name) => {
      options?.onError?.(err, name);
    },
  });
}

export function useTouchFileOpened() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: markFileOpened,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
  });
}
