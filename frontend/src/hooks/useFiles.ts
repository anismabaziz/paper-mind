import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getFiles,
  uploadFile,
  deleteFile,
  processFile,
  checkIsProcessed,
  getMessages,
  getFileMeta,
} from "@/services/files";
import type { File as DbFile } from "@/types/db";

// All per-document keys live under the "files" prefix so a single
// invalidateQueries({ queryKey: ["files"] }) after upload/delete/process
// drops every derived entry instead of leaving stale status/messages/meta
// behind. The filename is the backend identity for these routes.
export const fileKeys = {
  status: (name: string) => ["files", name, "is-processed"] as const,
  messages: (name: string) => ["files", name, "messages"] as const,
  meta: (name: string) => ["files", name, "meta"] as const,
};

export function useFiles() {
  return useQuery({
    queryKey: ["files"],
    queryFn: getFiles,
    refetchInterval: (q) => {
      const hasUnprocessed = q.state.data?.files.some((f) => !f.is_processed);
      return hasUnprocessed ? 3000 : false;
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
    refetchInterval: (q) => (q.state.data?.is_processed ? false : 3000),
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
type ProcessResult = Awaited<ReturnType<typeof processFile>>;

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

export function useDeleteFile() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
  });
}

export function useProcessFile(options?: MutationCallbacks<ProcessResult, DbFile>) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: processFile,
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      void options?.onSuccess?.(data, variables);
    },
    onError: (err, variables) => {
      options?.onError?.(err, variables);
    },
  });
}
