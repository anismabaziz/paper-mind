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

export function useFileStatus(file: Pick<DbFile, "name"> | null | undefined) {
  return useQuery({
    queryKey: [file?.name, "is-processed"],
    queryFn: () => checkIsProcessed(file as DbFile),
    enabled: !!file,
    refetchInterval: (q) => (q.state.data?.is_processed ? false : 3000),
  });
}

export function useFileMessages(file: Pick<DbFile, "name"> | null | undefined, enabled: boolean) {
  return useQuery({
    queryKey: [file?.name, "messages"],
    queryFn: () => getMessages((file as DbFile).name),
    enabled: !!file && enabled,
  });
}

export function useFileMeta(file: Pick<DbFile, "name"> | null | undefined) {
  return useQuery({
    queryKey: [file?.name, "meta"],
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
