import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  deleteFile,
  getFiles,
  processFile,
  uploadFile,
} from "@/services/files";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Plus,
  Library,
  UploadIcon,
  File,
  Trash2,
  MoreHorizontal,
  Loader2,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import usePdfStore from "@/store/pdf-state";
import { cn } from "@/lib/utils";
import { useEffect, useRef } from "react";

export default function ListPDF() {
  const queryClient = useQueryClient();
  const { setFile, file: selectedFile } = usePdfStore();
  
  const filesQuery = useQuery({ 
    queryKey: ["files"], 
    queryFn: getFiles,
    refetchInterval: (query) => {
       // Refetch while any file is still processing to update the "Analyzed" status
       const hasUnprocessed = query.state.data?.files.some(f => !f.is_processed);
       return hasUnprocessed ? 3000 : false;
    }
  });
  const files = filesQuery.data?.files;

  const uploadFileMutation = useMutation({
    mutationFn: uploadFile,
    onSuccess: async (data) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      try {
        await processFile(data.file);
      } catch (err) {
        console.error("Indexing failed:", err);
        // Surface the failure – the file will stay is_processed=false,
        // the Library row keeps the "Index" badge but polling will
        // stop on the next refetch; show a visible error so the user
        // knows it didn't just hang.
        alert(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        queryClient.invalidateQueries({ queryKey: ["files"] });
        queryClient.invalidateQueries({ queryKey: [data.file.name, "is-processed"] });
      }
    },
    onError: (err) => {
      console.error("Upload failed:", err);
      alert(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    },
  });

  const deleteFileMutation = useMutation({
    mutationFn: deleteFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
  });

  const processFileMutation = useMutation({
    mutationFn: processFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
    onError: (err) => {
      console.error("Re-indexing failed:", err);
      alert(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`);
    },
  });

  // Automatically pick a new file if the selected one is gone or if all files are deleted
  useEffect(() => {
    if (files) {
      if (files.length === 0) {
        if (selectedFile !== null) {
          setFile(null);
        }
      } else {
        const currentExists = files.find(f => f.id === selectedFile?.id);
        if (!currentExists || !selectedFile) {
          // Find the first available processed file, or just the first file
          const nextFile = files.find(f => f.is_processed) || files[0];
          setFile(nextFile);
        }
      }
    }
  }, [files, selectedFile, setFile]);

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      uploadFileMutation.mutate(file);
    }
  };
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(0) + " KB";
    else return (bytes / 1048576).toFixed(1) + " MB";
  };

  return (
    <div className="flex flex-col h-full glass rounded-xl overflow-hidden shadow-sm">
      <div className="h-[52px] px-4 border-b border-slate-200/80 bg-slate-50 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <Library size={16} className="text-slate-600" />
          <h3 className="font-semibold text-xs uppercase tracking-wider text-slate-700">Library</h3>
        </div>
        <Button
          size="sm"
          variant="ghost"
          className="h-7 w-7 p-0 hover:bg-slate-200/80 text-slate-500 hover:text-slate-800 transition-all rounded"
          onClick={handleButtonClick}
          disabled={uploadFileMutation.isPending}
        >
          {uploadFileMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : <Plus size={16} strokeWidth={2} />}
        </Button>
        <Input
          type="file"
          hidden
          onChange={handleFileChange}
          id="file"
          ref={fileInputRef}
          accept=".pdf"
        />
      </div>

      <div className="flex-grow overflow-y-auto p-2.5 space-y-1.5">
        {filesQuery.isPending && (
          <div className="space-y-2">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-12 w-full bg-slate-100/80 border border-slate-200/40 animate-pulse rounded" />
            ))}
          </div>
        )}

        {!filesQuery.isPending && files?.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full py-8 px-4 text-center">
            <div className="w-10 h-10 bg-slate-50 border border-slate-200 rounded flex items-center justify-center mb-3 text-slate-400">
              <File size={18} />
            </div>
            <p className="text-xs font-semibold text-slate-700 mb-0.5">No Documents Uploaded</p>
            <p className="text-[11px] text-slate-400 mb-4 max-w-[180px]">Ingest a PDF research paper to begin analysis.</p>
            <Button
              className="w-full bg-white hover:bg-slate-50 border-slate-200 text-slate-700 shadow-sm text-xs rounded h-9"
              variant="outline"
              onClick={handleButtonClick}
              disabled={uploadFileMutation.isPending}
            >
              <UploadIcon size={14} className="mr-1.5" />
              Ingest Document
            </Button>
          </div>
        )}

        {files && files.map((file) => {
          const isRemoving = deleteFileMutation.isPending && deleteFileMutation.variables?.id === file.id;
          const isProcessing = !file.is_processed;
          const isRetrying = processFileMutation.isPending && (processFileMutation.variables as any)?.name === file.name;

          return (
            <div
              key={file.id}
              className={cn(
                "group relative p-2.5 rounded flex items-center gap-2.5 transition-colors border",
                selectedFile?.id === file.id 
                  ? "bg-slate-100/80 border-slate-200 text-slate-900 active-glow" 
                  : "bg-white hover:bg-slate-50 border-slate-100 hover:border-slate-200 text-slate-700",
                isProcessing ? "opacity-80" : "cursor-pointer",
                isRemoving && "opacity-50 pointer-events-none"
              )}
              onClick={() => !isProcessing && setFile(file)}
            >
              <div className={cn(
                "p-1.5 rounded transition-colors border",
                selectedFile?.id === file.id ? "bg-primary text-white border-primary" : "bg-slate-50 text-slate-400 border-slate-100 group-hover:bg-slate-100"
              )}>
                {isRemoving || isRetrying ? <Loader2 size={13} className="animate-spin" /> : <File size={13} />}
              </div>
              
              <div className="flex-grow min-w-0">
                <div className="flex items-center gap-1.5">
                  <p className={cn(
                    "text-[12px] font-medium truncate transition-colors",
                    selectedFile?.id === file.id ? "text-slate-950 font-semibold" : "text-slate-700"
                  )}>
                    {file.name.replace(/\.[^/.]+$/, "")}
                  </p>
                  {isProcessing && (
                    <div className="flex items-center gap-1 font-semibold text-[8px] uppercase tracking-wider text-slate-500 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                      <Loader2 size={8} className="animate-spin" />
                      {isRetrying ? "Re-indexing" : "Indexing"}
                    </div>
                  )}
                </div>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  {isRemoving ? "Purging storage..." : isRetrying ? "Retrying embedding..." : formatFileSize(file.metadata.size)}
                </p>
              </div>

              {isProcessing ? (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 text-[10px] px-2 shrink-0"
                  disabled={isRetrying}
                  onClick={(e) => {
                    e.stopPropagation();
                    processFileMutation.mutate(file);
                  }}
                >
                  {isRetrying ? <Loader2 size={12} className="animate-spin" /> : "Retry"}
                </Button>
              ) : !isRemoving && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className={cn(
                        "h-7 w-7 rounded transition-opacity",
                        selectedFile?.id === file.id ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                      )}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <MoreHorizontal size={14} />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="bg-white text-slate-800 border border-slate-200/80 shadow-md min-w-[140px] p-1 rounded-md z-50">
                    <DropdownMenuItem
                      className="text-red-600 focus:bg-red-50 focus:text-red-700 cursor-pointer p-2 rounded text-xs font-medium flex items-center transition-colors"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteFileMutation.mutate(file);
                      }}
                    >
                      <Trash2 size={14} className="mr-2" />
                      <span>Remove Paper</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </div>
          );
        })}
        

      </div>
    </div>
  );
}
