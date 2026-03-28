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
       // @ts-ignore
       const hasUnprocessed = query.state.data?.files.some(f => !f.is_processed);
       return hasUnprocessed ? 3000 : false;
    }
  });
  const files = filesQuery.data?.files;

  const uploadFileMutation = useMutation({
    mutationFn: uploadFile,
    onSuccess: async (data) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      await processFile(data.file);
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
  });

  const deleteFileMutation = useMutation({
    mutationFn: deleteFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
  });

  // Automatically pick a new file if the selected one is gone
  useEffect(() => {
    if (files && files.length > 0) {
      const currentExists = files.find(f => f.id === selectedFile?.id);
      if (!currentExists || !selectedFile) {
        // Find the first available processed file, or just the first file
        const nextFile = files.find(f => f.is_processed) || files[0];
        setFile(nextFile);
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
    <div className="flex flex-col h-full glass rounded-3xl overflow-hidden shadow-xl border-white/40">
      <div className="p-5 border-b border-white/20 bg-white/20 backdrop-blur-md flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-primary/10 rounded-lg text-primary">
            <Library size={20} />
          </div>
          <h3 className="font-bold text-slate-800">Library</h3>
        </div>
        <Button
          size="sm"
          variant="ghost"
          className="h-9 w-9 p-0 hover:bg-primary/10 hover:text-primary transition-all rounded-full"
          onClick={handleButtonClick}
          disabled={uploadFileMutation.isPending}
        >
          <Plus size={20} strokeWidth={2.5} />
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

      <div className="flex-grow overflow-y-auto p-3 space-y-2">
        {filesQuery.isPending && (
          <div className="space-y-3 p-2">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-16 w-full bg-slate-200/40 animate-pulse rounded-2xl" />
            ))}
          </div>
        )}

        {!filesQuery.isPending && files?.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full py-10 px-6 text-center">
            <div className="w-16 h-16 bg-slate-100 rounded-2xl flex items-center justify-center mb-4 text-slate-300">
                <File size={32} />
            </div>
            <p className="text-sm font-semibold text-slate-600 mb-1">Your library is empty</p>
            <p className="text-xs text-slate-400 mb-6 max-w-[180px]">Upload research papers to start your analysis</p>
            <Button
              className="w-full bg-white hover:bg-slate-50 border-slate-200 text-slate-700 shadow-sm"
              variant="outline"
              onClick={handleButtonClick}
              disabled={uploadFileMutation.isPending}
            >
              <UploadIcon size={16} className="mr-2" />
              Upload PDF
            </Button>
          </div>
        )}

        {files && files.map((file) => {
          const isRemoving = deleteFileMutation.isPending && deleteFileMutation.variables?.id === file.id;
          const isProcessing = !file.is_processed;

          return (
            <div
              key={file.id}
              className={cn(
                "group relative p-3.5 rounded-2xl flex items-center gap-3 transition-all duration-300",
                selectedFile?.id === file.id 
                  ? "bg-white shadow-md scale-[1.02] ring-1 ring-primary/10 active-glow" 
                  : "hover:bg-white/60 hover:translate-x-1",
                isProcessing ? "grayscale pointer-events-none opacity-60" : "cursor-pointer",
                isRemoving && "opacity-50 pointer-events-none scale-95"
              )}
              onClick={() => !isProcessing && setFile(file)}
            >
              <div className={cn(
                  "p-2.5 rounded-xl transition-colors",
                  selectedFile?.id === file.id ? "bg-primary text-white" : "bg-slate-200/50 text-slate-500 group-hover:bg-white"
              )}>
                {isRemoving ? <Loader2 size={18} className="animate-spin" /> : <File size={18} />}
              </div>
              
              <div className="flex-grow min-w-0">
                <div className="flex items-center gap-1.5">
                  <p className={cn(
                      "text-[13px] font-semibold truncate transition-colors",
                      selectedFile?.id === file.id ? "text-slate-900" : "text-slate-700"
                  )}>
                    {file.name.replace(/\.[^/.]+$/, "")}
                  </p>
                  {isProcessing && (
                    <div className="flex items-center gap-1 font-bold text-[8px] uppercase tracking-tighter text-amber-600/80 bg-amber-50 px-1.5 py-0.5 rounded-full border border-amber-100">
                       <Loader2 size={8} className="animate-spin" />
                       Embedding
                    </div>
                  )}
                </div>
                <p className="text-[11px] text-slate-400 font-medium">
                  {isRemoving ? "Purging storage..." : formatFileSize(file.metadata.size)}
                </p>
              </div>

              {!isRemoving && !isProcessing && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className={cn(
                          "h-8 w-8 rounded-lg transition-opacity",
                          selectedFile?.id === file.id ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                      )}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <MoreHorizontal size={16} />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="glass-dark text-white border-none min-w-[140px]">
                    <DropdownMenuItem
                      className="text-red-400 focus:bg-red-500/10 focus:text-red-400 cursor-pointer p-2.5 rounded-lg"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteFileMutation.mutate(file);
                      }}
                    >
                      <Trash2 size={15} className="mr-2" />
                      <span className="text-[13px] font-medium">Remove Paper</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </div>
          );
        })}
        
        {uploadFileMutation.isPending && (
          <div className="p-3.5 rounded-2xl bg-primary/5 border border-primary/10 animate-pulse flex items-center gap-3">
             <div className="p-2.5 bg-primary/20 rounded-xl text-primary">
                <Loader2 size={18} className="animate-spin" />
             </div>
             <div className="flex-grow">
                <div className="h-3 w-24 bg-primary/20 rounded-full mb-1.5" />
                <div className="h-2 w-12 bg-primary/10 rounded-full" />
             </div>
          </div>
        )}
      </div>
    </div>
  );
}
