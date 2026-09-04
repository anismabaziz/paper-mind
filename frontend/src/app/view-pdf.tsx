import usePdfStore from "@/store/pdf-state";
import { File, Trash2, MoreHorizontal, UploadIcon } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { checkIsProcessed, deleteFile } from "@/services/files";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { cn } from "@/lib/utils";

export default function ViewPDF() {
  const { file } = usePdfStore();
  const queryClient = useQueryClient();

  const deleteFileMutation = useMutation({
    mutationFn: deleteFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
  });

  const checkProcessedQuery = useQuery({
    queryKey: [file?.name, "is-processed"],
    queryFn: () => checkIsProcessed(file!),
    enabled: !!file
  });

  return (
    <div className="flex flex-col h-full glass rounded-xl overflow-hidden shadow-sm">
      <div className="h-[52px] px-4 border-b border-slate-200/80 bg-slate-50 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <File size={16} className="text-slate-600" />
          <h3 className="font-semibold text-xs uppercase tracking-wider text-slate-700 truncate max-w-[200px]">
            {file ? file.name.replace(/\.[^/.]+$/, "") : "Document Viewer"}
          </h3>
        </div>

        {file && (
          <div className="flex items-center gap-2">
            <span className={cn(
              "px-2 py-0.5 rounded text-[9px] font-semibold uppercase tracking-wider border",
              checkProcessedQuery.data?.is_processed 
                ? "bg-slate-100 text-slate-600 border-slate-200" 
                : "bg-slate-50 text-slate-400 border-slate-200 animate-pulse"
            )}>
              {checkProcessedQuery.data?.is_processed ? "Indexed" : "Indexing"}
            </span>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-7 w-7 rounded">
                  <MoreHorizontal size={14} />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="bg-white text-slate-800 border border-slate-200/80 shadow-md min-w-[140px] p-1 rounded-md z-50">
                <DropdownMenuItem
                  className="text-red-600 focus:bg-red-50 focus:text-red-700 cursor-pointer p-2 rounded text-xs font-medium flex items-center transition-colors"
                  onClick={() => deleteFileMutation.mutate(file)}
                >
                  <Trash2 size={14} className="mr-2" />
                  <span>Delete Permanently</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}
      </div>

      <div className="flex-grow bg-slate-100 overflow-hidden relative">
        {file ? (
          <div className="h-full w-full flex flex-col">
            <iframe
              src={`${file.url}#toolbar=0&navpanes=0&scrollbar=0`}
              className="w-full h-full border-none bg-white"
              title={file.name}
            ></iframe>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full p-6 text-center">
            <div className="w-12 h-12 bg-white border border-slate-200 rounded flex items-center justify-center mb-4 text-slate-400 shadow-sm">
              <UploadIcon size={20} />
            </div>
            <h4 className="text-sm font-semibold text-slate-700 mb-1.5">No Document Selected</h4>
            <p className="text-xs text-slate-400 max-w-[240px]">
              Select a publication from your library to review its contents in this pane.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
