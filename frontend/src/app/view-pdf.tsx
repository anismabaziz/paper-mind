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
    <div className="flex flex-col h-full glass rounded-3xl overflow-hidden shadow-xl border-white/40">
      <div className="p-5 border-b border-white/20 bg-white/20 backdrop-blur-md flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-600">
            <File size={20} />
          </div>
          <h3 className="font-bold text-slate-800 truncate max-w-[200px]">
            {file ? file.name.replace(/\.[^/.]+$/, "") : "Document"}
          </h3>
        </div>

        {file && (
          <div className="flex items-center gap-2">
            <span className={cn(
                "px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider",
                checkProcessedQuery.data?.is_processed 
                    ? "bg-emerald-100 text-emerald-600 border border-emerald-200" 
                    : "bg-amber-100 text-amber-600 border border-amber-200"
            )}>
              {checkProcessedQuery.data?.is_processed ? "Analyzed" : "Pending"}
            </span>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8 rounded-lg">
                  <MoreHorizontal size={16} />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="glass-dark text-white border-none min-w-[140px]">
                <DropdownMenuItem
                  className="text-red-400 focus:bg-red-500/10 focus:text-red-400 cursor-pointer p-2.5 rounded-lg"
                  onClick={() => deleteFileMutation.mutate(file)}
                >
                  <Trash2 size={15} className="mr-2" />
                  <span className="text-[13px] font-medium">Delete Permanently</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        )}
      </div>

      <div className="flex-grow bg-slate-50/30 overflow-hidden relative">
        {file ? (
          <div className="h-full w-full flex flex-col">
            <iframe
              src={`${file.url}#toolbar=0&navpanes=0&scrollbar=0`}
              className="w-full h-full border-none"
              title={file.name}
            ></iframe>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full p-10 text-center">
            <div className="w-20 h-20 bg-slate-100 rounded-3xl flex items-center justify-center mb-6 text-slate-300">
                <UploadIcon size={40} />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-2">Immersive Viewer</h3>
            <p className="text-sm text-slate-500 max-w-[240px]">
               Select a research paper from your library to visualize it here in high fidelity.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
