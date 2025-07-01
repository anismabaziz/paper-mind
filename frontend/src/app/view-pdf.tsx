import usePdfStore from "@/store/pdf-state";
import { File, Trash2, MoreHorizontal, UploadIcon } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { checkIsProcessed, deleteFile, processFile } from "@/services/files";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

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
  });

  console.log(checkProcessedQuery.data);

  const processFileMutation = useMutation({
    mutationFn: processFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [file?.name, "is-processed"] });
    },
  });

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " bytes";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    else return (bytes / 1048576).toFixed(1) + " MB";
  };

  return (
    <div className="border rounded-md h-[80vh] flex flex-col">
      <div className="flex items-center justify-between p-4 border-b rounded-md">
        <div className="flex items-center gap-4">
          <File />
          <h2 className="text-xl font-semibold">Document Viewer</h2>
        </div>

        <Button
          disabled={checkProcessedQuery.data?.is_processed}
          onClick={() => {
            processFileMutation.mutate(file!);
          }}
        >
          {checkProcessedQuery.data?.is_processed
            ? "Already Processed"
            : "Process File"}
        </Button>
      </div>
      {file && (
        <div className="p-4 h-full flex flex-col">
          <div
            key={file.id}
            className="py-4 pr-1 rounded-md flex items-center gap-2 cursor-pointer justify-between overflow-hidden"
          >
            <div className="bg-slate-200 p-2 rounded-md w-fit">
              <File />
            </div>
            <div className="overflow-hidden">
              <p className="font-medium truncate">{file.name}</p>
              <p className="text-xs">{formatFileSize(file.metadata.size)}</p>
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <MoreHorizontal className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  className="text-red-600 dark:text-red-400 focus:text-red-600 dark:focus:text-red-400"
                  onClick={() => deleteFileMutation.mutate(file)}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <iframe
            src={`${file.url}#toolbar=0&navpanes=0&scrollbar=0`}
            className="w-full h-full border rounded-md flex-grow"
          ></iframe>
        </div>
      )}
      {!file && (
        <div className="p-4 flex flex-grow">
          <div className="rounded-md border w-full flex flex-col justify-center items-center gap-4">
            <div className="bg-slate-200 rounded-[50%] p-6">
              <UploadIcon size={50} />
            </div>
            <p className="font-medium text-2xl">No document selected</p>
            <p className="font-light text-md w-2/3 text-center ">
              Upload a PDF document to start chatting with AI about its contents
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
