import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { deleteFile, getFiles, uploadFile } from "@/services/files";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Plus,
  Library,
  UploadIcon,
  File,
  Trash2,
  MoreHorizontal,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import usePdfStore from "@/store/pdf-state";
import { cn } from "@/lib/utils";
import { useEffect } from "react";

export default function ListPdf() {
  const queryClient = useQueryClient();
  const { setFile, file: selectedFile } = usePdfStore();
  const filesQuery = useQuery({ queryKey: ["files"], queryFn: getFiles });
  const uploadFileMutation = useMutation({
    mutationFn: uploadFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
  });
  const deleteFileMutation = useMutation({
    mutationFn: deleteFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
  });

  const files = filesQuery.data?.files;

  useEffect(() => {
    if (files) setFile(files[0]);
  }, [files, setFile]);

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      uploadFileMutation.mutate(file);
    }
  };

  const handleButtonClick = () => {
    document.getElementById("file")?.click();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " bytes";
    else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    else return (bytes / 1048576).toFixed(1) + " MB";
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Library />
          <h3 className="font-medium text-xl">PDF Library</h3>
        </div>
        <Button
          className="cursor-pointer"
          variant={"outline"}
          onClick={handleButtonClick}
          disabled={uploadFileMutation.isPending}
        >
          <Plus strokeWidth={2.5} />
          <span>New PDF</span>
        </Button>
        <Input type="file" hidden onChange={handleFileChange} id="file" />
      </div>

      {/* Display when the pdf list is empty */}
      {files?.length === 0 && (
        <div className="border border-slate-200 flex items-center flex-col gap-4 p-6 rounded-xl text-center">
          <div className="bg-slate-200 rounded-[50%] p-4">
            <Library />
          </div>
          <p className="font-medium text-xl">Your PDF library is empty</p>
          <p className="font-light text-sm">
            Upload your first PDF to start chatting
          </p>
          <Button className="cursor-pointer" variant={"outline"}>
            <UploadIcon strokeWidth={2.5} />
            <span>Upload PDF</span>
          </Button>
        </div>
      )}

      {/* display list of files when there are files */}
      {files && (
        <div className="space-y-2">
          {files.map((file) => {
            return (
              <div
                key={file.id}
                className={cn(
                  "p-3 rounded-md flex items-center gap-2 cursor-pointer justify-between group overflow-hidden",
                  selectedFile?.id == file.id && "bg-slate-100"
                )}
                onClick={() => setFile(file)}
              >
                <div className="bg-slate-200 p-2 rounded-md w-fit">
                  <File />
                </div>
                <div className="overflow-hidden">
                  <p className="font-medium truncate">{file.name}</p>
                  <p className="text-xs">
                    {formatFileSize(file.metadata.size)}
                  </p>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 opacity-0 group-hover:opacity-100 cursor-pointer"
                    >
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem
                      className="text-red-600 dark:text-red-400 focus:text-red-600 dark:focus:text-red-400 cursor-pointer"
                      onClick={() => deleteFileMutation.mutate(file)}
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
