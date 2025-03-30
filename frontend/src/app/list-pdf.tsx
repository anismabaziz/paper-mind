import { Button } from "@/components/ui/button";
import { getFiles } from "@/services/files";
import { useQuery } from "@tanstack/react-query";
import { Plus, Library, UploadIcon, File } from "lucide-react";

export default function ListPdf() {
  const filesQuery = useQuery({ queryKey: ["files"], queryFn: getFiles });
  const files = filesQuery.data?.files;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Library />
          <h3 className="font-medium text-xl">PDF Library</h3>
        </div>
        <Button className="cursor-pointer" variant={"outline"}>
          <Plus strokeWidth={2.5} />
          <span>New PDF</span>
        </Button>
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
      {files && (
        <div>
          {files.map((file) => {
            return (
              <div
                key={file.id}
                className="bg-slate-100 p-4 rounded-md flex items-center gap-2"
              >
                <div className="bg-slate-200 p-2 rounded-md w-fit">
                  <File />
                </div>
                <div>
                  <p className="font-medium">{file.name.slice(0, 20)}</p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
