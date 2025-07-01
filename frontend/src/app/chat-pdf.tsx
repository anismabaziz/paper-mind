import { MessageSquare, UploadIcon, Send, ChevronRight } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import usePdfStore from "@/store/pdf-state";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { checkIsProcessed } from "@/services/files";

export default function ChatPDF() {
  const { file } = usePdfStore();
  const questions = [
    "What is the main topic of this document?",
    "Summarize the key points in 3 bullet points",
  ];
  const [inputValue, setInputValue] = useState("");

  const checkProcessedQuery = useQuery({
    queryKey: [file?.name, "is-processed"],
    queryFn: () => checkIsProcessed(file!),
  });

  useEffect(() => {
    if (!file) setInputValue("");
  }, [file]);

  return (
    <div className="border rounded-md flex flex-col">
      <div className="flex items-center gap-4 p-4 border-b rounded-md">
        <MessageSquare />
        <h2 className="text-xl font-semibold">Chat</h2>
      </div>
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

      {file && !checkProcessedQuery.data?.is_processed && (
        <div className="p-4 flex-grow">Process File</div>
      )}

      {file && checkProcessedQuery.data?.is_processed && (
        <div className="p-4 flex flex-grow">
          <div className="rounded-md border w-full flex flex-col justify-center items-center gap-4">
            <div className="bg-slate-200 rounded-[50%] p-4">
              <MessageSquare size={40} />
            </div>
            <p className="font-medium text-2xl">Start the conversation</p>
            <p className="font-light text-md w-2/3 text-center ">
              Ask questions and get AI-powered answers
            </p>
            <div className="bg-slate-100 p-4 rounded-md">
              <p className="mb-4 text-sm">Example questions:</p>
              <div className="space-y-3 pl-4">
                {questions.map((question) => {
                  return (
                    <div
                      className="flex items-center gap-2 cursor-pointer hover:bg-slate-200 p-2 rounded-md"
                      onClick={() => setInputValue(question)}
                    >
                      <ChevronRight />
                      <p className="text-sm">{question}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex items-center gap-4 p-4 border-t rounded-md">
        <Input
          placeholder={
            file ? "Ask about your document...." : "Select a document first...."
          }
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          className="h-12"
          disabled={!file}
        />
        <Button className="cursor-pointer h-12 w-12" disabled={!file}>
          <Send />
        </Button>
      </div>
    </div>
  );
}
