import client from "./client";
import { File as FileType } from "@/types/db";

interface IGetFiles {
  files: FileType[];
}
export async function getFiles() {
  return (await client.get<IGetFiles>("/files")).data;
}

interface IUploadFile {
  message: string;
}
export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return client.post<IUploadFile>("/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
}
