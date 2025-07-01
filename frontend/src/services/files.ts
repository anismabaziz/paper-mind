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
  file: FileType;
}
export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return (
    await client.post<IUploadFile>("/upload", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    })
  ).data;
}

interface IDeleteFile {
  message: string;
}
export async function deleteFile(file: FileType) {
  return (
    await client.delete<IDeleteFile>("/files/remove", {
      params: {
        path: file.name,
      },
    })
  ).data;
}

interface ICheckIsProcessed {
  is_processed: boolean;
}

export async function checkIsProcessed(file: FileType) {
  return (
    await client.post<ICheckIsProcessed>("/file/is-processed", {
      filename: file.name,
    })
  ).data;
}

interface IProcessFile {
  message: string;
}
export async function processFile(file: FileType) {
  return (
    await client.post<IProcessFile>("/process-file", {
      filename: file.name,
    })
  ).data;
}
