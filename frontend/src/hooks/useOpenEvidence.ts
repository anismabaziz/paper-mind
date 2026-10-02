import { useCallback } from "react";
import { useFiles } from "@/hooks/useFiles";
import usePdfStore from "@/store/pdf-state";
import useBriefUi from "@/store/brief-ui";
import type { IBriefEvidence } from "@/services/research";

/**
 * Open one evidence Passage in its own Document, at its Page.
 *
 * Opening an evidence Passage switches the reader to the Document it came from,
 * so the brief closes: leaving both on screen at once is what a reader has to
 * do to check a claim, and a half-toggled pair would only confuse them.
 */
export function useOpenEvidence() {
  const setCitationTarget = usePdfStore((state) => state.setCitationTarget);
  const setFile = usePdfStore((state) => state.setFile);
  const closeBrief = useBriefUi((state) => state.close);
  const filesQuery = useFiles();

  return useCallback(
    (item: IBriefEvidence | undefined) => {
      if (!item || item.page == null) return;
      const files = filesQuery.data?.files ?? [];
      const match =
        files.find((file) => file.id === item.document_id) ??
        files.find((file) => file.name === item.document) ??
        null;
      if (match) setFile(match);
      setCitationTarget(item.page);
      closeBrief();
    },
    [closeBrief, filesQuery.data?.files, setCitationTarget, setFile],
  );
}