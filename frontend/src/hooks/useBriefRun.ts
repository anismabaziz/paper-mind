import { useEffect, useRef, useState } from "react";
import {
  BriefRequestError,
  briefStream,
  cancelBrief,
  loadBrief,
  saveBrief,
} from "@/services/research";
import { NO_BUDGET, type Run } from "@/features/brief/brief-run";

/**
 * One brief's run: started, streamed, stopped, and kept across a reload.
 *
 * Stopping asks the server to finish rather than dropping the connection, so a
 * reader who presses stop keeps what the brief found. The connection is only
 * closed as a fallback, for a server that never answers, because a spinner
 * that never resolves is worse than a partial result.
 */
export function useBriefRun({
  question,
  documents,
  canStart,
}: {
  question: string;
  documents: string[];
  canStart: boolean;
}) {
  const [run, setRun] = useState<Run>({ state: "idle" });
  const [failure, setFailure] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  // Restore the last finished brief so its claims, evidence, status, prompt
  // version, and model survive a reload.
  useEffect(() => {
    const saved = loadBrief();
    if (!saved) return;
    setRun({
      state: "done",
      status: saved.run.status,
      answer: saved.run.answer,
      stoppedBy: saved.run.stoppedBy,
      message: saved.run.message,
      evidence: saved.run.evidence ?? [],
      budget: saved.run.budget,
      documents: saved.run.scope ?? [],
      brief: saved.run.brief ?? null,
      claims: saved.run.claims ?? [],
      gaps: saved.run.gaps ?? [],
      abstained: saved.run.abstained ?? saved.run.brief?.abstained ?? false,
      promptVersion: saved.run.promptVersion ?? null,
      model: saved.run.model ?? null,
    });
  }, []);

  const start = async () => {
    const body = question.trim();
    if (!canStart || !body) return;
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setFailure(null);
    setRun({ state: "running", documents: [], briefId: null });

    try {
      await briefStream(
        body,
        [...documents],
        {
          onStart: (opened) =>
            setRun({ state: "running", documents: opened.documents, briefId: opened.briefId }),
          onDone: (done) => {
            const finished = {
              state: "done" as const,
              status: done.status,
              answer: done.answer,
              stoppedBy: done.stoppedBy,
              message: done.message,
              evidence: done.evidence ?? [],
              budget: done.budget,
              documents: done.documents ?? [],
              brief: done.brief ?? null,
              claims: done.claims ?? [],
              gaps: done.gaps ?? [],
              abstained: done.abstained ?? done.brief?.abstained ?? false,
              promptVersion: done.promptVersion ?? null,
              model: done.model ?? null,
            };
            setRun(finished);
            saveBrief({
              question: body,
              documents: [...documents],
              run: {
                status: finished.status,
                answer: finished.answer,
                stoppedBy: finished.stoppedBy,
                message: finished.message,
                evidence: finished.evidence,
                budget: finished.budget,
                scope: finished.documents,
                brief: finished.brief,
                claims: finished.claims,
                gaps: finished.gaps,
                abstained: finished.abstained,
                promptVersion: finished.promptVersion,
                model: finished.model,
              },
              savedAt: new Date().toISOString(),
            });
          },
          onAbstained: (result) =>
            setRun({
              state: "abstained",
              message: result.message,
              documents: result.documents,
              stoppedBy: result.stoppedBy,
            }),
          onProviderError: (providerFailure) =>
            setRun({
              state: "failed",
              message: providerFailure.message,
              evidence: providerFailure.evidence,
              documents: providerFailure.documents,
            }),
        },
        { signal: controller.signal },
      );
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        // The reader stopped the brief. The server answers that with the
        // evidence it had collected; this branch only runs when the connection
        // itself went away, and says so rather than inventing a result.
        setRun((current) =>
          current.state === "running"
            ? {
                state: "done",
                status: "incomplete",
                answer: "",
                stoppedBy: "cancelled",
                message: "You stopped this brief, so it was not finished.",
                evidence: [],
                budget: NO_BUDGET,
                documents: current.documents,
                brief: null,
                claims: [],
                gaps: [],
                abstained: false,
                promptVersion: null,
                model: null,
              }
            : current,
        );
        return;
      }
      setFailure(
        error instanceof BriefRequestError || error instanceof Error
          ? error.message
          : "The brief could not be read. Try again.",
      );
      setRun({ state: "idle" });
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  };

  const stop = async () => {
    const briefId = run.state === "running" ? run.briefId : null;
    if (briefId) {
      try {
        await cancelBrief(briefId);
        return;
      } catch {
        // The run may already have ended between the reader pressing stop and
        // this arriving; the abort below is the fallback either way.
      }
    }
    controllerRef.current?.abort();
  };

  return { run, failure, start, stop, running: run.state === "running" };
}