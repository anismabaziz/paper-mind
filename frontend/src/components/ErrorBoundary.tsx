import { Component, type ReactNode } from "react";
import { FailureNotice } from "./FailureNotice";
import { getErrorMessage } from "@/lib/api-error";

/**
 * Catches a render crash anywhere below it and shows a retry instead of a
 * blank page. Request failures keep their inline notices; this is only for
 * the app itself breaking while rendering.
 */
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: unknown }
> {
  state = { error: null as unknown };

  static getDerivedStateFromError(error: unknown) {
    return { error };
  }

  componentDidCatch(error: unknown) {
    console.error(error);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-background p-6">
          <div className="w-full max-w-md">
            <FailureNotice
              title="Something went wrong"
              message={getErrorMessage(
                this.state.error,
                "The app hit an unexpected error while rendering.",
              )}
              actionLabel="Try again"
              onAction={() => this.setState({ error: null })}
              testId="app-error"
              actionTestId="app-error-retry"
            />
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
