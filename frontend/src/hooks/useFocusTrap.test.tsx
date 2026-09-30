import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useRef } from "react";
import { afterEach, describe, expect, it } from "vitest";
import { useFocusTrap } from "./useFocusTrap";

function Harness({ active = true }: { active?: boolean }) {
  const containerRef = useRef<HTMLDivElement>(null);
  useFocusTrap(active, containerRef);
  return (
    <div>
      <button type="button">outside</button>
      <div ref={containerRef} tabIndex={-1} data-testid="trap">
        <button type="button">first</button>
        <button type="button">second</button>
      </div>
    </div>
  );
}

afterEach(cleanup);

describe("useFocusTrap", () => {
  it("moves focus inside the container on open", () => {
    render(<Harness />);
    expect(screen.getByTestId("trap")).toHaveFocus();
  });

  it("wraps Tab from the last control back to the first", () => {
    render(<Harness />);
    const first = screen.getByRole("button", { name: "first" });
    const second = screen.getByRole("button", { name: "second" });
    second.focus();
    fireEvent.keyDown(document, { key: "Tab", code: "Tab" });
    expect(first).toHaveFocus();
  });

  it("wraps Shift+Tab from the first control to the last", () => {
    render(<Harness />);
    const first = screen.getByRole("button", { name: "first" });
    const second = screen.getByRole("button", { name: "second" });
    first.focus();
    fireEvent.keyDown(document, { key: "Tab", code: "Tab", shiftKey: true });
    expect(second).toHaveFocus();
  });

  it("restores focus to the invoking control when closed", () => {
    const { rerender } = render(<Harness active={false} />);
    screen.getByRole("button", { name: "outside" }).focus();
    rerender(<Harness active />);
    expect(screen.getByTestId("trap")).toHaveFocus();
    rerender(<Harness active={false} />);
    expect(screen.getByRole("button", { name: "outside" })).toHaveFocus();
  });
});
